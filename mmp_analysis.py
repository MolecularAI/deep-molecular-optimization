import pandas as pd
import os
import pickle
import datetime
import random
import math
from collections import Counter
import argparse
import subprocess
from pandas.io.common import EmptyDataError

import rdkit.Chem as rkc
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS

import configuration.opts as opts
import utils.log as ul
import utils.file as uf
import utils.chem as uc

nofcuts = 1
hac, ratio = 40, 0.8
ratio_list = [33, 50]
topk = 20

def prepare_mmp_input(data_test, path, num_samples, is_only_desirable=False):
    mmp_input_mol, mmp_input_ID = [], []
    for j in range(len(data_test)):
        source_id = data_test.iloc[j]['Source_Mol_ID']
        if source_id not in mmp_input_ID:
            mmp_input_mol.append(data_test.iloc[j]['Source_Mol'])
            mmp_input_ID.append(data_test.iloc[j]['Source_Mol_ID'])
        for i in range(num_samples):
            # generated mol and id
            smi = data_test.iloc[j]['Predicted_smi_{}'.format(i + 1)]
            identifier = data_test.iloc[j]['FakeID_{}'.format(i + 1)]
            # if valid smile
            if rkc.MolFromSmiles(str(smi)) and identifier not in mmp_input_ID:
                if is_only_desirable:
                    if str(data_test.iloc[j][f'Predict_eval_{i+1}_allprop']) == '1':
                        mmp_input_mol.append(smi)
                        mmp_input_ID.append(identifier)
                else:
                    mmp_input_mol.append(smi)
                    mmp_input_ID.append(identifier)

    # save to file for mmp input
    df_output = pd.DataFrame(list(zip(mmp_input_mol, mmp_input_ID)),
                             columns=['smi', 'id'])
    out_file = os.path.join(path, 'mmp_input.smi')
    df_output.to_csv(out_file, index=False, header=False)

    return out_file

def remove_duplicated_transformations_id(data_removenan):
    data_removenan['Source_R_len'] = data_removenan['Transformation'].apply(len)
    data_removenan = data_removenan.sort_values('Source_R_len')
    data_removenan_duplicated_transform = data_removenan.drop_duplicates(subset=['Source_Mol_ID', 'Target_Mol_ID'])
    LOG.info(
        "After removing duplicates transformations: {}, {} missing".format(len(data_removenan_duplicated_transform),
                                                                           len(data_removenan) - len(
                                                                               data_removenan_duplicated_transform)))
    return data_removenan_duplicated_transform

def keep_source_predict_pair(result_dup_id):
    # Remove pairs with predict and predict molecules
    data = result_dup_id[(result_dup_id['Source_Mol_ID'].str.contains('CHEMBL')) & (
        result_dup_id['Target_Mol_ID'].str.contains('Predict'))]
    LOG.info("After removing pred-pred pairs: {}, {} missing".format(len(data), len(result_dup_id) - len(data)))
    return data


def get_mmp_df_from_file(input_path):
    try:
        results = pd.read_csv(input_path, sep='\t', header=None)
    except EmptyDataError:
        results = pd.DataFrame()
    if len(results) > 0:
        results.columns = ['Source_Smi', 'Target_Smi', 'Source_Mol_ID', 'Target_Mol_ID', 'Transformation', 'Core']
        # Remove duplicated and pred-pred pairs
        mmp_c_p_pair = keep_source_predict_pair(results)
        mmp_all_df = remove_duplicated_transformations_id(mmp_c_p_pair)

        return mmp_all_df
    else:
        return None

def _num_heavy_atoms(smi):
    if smi is not None and smi != "":
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return mol.GetNumHeavyAtoms()
        elif smi == 'H':
            return 0
        else:
            LOG.info("Mol=None ", smi)


def generate_mmps(data_test, nofcuts, path, mmpdb_path, num_samples, is_only_desirable):
    # For every n=batch starting molecule and its generated molecules, do MMPA
    batch = 2000
    all_mmps_df_list = []
    number_of_mmps_per_starting_mol = []
    for j in range(math.ceil(len(data_test) / batch)):
        end = min(len(data_test), j + batch * (j + 1))
        data = data_test[j + batch * j:end]
        # prepare input
        mmp_input_file = prepare_mmp_input(data, path, num_samples, is_only_desirable)

        LOG.info("{} Processing {} to {} in test".format(datetime.datetime.now(), j + batch * j, end))

        fragement_outfile = mmp_input_file.split('.')[0] + '.fragments'
        LOG.info("Fragmenting")
        command = f"python {mmpdb_path}mmpdb fragment {mmp_input_file} --num-cuts {nofcuts} --delimiter='comma' --output {fragement_outfile}"
        LOG.info(command)
        subprocess.run(command, shell=True)
        LOG.info("Done fragmenting")

        index_outfile = fragement_outfile.split('.')[0] + '_indexed.csv'
        LOG.info("Indexing")
        command = f"python {mmpdb_path}mmpdb index {fragement_outfile} --out 'csv' -s --smallest-transformation-only --max-variable-heavies {hac} --output {index_outfile} --max-variable-ratio {ratio}"
        LOG.info(command)
        subprocess.run(command, shell=True)
        LOG.info("Done indexing")

        mmp_df = get_mmp_df_from_file(index_outfile)
        if mmp_df is not None:
            all_mmps_df_list.append(mmp_df)
            number_of_mmps_per_starting_mol.append(len(mmp_df))
        else:
            number_of_mmps_per_starting_mol.append(0)
    # merge all batch's pairs
    all_mmps_df = pd.concat(all_mmps_df_list)
    all_mmps_df = remove_duplicated_transformations_id(all_mmps_df)
    all_mmps_df = all_mmps_df.reset_index(drop=True)
    LOG.info(f"{len(all_mmps_df)} pairs with {nofcuts} cut")

    return all_mmps_df

def analyse_mmp_results(data_test, mmp_all_df, ratio_list, num_samples, tmp_path, save_path, is_only_desirable):
    # (pair) as key, index as value
    keys = zip(mmp_all_df['Source_Mol_ID'].tolist(), mmp_all_df['Target_Mol_ID'].tolist())
    values = list(mmp_all_df.index)
    pair_index_dict = dict(zip(keys, values)) # matched molecular pairs

    # Collect rows with MMPs 
    not_mmp_paris = []
    mmp_row_list_dict, mmp_df_list = {}, []
    for ratio in ratio_list:
        mmp_row_list_dict[ratio] = []
    total_valid, not_valid, not_mmps = 0, 0, 0
    for i in range(len(data_test)):
        source_mol_ID = data_test.iloc[i]['Source_Mol_ID']
        for j in range(1, 1+num_samples):
            predict_ID = data_test.iloc[i]['FakeID_{}'.format(j)]
            predict_smi = data_test.iloc[i]['Predicted_smi_{}'.format(j)]
            pair = (source_mol_ID, predict_ID)
            if uc.is_valid(predict_smi):
                if is_only_desirable:
                    if str(data_test.iloc[i][f'Predict_eval_{j}_allprop']) == '1':
                        total_valid += 1
                        if pair in pair_index_dict:
                            row = mmp_all_df.iloc[pair_index_dict[pair]]
                            for ratio in ratio_list:
                                if _num_heavy_atoms(row['Transformation'].split('>>')[1]) * 1.0 / _num_heavy_atoms(
                                        row['Target_Smi']) <= ratio / 100.0:
                                    mmp_row_list_dict[ratio].append(row)
                        else:
                            not_mmps += 1
                            pair_smi = (data_test.iloc[i]['Source_Mol'], predict_smi)
                            not_mmp_paris.append(pair_smi)
                else:
                    total_valid += 1
                    if pair in pair_index_dict:
                        row = mmp_all_df.iloc[pair_index_dict[pair]]
                        for ratio in ratio_list:
                            if _num_heavy_atoms(row['Transformation'].split('>>')[1]) * 1.0 / _num_heavy_atoms(
                                    row['Target_Smi']) <= ratio / 100.0:
                                mmp_row_list_dict[ratio].append(row)
                    else:
                        not_mmps += 1
                        pair_smi = (data_test.iloc[i]['Source_Mol'], predict_smi)
                        not_mmp_paris.append(pair_smi)
            else:
                not_valid += 1

    for ratio in ratio_list:
        mmp_df_result = pd.DataFrame(mmp_row_list_dict[ratio], columns=mmp_all_df.columns)
        mmp_df_list.append(mmp_df_result)

    with open(os.path.join(tmp_path, "mmps_list_analysed.pkl"), "wb") as output_file:
        pickle.dump(mmp_df_list, output_file)
    with open(os.path.join(tmp_path, "not_mmp_pairs.pkl"), "wb") as output_file:
        pickle.dump(not_mmp_paris, output_file)

    if is_only_desirable:
        LOG.info(
            f"Ideally should generate {len(data_test)*num_samples} molecules with desirable properties for {len(data_test)} starting molecules")
        LOG.info(f"In fact generate {total_valid}({round(total_valid/(len(data_test)*num_samples)*100, 2)}%)")
    else:
        LOG.info(f"Theoretically should generate {len(data_test)*num_samples} molecules for {len(data_test)} starting molecules")
        LOG.info(f"In fact generate {total_valid}({round(total_valid/(len(data_test)*num_samples)*100, 2)}%)")

    for i, df in enumerate(mmp_df_list):
        LOG.info(f"===============Ratio={ratio_list[i]}================")
        if is_only_desirable:
            LOG.info(
                f"{round(len(df)/total_valid*100, 2)}% of generated molecules with desirable properties that are MMPs with starting molecules and " +
                f"the ratio of heavy atoms in R group to the generated molecule is not greater than {ratio_list[i]/100}")
        else:
            LOG.info(
                f"{round(len(df)/total_valid*100, 2)}% of generated molecules that are MMPs with starting molecules and " +
                f"the ratio of heavy atoms in R group to the generated molecule is not greater than {ratio_list[i]/100}")
        LOG.info(
            f"{round(len(set(df['Transformation']))/len(df)*100, 2)}% unique transformations out of {len(df)}")

        LOG.info(f"Top {topk} frequently occured transformations generated:")
        x = Counter(df['Transformation'].tolist())
        topk_transformations = x.most_common(topk)
        LOG.info(topk_transformations)
        draw_transformations(topk_transformations, save_path, f"top_{topk}_generated_{ratio_list[i]}")
    return mmp_df_list, not_mmp_paris, total_valid

def transformations_in_train(train_path, mmp_df_list, total_valid):
    # How many in Train and not in Train
    train = pd.read_csv(os.path.join(train_path), sep=",")
    train_transformations = set(train['Transformation'].tolist())
    LOG.info(f"Number of samples in Train: {len(train)}")
    LOG.info(f"Number of unique transformations in Train: {len(train_transformations)}")

    n_exist_transf_in_train = 0
    not_exist_transf_list = []
    exist_unique_list = []
    test_transformations_df = mmp_df_list
    for i in range(len(test_transformations_df)):
        row = test_transformations_df.iloc[i]
        t = row['Transformation']
        if t in train_transformations:
            n_exist_transf_in_train += 1
            exist_unique_list.append(t)
        else:
            not_exist_transf_list.append(t)
    LOG.info("Perc. of generated molecules out of MMPs whose transformations are in Train: {}%".format(
        round(n_exist_transf_in_train / len(test_transformations_df) * 100, 2)))
    LOG.info("Perc. of generated molecules out of all generated whose transformations are in Train: {}%".format(
        round(n_exist_transf_in_train / total_valid * 100, 2)))

    return not_exist_transf_list

def transformations_not_in_train(not_exist_transf_list, save_path):
    # Check MMPs not in Train
    x = Counter(not_exist_transf_list)
    topk_transformations = x.most_common(topk)
    LOG.info(f"Top {topk} frequent occured transformations not in Train:")
    LOG.info(topk_transformations)
    draw_transformations(topk_transformations, save_path, 'transformations_not_in_train')

def draw_transformations(transformations, save_path, name):
    mols = []
    for t in transformations:
        s_t = t[0].split(">>")
        try:
            mols.append(Chem.MolFromSmiles(s_t[0]))
            mols.append(Chem.MolFromSmiles(s_t[1]))
        except Exception:
            LOG.info(f"Can't draw {s_t}")
            pass
    image = Draw.MolsToGridImage(mols, molsPerRow=2)
    image.save(os.path.join(save_path, f'{name}.png'), format='png')

def draw_not_mmps(not_mmp_paris, save_path):
    mols = []
    matches_list = []
    SEED = 42
    random.seed(SEED)
    sampled_index = random.sample(list(range(len(not_mmp_paris))), topk)
    for i in sampled_index:
        pairs = not_mmp_paris[i]
        curr_mols = [Chem.MolFromSmiles(pairs[0]), Chem.MolFromSmiles(pairs[1])]
        mols.append(Chem.MolFromSmiles(pairs[0]))
        mols.append(Chem.MolFromSmiles(pairs[1]))

        res = rdFMCS.FindMCS(curr_mols)
        patt = Chem.MolFromSmarts(res.smartsString)
        for mol in curr_mols:
            matches = mol.GetSubstructMatches(patt)
            not_matches = tuple(tuple(set(range(len(mol.GetAtoms()))) - set(matches[0])))
            matches_list.append(not_matches)
    image = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(400, 400), highlightAtomLists=matches_list)
    image.save(os.path.join(save_path, f'{name}.png'), format='png')

def perform_mmp_analysis(data_path, train_path, temp_files_path, save_path, mmpdb_path, num_samples, is_only_desirable):
    # Read evaluation file containing generated smiles
    data_test = pd.read_csv(data_path)
    LOG.info(f"Number of starting molecules in test file: {len(data_test)}")

    mmp_all_df = generate_mmps(data_test, nofcuts, temp_files_path, mmpdb_path, num_samples, is_only_desirable)
    mmp_df_list, not_mmp_paris, total_valid = analyse_mmp_results(data_test, mmp_all_df, ratio_list, num_samples,
                                                                  temp_files_path, save_path, is_only_desirable)
    not_exist_transf_list = transformations_in_train(train_path, mmp_df_list[0], total_valid)
    transformations_not_in_train(not_exist_transf_list, save_path)

def run_main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='mmp_analysis.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.evaluation_opts(parser)
    opt = parser.parse_args()

    save_path = uf.get_parent_dir(opt.data_path)
    global LOG
    LOG = ul.get_logger(name="mmp_analysis", log_path=os.path.join(save_path, 'mmp_analysis.log'))
    LOG.info(opt)
    if opt.only_desirable:
        temp_files_path = os.path.join(save_path, 'temp_files', 'MMP_desirable')
    else:
        temp_files_path = os.path.join(save_path, 'temp_files', 'MMP')
    uf.make_directory(temp_files_path, is_dir=True)

    perform_mmp_analysis(opt.data_path, opt.train_path, temp_files_path, save_path, opt.mmpdb_path, opt.num_samples,
                         opt.only_desirable)


if __name__ == "__main__":
    run_main()
