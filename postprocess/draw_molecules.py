import random
import numpy as np
from PIL import ImageDraw
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS

import configuration.config_default as cfgd

SEED = 0
IMG_SIZE = 300
MOLS_PER_ROW = 6
NUM_SAMPLES = 10


def get_plot_sample(df_predictions, nr_of_source_mol=50, range_evaluation=""):
    # Sample random indices
    random.seed(SEED)
    nr_of_source_mol = min(nr_of_source_mol, len((df_predictions)))
    sampled_index = random.sample(list(range(len(df_predictions))), nr_of_source_mol)
    molecules, green_boxes, red_boxes, all_tuples_mol, matches_list = _create_boxes_and_molecules(
        df_predictions, sampled_index, nr_of_source_mol
    )

    # Get legends
    legends = _get_legends(df_predictions, molecules, all_tuples_mol, sampled_index, range_evaluation)

    img = Draw.MolsToGridImage(
        molecules,
        molsPerRow=MOLS_PER_ROW,
        subImgSize=(IMG_SIZE, IMG_SIZE),
        legends=legends,
        highlightAtomLists=matches_list
    )

    # Add boxes and additional text
    for i in range(nr_of_source_mol):
        # Add text
        img = _add_property_threshold_test(img, i, range_evaluation)
        img = _add_source_target_text(img, i, range_evaluation, legends)
        img = _add_generated_mol_text(img, i)

        # Add boxes
        img = _add_box_around_coherent_mol(img, i)

        # Draw green boxes around molecules satisfying whole delta vector
        img, last_index = _add_green_boxes(img, i, green_boxes[i], all_tuples_mol)

        img, last_index = _add_red_boxes_option_1(
            img, i, nr_boxes=int(red_boxes[i][0]), previous_index=last_index
        )
        img, last_index = _add_red_boxes_option_2(
            img, i, nr_boxes=int(red_boxes[i][1]), previous_index=last_index
        )
        img = _add_red_boxes_option_3(
            img, i, nr_boxes=int(red_boxes[i][2]), previous_index=last_index
        )

        img = _add_red_LogD_boxes(img, i, all_tuples_mol)

    return img


def _add_red_LogD_boxes(img, i, all_gen_mols):
    # Create red boxes for unsatisfied logD

    shift_x = 2 * IMG_SIZE
    shift_y = 2 * i * IMG_SIZE
    x_bias = 40
    y_bias = 273
    width = 80
    height = 22  # All manually chosen to fit picture
    color = "Red"

    for l in range(NUM_SAMPLES):
        # Check if new row
        if (l + 2) == MOLS_PER_ROW:
            shift_y += IMG_SIZE
            shift_x = 0

        # Check if delta_logd is above threshold
        if all_gen_mols[i][l][1][0]:
            # Draw rectangle
            upper_left = (shift_x + x_bias, shift_y + y_bias)
            upper_right = (shift_x + x_bias + width, shift_y + y_bias)
            lower_right = (shift_x + x_bias + width, shift_y + y_bias + height)
            lower_left = (shift_x + x_bias, shift_y + y_bias + height)
            ImageDraw.Draw(img).polygon(
                [upper_left, upper_right, lower_right, lower_left], outline=color
            )
        # Shift coordinates
        shift_x += IMG_SIZE
    return img


def _add_red_boxes_option_3(img, i, nr_boxes, previous_index):
    # Draw red boxes around non-fulfilled Solubility and Clint
    x_bias = 124
    y_bias = 273
    width = 60 + 79
    height = 22  # All manually chosen to fit picture

    # Calculate previous shifts
    if previous_index <= MOLS_PER_ROW:
        shift_y = 2 * i * IMG_SIZE
        shift_x = previous_index * IMG_SIZE
    else:
        shift_y = IMG_SIZE + 2 * i * IMG_SIZE
        shift_x = (previous_index - MOLS_PER_ROW) * IMG_SIZE
    index = previous_index

    for k in range(index, index + nr_boxes):
        # Check if new row
        if k == MOLS_PER_ROW:
            shift_y += IMG_SIZE
            shift_x = 0

        # Calculate coordinates
        upper_left = (shift_x + x_bias, shift_y + y_bias)
        upper_right = (shift_x + x_bias + width, shift_y + y_bias)
        lower_right = (shift_x + x_bias + width, shift_y + y_bias + height)
        lower_left = (shift_x + x_bias, shift_y + y_bias + height)
        ImageDraw.Draw(img).polygon(
            [upper_left, upper_right, lower_right, lower_left], outline="Red"
        )

        # Shift coordinates
        shift_x += IMG_SIZE
    return img


def _add_red_boxes_option_2(img, i, nr_boxes, previous_index):
    # Draw red boxes around non-fulfilled Solubility
    x_bias = 124
    y_bias = 273
    width = 60
    height = 22  # All manually chosen to fit picture

    # Calculate previous shifts
    if previous_index <= MOLS_PER_ROW:
        shift_y = 2 * i * IMG_SIZE
        shift_x = previous_index * IMG_SIZE
    else:
        shift_y = IMG_SIZE + 2 * i * IMG_SIZE
        shift_x = (previous_index - MOLS_PER_ROW) * IMG_SIZE

    index = previous_index
    for k in range(index, index + nr_boxes):
        # Check if new row
        if k == MOLS_PER_ROW:
            shift_y += IMG_SIZE
            shift_x = 0

        # Calculate coordinates
        upper_left = (shift_x + x_bias, shift_y + y_bias)
        upper_right = (shift_x + x_bias + width, shift_y + y_bias)
        lower_right = (shift_x + x_bias + width, shift_y + y_bias + height)
        lower_left = (shift_x + x_bias, shift_y + y_bias + height)
        ImageDraw.Draw(img).polygon(
            [upper_left, upper_right, lower_right, lower_left], outline="Red"
        )

        # Shift coordinates
        shift_x += IMG_SIZE
        index += 1
    return img, index


def _add_red_boxes_option_1(img, i, nr_boxes, previous_index):
    # Draw red boxes around non-fulfilled Clint

    x_bias = 186
    y_bias = 273
    width = 79
    height = 22  # All manually chosen to fit picture

    # Calculate previous shifts
    if previous_index <= MOLS_PER_ROW:
        shift_y = 0 + 2 * i * IMG_SIZE
        shift_x = previous_index * IMG_SIZE
    else:
        shift_y = IMG_SIZE + 2 * i * IMG_SIZE
        shift_x = (previous_index - MOLS_PER_ROW) * IMG_SIZE

    index = previous_index
    for k in range(index, index + nr_boxes):
        # Check if new row
        if k == MOLS_PER_ROW:
            shift_y += IMG_SIZE
            shift_x = 0

        # Calculate coordinates
        upper_left = (shift_x + x_bias, shift_y + y_bias)
        upper_right = (shift_x + x_bias + width, shift_y + y_bias)
        lower_right = (shift_x + x_bias + width, shift_y + y_bias + height)
        lower_left = (shift_x + x_bias, shift_y + y_bias + height)
        ImageDraw.Draw(img).polygon(
            [upper_left, upper_right, lower_right, lower_left], outline="Red"
        )

        # Shift coordinates
        shift_x += IMG_SIZE
        index += 1
    return img, index


def _add_green_boxes(img, i, green_boxes, all_gen_mols):
    x_bias = 35
    y_bias = 240
    width = 235
    height = 60  # All manually chosen to fit picture

    shift_x = 2 * IMG_SIZE
    shift_y = 2 * i * IMG_SIZE
    index = 2  # Need to start 2 to the right

    for j in range(int(green_boxes)):
        # Check if new row
        if (j + 2) == MOLS_PER_ROW:
            shift_y += IMG_SIZE
            shift_x = 0

        # Check if logD is satisfied. We already know that Solubility and Clint are both satisfied.
        if not all_gen_mols[i][j][1][0]:
            # Calculate coordinates
            upper_left = (shift_x + x_bias, shift_y + y_bias)
            upper_right = (shift_x + x_bias + width, shift_y + y_bias)
            lower_right = (shift_x + x_bias + width, shift_y + y_bias + height)
            lower_left = (shift_x + x_bias, shift_y + y_bias + height)
            ImageDraw.Draw(img).polygon(
                [upper_left, upper_right, lower_right, lower_left], outline="Green"
            )

        # Shift coordinates
        shift_x += IMG_SIZE
        index += 1
    return img, index


def _add_box_around_coherent_mol(img, i):
    x_bias = 15
    y_bias = 30  # Bias manually chosen to fit picture
    ImageDraw.Draw(img).polygon(
        [
            (x_bias, i * IMG_SIZE * 2 + y_bias),
            (IMG_SIZE * (2 + NUM_SAMPLES) // 2 - x_bias, i * IMG_SIZE * 2 + y_bias),
            (
                IMG_SIZE * (2 + NUM_SAMPLES) // 2 - x_bias,
                (i + 2) * IMG_SIZE * 2 + y_bias,
            ),
            (x_bias, (i + 2) * IMG_SIZE * 2 + y_bias),
        ],
        outline="Black",
    )
    return img


def _add_property_threshold_test(img, i, range_evaluation):
    ImageDraw.Draw(img).text(
        (30, 31 + IMG_SIZE * 2 * i), f"General Thresholds", (0, 0, 0)
    )
    if range_evaluation == 'lower':
        ImageDraw.Draw(img).text(
            (160, 31 + IMG_SIZE * 2 * i), f"LogD upper fluctuation {cfgd.PROPERTY_ERROR['LogD']}", (0, 0, 0)
        )
    elif range_evaluation == 'higher':
        ImageDraw.Draw(img).text(
            (160, 31 + IMG_SIZE * 2 * i), f"LogD lower fluctuation {cfgd.PROPERTY_ERROR['LogD']}", (0, 0, 0)
        )
    else:
        ImageDraw.Draw(img).text(
            (160, 31 + IMG_SIZE * 2 * i), f"Required abs(delta-LogD) <= {cfgd.PROPERTY_ERROR['LogD']}", (0, 0, 0)
        )
    ImageDraw.Draw(img).text(
        (160, 45 + IMG_SIZE * 2 * i),
        f"(High) Solubility > {cfgd.PROPERTY_THRESHOLD['Solubility']:.2f}-{cfgd.PROPERTY_ERROR['Solubility']}="
        f"{cfgd.PROPERTY_THRESHOLD['Solubility']-cfgd.PROPERTY_ERROR['Solubility']:.2f},"
        f"(Low) Solubility < {cfgd.PROPERTY_THRESHOLD['Solubility']:.2f}+{cfgd.PROPERTY_ERROR['Solubility']}="
        f"{cfgd.PROPERTY_THRESHOLD['Solubility']+cfgd.PROPERTY_ERROR['Solubility']:.2f}",
        (0, 0, 0),
    )
    ImageDraw.Draw(img).text(
        (160, 61 + IMG_SIZE * 2 * i), f"(High) Clint > {cfgd.PROPERTY_THRESHOLD['Clint']:.2f}-{cfgd.PROPERTY_ERROR['Clint']}="
                                      f"{cfgd.PROPERTY_THRESHOLD['Clint']-cfgd.PROPERTY_ERROR['Clint']:.2f},"
                                      f"(Low) Clint < {cfgd.PROPERTY_THRESHOLD['Clint']:.2f}+{cfgd.PROPERTY_ERROR['Clint']}="
                                      f"{cfgd.PROPERTY_THRESHOLD['Clint'] + cfgd.PROPERTY_ERROR['Clint']:.2f}",
        (0, 0, 0)
    )
    return img


def _add_source_target_text(img, i, range_evaluation, legends):
    x_bias = 128
    y_bias = 250  # Bias manually chosen to fit picture
    coordinates_source = (x_bias, 2 * i * IMG_SIZE + y_bias)
    ImageDraw.Draw(img).text(coordinates_source, "Source", (0, 0, 0))
    coordinates_target = (x_bias + IMG_SIZE, 2 * i * IMG_SIZE + y_bias)
    ImageDraw.Draw(img).text(coordinates_target, "Target", (0, 0, 0))
    if range_evaluation != '':
        coordinates_target_prop = (x_bias + IMG_SIZE - 120, 2 * i * IMG_SIZE + y_bias + 28)
        legend = legends[(NUM_SAMPLES+2) * i + 1]
        ImageDraw.Draw(img).text(coordinates_target_prop, legend, (0, 0, 0))
    return img



def _add_generated_mol_text(img, i):
    # Start from 2 to the right
    shift_x = 2 * IMG_SIZE
    shift_y = 0
    x_bias = 128
    y_bias = 250  # Bias manually chosen to fit picture

    for j in range(NUM_SAMPLES):
        # Check if new row
        if (j + 2) == MOLS_PER_ROW:
            shift_y = IMG_SIZE
            shift_x = 0

        # Calculate new coordinates
        coordinates_target = (x_bias + shift_x, 2 * i * IMG_SIZE + shift_y + y_bias)
        ImageDraw.Draw(img).text(coordinates_target, "Gen " + str(j + 1), (0, 0, 0))
        shift_x += IMG_SIZE
    return img


def _get_legends(predictions, molecules, all_gen_mols, sampled_indices, range_evaluation):

    legends = []
    for i in range(len(molecules) // (NUM_SAMPLES + 2)):
        row = predictions.loc[sampled_indices[i]]

        # Add source and target
        legends.append(
            f"[LogD: {row['Source_Mol_LogD']:.2f}  Sol: {row['Source_Mol_Solubility']:.2f}  Clint:{row['Source_Mol_Clint']:.2f}]"
        )
        if range_evaluation == '':
            legends.append(
                f"[LogD: {row['Target_Mol_LogD']:.2f}  Sol: {row['Delta_Solubility']}  Clint: {row['Delta_Clint']}]"
            )
        elif range_evaluation == 'lower':
            upper_value = min(row['Source_Mol_LogD'], cfgd.LOD_MAX)
            legends.append(
                f"[LogD: {cfgd.LOD_MIN}-{upper_value:.2f}  Sol: {row['Delta_Solubility']}  Clint: {row['Delta_Clint']}]"
            )
        elif range_evaluation == 'higher':
            lower_value = max(row['Source_Mol_LogD'], cfgd.LOD_MIN)
            legends.append(
            f"[LogD: {lower_value:.2f}-{cfgd.LOD_MAX}  Sol: {row['Delta_Solubility']}  Clint: {row['Delta_Clint']}]"
        )

        # Add generated molecules
        for j in range(NUM_SAMPLES):
            index = all_gen_mols[i][j][-2]
            legends.append(
                f"[LogD: {row['Predict_smi_' + str(index) + '_cLogD']:.2f}   Sol: {row['Predict_smi_' + str(index) + '_cSolubility']:.2f}  Clint: {row['Predict_smi_' + str(index) + '_cClint']:.2f}]"
            )
    return legends


def _create_boxes_and_molecules(predictions, sampled_indices, nr_of_source_mol):
    # This function calculates how many green-boxes and red-boxes (and their type) to add to each batch of generated molecules.
    # Initiate arrays to store data in
    green_boxes = np.zeros(nr_of_source_mol)
    red_boxes = np.zeros((nr_of_source_mol, 3))
    smiles = []
    all_gen_mols = []
    matches_all =[]

    for i, sample_idx in enumerate(sampled_indices):
        row = predictions.loc[sample_idx]

        # Fill batch with source, target and all 10 generated molecules
        batch = [row["Source_Mol"], row["Target_Mol"]]
        generated_mols = []
        matches_list = []

        # find maximum common structure
        mols = [Chem.MolFromSmiles(row["Source_Mol"]), Chem.MolFromSmiles(str(row["Target_Mol"]))]
        if mols[1]:
            res = rdFMCS.FindMCS(mols)
            patt = Chem.MolFromSmarts(res.smartsString)
            for mol in mols:
                matches = mol.GetSubstructMatches(patt)
                not_match = tuple(tuple(set(range(len(mol.GetAtoms()))) - set(matches[0])))
                matches_list.append(not_match)
        else:
            matches_list.extend([(), ()])

        for j in range(1, NUM_SAMPLES + 1):
            if row["Predicted_smi_" + str(j)] == 0:
                generated_mols.append("NOSMILE")
            else:
                not_satisfy = 1-row['Predict_eval_{}_{}_{}'.format(j, 'LogD', cfgd.PROPERTY_ERROR['LogD'])]

                save_info = (not_satisfy, row['num_correct_allprop_sumoversample_allerror'])

                solubility_bool = row['Predict_eval_{}_{}_{}'.format(j, 'Solubility', cfgd.PROPERTY_ERROR['Solubility'])]
                clint_bool = row['Predict_eval_{}_{}_{}'.format(j, 'Clint', cfgd.PROPERTY_ERROR['Clint'])]

                if solubility_bool and clint_bool:
                    option = 0
                    green_boxes[i] = green_boxes[i] + 1

                elif solubility_bool and not clint_bool:
                    option = 1
                    red_boxes[i, 0] = red_boxes[i, 0] + 1

                elif not solubility_bool and clint_bool:
                    option = 2
                    red_boxes[i, 1] = red_boxes[i, 1] + 1
                else:
                    option = 3
                    red_boxes[i, 2] = red_boxes[i, 2] + 1

                # find maximum common structure
                mols = [Chem.MolFromSmiles(row["Source_Mol"])]
                mol_gen = Chem.MolFromSmiles(str(row["Predicted_smi_" + str(j)]))
                if mol_gen:
                    mols.append(mol_gen)
                    res = rdFMCS.FindMCS(mols)
                    patt = Chem.MolFromSmarts(res.smartsString)
                    sub = mols[1].GetSubstructMatches(patt)

                    if len(sub) > 0:
                        matches = sub[0]
                        matches = tuple(tuple(set(range(len(mols[1].GetAtoms()))) - set(matches)))
                    else:
                        matches = ()
                else:
                    matches = ()
            smiles_status = (str(row["Predicted_smi_" + str(j)]), save_info, option, j, matches)
            generated_mols.append(smiles_status)

        # Sort molecules after option(Green first then red in consequtive order) and then after abs(delta LogD)
        generated_mols = sorted(generated_mols, key=lambda x: (x[2], x[1][0]))

        batch.extend([tuple[0] for tuple in generated_mols])
        all_gen_mols.append(generated_mols)
        smiles.extend(batch)

        matches_list.extend([tuple[-1] for tuple in generated_mols])
        matches_all.extend(matches_list)

    # Create array with all smiles to plot in correct order
    molecules = [Chem.MolFromSmiles(str(smile)) for smile in smiles]

    return molecules, green_boxes, red_boxes, all_gen_mols, matches_all
