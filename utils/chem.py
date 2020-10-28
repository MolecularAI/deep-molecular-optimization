"""
RDKit util functions.
"""
import rdkit.Chem as rkc
from rdkit.Chem import AllChem
from rdkit import DataStructs

def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)

    import rdkit.rdBase as rkrb
    rkrb.DisableLog('rdApp.error')


disable_rdkit_logging()

def to_fp_ECFP(smi):
    if smi:
        mol = rkc.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprint(mol, 2)

def tanimoto_similarity_pool(args):
    return tanimoto_similarity(*args)

def tanimoto_similarity(smi1, smi2):
    fp1, fp2 = None, None
    if smi1 and type(smi1)==str and len(smi1)>0:
        fp1 = to_fp_ECFP(smi1)
    if smi2 and type(smi2)==str and len(smi2)>0:
        fp2 = to_fp_ECFP(smi2)

    if fp1 is not None and fp2 is not None:
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    else:
        return None

def is_valid(smi):
    return 1 if to_mol(smi) else 0

def to_mol(smi):
    """
    Creates a Mol object from a SMILES string.
    :param smi: SMILES string.
    :return: A Mol object or None if it's not valid.
    """
    if isinstance(smi, str) and smi and len(smi)>0 and smi != 'nan':
        return rkc.MolFromSmiles(smi)

def get_canonical_smile(smile):
    if smile != 'None':
        mol = rkc.MolFromSmiles(smile)
        if mol is not None:
            smi = rkc.MolToSmiles(mol, canonical=True, doRandom=False, isomericSmiles=False)
            return smi
        else:
            return None
    else:
        return None