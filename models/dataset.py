# coding=utf-8

"""
Implementation of a SMILES dataset.
"""
import pandas as pd

import torch
import torch.utils.data as tud
from torch.autograd import Variable

import configuration.config_default as cfgd
from models.transformer.module.subsequent_mask import subsequent_mask

class Dataset(tud.Dataset):
    """Custom PyTorch Dataset that takes a file containing
    Source_Mol_ID,Target_Mol_ID,Source_Mol,Target_Mol,
    Source_Mol_LogD,Target_Mol_LogD,Delta_LogD,
    Source_Mol_Solubility,Target_Mol_Solubility,Delta_Solubility,
    Source_Mol_Clint,Target_Mol_Clint,Delta_Clint,
    Transformation,Core"""

    def __init__(self, data, vocabulary, tokenizer, prediction_mode=False):
        """

        :param data: dataframe read from training, validation or test file
        :param vocabulary: used to encode source/target tokens
        :param tokenizer: used to tokenize source/target smiles
        :param prediction_mode: if use target smiles or not (training or test)
        """
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        self._data = data
        self._prediction_mode = prediction_mode

    def __getitem__(self, i):
        """
        Tokenize and encode source smile and/or target smile (if prediction_mode is True)
        :param i:
        :return:
        """

        row = self._data.iloc[i]

        # tokenize and encode source smiles
        source_smi = row['Source_Mol']
        source_tokens = []

        for property_name in cfgd.PROPERTIES:
            if property_name == 'LogD':
                source_tokens.append(row['Delta_{}'.format(property_name)])
            else:
                change = row['Delta_{}'.format(property_name)]
                source_tokens.append(f"{property_name}_{change}")

        source_tokens.extend(self._tokenizer.tokenize(source_smi))
        source_encoded = self._vocabulary.encode(source_tokens)

        # tokenize and encode target smiles if it is for training instead of evaluation
        if not self._prediction_mode:
            target_smi = row['Target_Mol']
            target_tokens = self._tokenizer.tokenize(target_smi)
            target_encoded = self._vocabulary.encode(target_tokens)

            return torch.tensor(source_encoded, dtype=torch.long), torch.tensor(
                target_encoded, dtype=torch.long), row
        else:
            return torch.tensor(source_encoded, dtype=torch.long),  row

    def __len__(self):
        return len(self._data)

    @classmethod
    def collate_fn(cls, data_all):
        # sort based on source sequence's length
        data_all.sort(key=lambda x: len(x[0]), reverse=True)
        is_prediction_mode = True if len(data_all[0]) == 2 else False
        if is_prediction_mode:
            source_encoded, data = zip(*data_all)
            data = pd.DataFrame(data)
        else:
            source_encoded, target_encoded, data = zip(*data_all)
            data = pd.DataFrame(data)

        # maximum length of source sequences
        max_length_source = max([seq.size(0) for seq in source_encoded])
        # padded source sequences with zeroes
        collated_arr_source = torch.zeros(len(source_encoded), max_length_source, dtype=torch.long)
        for i, seq in enumerate(source_encoded):
            collated_arr_source[i, :seq.size(0)] = seq
        # length of each source sequence
        source_length = [seq.size(0) for seq in source_encoded]
        source_length = torch.tensor(source_length)
        # mask of source seqs
        src_mask = (collated_arr_source !=0).unsqueeze(-2)

        # target seq
        if not is_prediction_mode:
            max_length_target = max([seq.size(0) for seq in target_encoded])
            collated_arr_target = torch.zeros(len(target_encoded), max_length_target, dtype=torch.long)
            for i, seq in enumerate(target_encoded):
                collated_arr_target[i, :seq.size(0)] = seq

            trg_mask = (collated_arr_target != 0).unsqueeze(-2)
            trg_mask = trg_mask & Variable(subsequent_mask(collated_arr_target.size(-1)).type_as(trg_mask))
            trg_mask = trg_mask[:, :-1, :-1]  # save start token, skip end token
        else:
            trg_mask = None
            max_length_target = None
            collated_arr_target = None

        return collated_arr_source, source_length, collated_arr_target, src_mask, trg_mask, max_length_target, data

