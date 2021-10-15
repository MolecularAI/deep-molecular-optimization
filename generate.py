import numpy as np
import pickle as pkl
import os
import argparse
import pandas as pd

import torch

import utils.chem as uc
import utils.torch_util as ut
import utils.log as ul
import utils.plot as up
import configuration.config_default as cfgd
import models.dataset as md
import preprocess.vocabulary as mv
import configuration.opts as opts
from models.transformer.module.decode import decode
from models.transformer.encode_decode.model import EncoderDecoder
from models.seq2seq.model import Model


class GenerateRunner():

    def __init__(self, opt):

        self.save_path = os.path.join('experiments', opt.save_directory, opt.test_file_name,
                                      f'evaluation_{opt.epoch}')
        global LOG
        LOG = ul.get_logger(name="generate",
                            log_path=os.path.join(self.save_path, 'generate.log'))
        LOG.info(opt)
        LOG.info("Save directory: {}".format(self.save_path))

        # Load vocabulary
        with open(os.path.join(opt.data_path, 'vocab.pkl'), "rb") as input_file:
            vocab = pkl.load(input_file)
        self.vocab = vocab
        self.tokenizer = mv.SMILESTokenizer()

    def initialize_dataloader(self, opt, vocab, test_file):
        """
        Initialize dataloader
        :param opt:
        :param vocab: vocabulary
        :param test_file: test_file_name
        :return:
        """

        # Read test
        data = pd.read_csv(os.path.join(opt.data_path, test_file + '.csv'), sep=",")
        dataset = md.Dataset(data=data, vocabulary=vocab, tokenizer=self.tokenizer, prediction_mode=True)
        dataloader = torch.utils.data.DataLoader(dataset, opt.batch_size,
                                                 shuffle=False, collate_fn=md.Dataset.collate_fn)
        return dataloader

    def generate(self, opt):

        # set device
        device = ut.allocate_gpu()

        # Data loader
        dataloader_test = self.initialize_dataloader(opt, self.vocab, opt.test_file_name)

        # Load model
        file_name = os.path.join(opt.model_path, f'model_{opt.epoch}.pt')
        if opt.model_choice == 'transformer':
            model = EncoderDecoder.load_from_file(file_name)
            model.to(device)
            model.eval()
        elif opt.model_choice == 'seq2seq':
            model = Model.load_from_file(file_name, evaluation_mode=True)
            # move to GPU
            model.network.encoder.to(device)
            model.network.decoder.to(device)
        max_len = cfgd.DATA_DEFAULT['max_sequence_length']
        df_list = []
        sampled_smiles_list = []
        for j, batch in enumerate(ul.progress_bar(dataloader_test, total=len(dataloader_test))):

            src, source_length, _, src_mask, _, _, df = batch

            # Move to GPU
            src = src.to(device)
            src_mask = src_mask.to(device)
            smiles= self.sample(opt.model_choice, model, src, src_mask,
                                                                       source_length,
                                                                       opt.decode_type,
                                                                       num_samples=opt.num_samples,
                                                                       max_len=max_len,
                                                                       device=device)

            df_list.append(df)
            sampled_smiles_list.extend(smiles)

        # prepare dataframe
        data_sorted = pd.concat(df_list)
        sampled_smiles_list = np.array(sampled_smiles_list)

        for i in range(opt.num_samples):
            data_sorted['Predicted_smi_{}'.format(i + 1)] = sampled_smiles_list[:, i]

        result_path = os.path.join(self.save_path, "generated_molecules.csv")
        LOG.info("Save to {}".format(result_path))
        data_sorted.to_csv(result_path, index=False)

    def sample(self, model_choice, model, src, src_mask, source_length, decode_type, num_samples=10,
               max_len=cfgd.DATA_DEFAULT['max_sequence_length'],
               device=None):
        batch_size = src.shape[0]
        num_valid_batch = np.zeros(batch_size)  # current number of unique and valid samples out of total sampled
        num_valid_batch_total = np.zeros(batch_size)  # current number of sampling times no matter unique or valid
        num_valid_batch_desired = np.asarray([num_samples] * batch_size)
        unique_set_num_samples = [set() for i in range(batch_size)]   # for each starting molecule
        batch_index = torch.LongTensor(range(batch_size))
        batch_index_current = torch.LongTensor(range(batch_size)).to(device)
        start_mols = []
        # zeros correspondes to ****** which is valid according to RDKit
        sequences_all = torch.ones((num_samples, batch_size, max_len))
        sequences_all = sequences_all.type(torch.LongTensor)
        max_trials = 100  # Maximum trials for sampling
        current_trials = 0

        if decode_type == 'greedy':
            max_trials = 1

        # Set of unique starting molecules
        if src is not None:
            start_ind = len(cfgd.PROPERTIES)
            for ibatch in range(batch_size):
                source_smi = self.tokenizer.untokenize(self.vocab.decode(src[ibatch].tolist()[start_ind:]))
                source_smi = uc.get_canonical_smile(source_smi)
                unique_set_num_samples[ibatch].add(source_smi)
                start_mols.append(source_smi)

        with torch.no_grad():
            if model_choice == 'seq2seq':
                encoder_outputs, decoder_hidden = model.network.encoder(src, source_length)
            while not all(num_valid_batch >= num_valid_batch_desired) and current_trials < max_trials:
                current_trials += 1

                # batch input for current trial
                if src is not None:
                    src_current = src.index_select(0, batch_index_current)
                if src_mask is not None:
                    mask_current = src_mask.index_select(0, batch_index_current)
                batch_size = src_current.shape[0]

                # sample molecule
                if model_choice == 'transformer':
                    sequences = decode(model, src_current, mask_current, max_len, decode_type)
                    padding = (0, max_len-sequences.shape[1],
                               0, 0)
                    sequences = torch.nn.functional.pad(sequences, padding)
                elif model_choice == 'seq2seq':
                    sequences = self.sample_seq2seq(model, mask_current, batch_index_current, decoder_hidden,
                                                    encoder_outputs, max_len, device)
                else:
                    LOG.info('Specify transformer or seq2seq for model_choice')

                # Check valid and unique
                smiles = []
                is_valid_index = []
                batch_index_map = dict(zip(list(range(batch_size)), batch_index_current))
                # Valid, ibatch index is different from original, need map back
                for ibatch in range(batch_size):
                    seq = sequences[ibatch]
                    smi = self.tokenizer.untokenize(self.vocab.decode(seq.cpu().numpy()))
                    smi = uc.get_canonical_smile(smi)
                    smiles.append(smi)
                    # valid and not same as starting molecules
                    if uc.is_valid(smi):
                        is_valid_index.append(ibatch)
                    # total sampled times
                    num_valid_batch_total[batch_index_map[ibatch]] += 1

                # Check if duplicated and update num_valid_batch and unique
                for good_index in is_valid_index:
                    index_in_original_batch = batch_index_map[good_index]
                    if smiles[good_index] not in unique_set_num_samples[index_in_original_batch]:
                        unique_set_num_samples[index_in_original_batch].add(smiles[good_index])
                        num_valid_batch[index_in_original_batch] += 1

                        sequences_all[int(num_valid_batch[index_in_original_batch] - 1), index_in_original_batch, :] = \
                            sequences[good_index]

                not_completed_index = np.where(num_valid_batch < num_valid_batch_desired)[0]
                if len(not_completed_index) > 0:
                    batch_index_current = batch_index.index_select(0, torch.LongTensor(not_completed_index)).to(device)

        # Convert to SMILES
        smiles_list = [] # [batch, topk]
        seqs = np.asarray(sequences_all.numpy())
        # [num_sample, batch_size, max_len]
        batch_size = len(seqs[0])
        for ibatch in range(batch_size):
            topk_list = []
            for k in range(num_samples):
                seq = seqs[k, ibatch, :]
                topk_list.extend([self.tokenizer.untokenize(self.vocab.decode(seq))])
            smiles_list.append(topk_list)


        return smiles_list

    def sample_seq2seq(self, model, mask, batch_index_current, decoder_hidden, encoder_outputs, max_len, device):
        # batch size will change when some of the generated molecules are valid
        encoder_outputs_current = encoder_outputs.index_select(0, batch_index_current)
        batch_size = encoder_outputs_current.shape[0]

        # start token
        start_token = torch.zeros(batch_size, dtype=torch.long)
        start_token[:] = self.vocab["^"]
        decoder_input = start_token.to(device)
        sequences = []
        mask = torch.squeeze(mask, 1).to(device)

        # initial decoder hidden states
        if isinstance(decoder_hidden, tuple):
            decoder_hidden_current = (decoder_hidden[0].index_select(1, batch_index_current),
                                      decoder_hidden[1].index_select(1, batch_index_current))
        else:
            decoder_hidden_current = decoder_hidden.index_select(1, batch_index_current)
        for i in range(max_len):
            logits, decoder_hidden_current = model.network.decoder(decoder_input.unsqueeze(1),
                                                                  decoder_hidden_current,
                                                                  encoder_outputs_current, mask)
            logits = logits.squeeze(1)
            probabilities = logits.softmax(dim=1)  # torch.Size([batch_size, vocab_size])
            topi = torch.multinomial(probabilities, 1)  # torch.Size([batch_size, 1])
            decoder_input = topi.view(-1).detach()
            sequences.append(decoder_input.view(-1, 1))

        sequences = torch.cat(sequences, 1)
        return sequences

def run_main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='generate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.generate_opts(parser)
    opt = parser.parse_args()

    runner = GenerateRunner(opt)
    runner.generate(opt)


if __name__ == "__main__":
    run_main()
