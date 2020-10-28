import torch
import torch.nn as tnn
from torch.nn.utils.rnn import pad_sequence

import utils.file as uf
import configuration.config_default as cfgd
import preprocess.vocabulary as mv
import models.seq2seq.seq2seq as mseq2seq

class Model:
    def __init__(self, vocabulary, tokenizer, encoder_params=None, decoder_params=None,
                 max_sequence_length=cfgd.DATA_DEFAULT['max_sequence_length']):
        """

        :param vocabulary:
        :param tokenizer:
        :param encoder_params: encoder parameters
        :param decoder_params: decoder parameters
        :param max_sequence_length:
        :param attn_model: attention model
        """
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.network = mseq2seq.Seq2Seq(len(vocabulary), encoder_params, decoder_params)
        self._nll_loss = tnn.NLLLoss(reduction="none", ignore_index=0)

    @classmethod
    def make_model(cls, num_layers, layer_size, cell_type, embedding_layer_size, dropout,
                   bidirectional, bidirect_model, attn_model, max_sequence_length, vocabulary, tokenizer, LOG=None):
        encoder_params = {
            'num_layers': num_layers,
            'layer_size': layer_size,
            'cell_type': cell_type,
            'embedding_layer_size': embedding_layer_size,
            'dropout': dropout,
            'bidirectional': bidirectional,
            'bidirect_model': bidirect_model,
        }

        layer_size_decoder = layer_size
        if bidirectional:
            if bidirect_model == 'concat':
                layer_size_decoder = 2 * layer_size

        decoder_params = {
            'num_layers': num_layers,
            'layer_size': layer_size_decoder,
            'cell_type': cell_type,
            'embedding_layer_size': embedding_layer_size,
            'dropout': dropout,
            'attn_model': attn_model,
            'bidirect_model': bidirect_model,
        }
        if LOG:
            LOG.info("encoder params: {}".format(encoder_params))
            LOG.info("decoder parmas: {}".format(decoder_params))
        print(LOG, attn_model)
        model = Model(vocabulary=vocabulary, tokenizer=tokenizer, encoder_params=encoder_params,
                      decoder_params=decoder_params, max_sequence_length=max_sequence_length)
        return model

    @classmethod
    def load_from_file(cls, file_path, evaluation_mode=False, LOG=None):
        """
        Load a model from specified file path
        :param file_path: model file
        :param evaluation_mode: training or evaluation mode
        :return:
        """
        # model parameters
        if torch.cuda.is_available():
            save_dict = torch.load(file_path)
        else:
            save_dict = torch.load(
                file_path, map_location=lambda storage, loc: storage)

        # encoder and decoder params
        encoder_params = save_dict.get("encoder_params", {})
        decoder_params = save_dict.get("decoder_params", {})

        # load model
        model = Model(
            vocabulary=save_dict['vocabulary'],
            tokenizer=save_dict.get('tokenizer', mv.SMILESTokenizer()),
            encoder_params=encoder_params,
            decoder_params=decoder_params,
            max_sequence_length=save_dict['max_sequence_length']
        )
        model.network.encoder.load_state_dict(save_dict["encoder"])
        model.network.decoder.load_state_dict(save_dict["decoder"])
        if evaluation_mode:
            model.network.encoder.eval()
            model.network.decoder.eval()
        if LOG:
            LOG.info(model.network.encoder)
            LOG.info(model.network.decoder)

        return model

    def save(self, file):
        """
        Saves the model into a file
        :param file: file path
        """
        save_dict = {
            'vocabulary': self.vocabulary,
            'tokenizer': self.tokenizer,
            'max_sequence_length': self.max_sequence_length,
            'encoder': self.network.encoder.state_dict(),
            'decoder': self.network.decoder.state_dict(),
            'encoder_params': self.network.encoder.get_params(),
            'decoder_params': self.network.decoder.get_params()
        }
        uf.make_directory(file, is_dir=False)
        torch.save(save_dict, file)

    def loss_step(self, input_variable, lengths, target_variable, mask, max_target_len, device):
        """
        Compute loss one step at a time for the whole seqs
        :param input_variable: source sequences, encoder input (batch_size, seq_len)
        :param lengths:
        :param target_variable: target sequences, decoder input (batch_size, seq_len)
        :param mask:
        :param max_target_len: the maximum length of allowed generated sequence
        :param device: device
        :return:
        """
        mask_loss = 0

        # Forward pass through encoder
        encoder_outputs, decoder_hidden = self.network.encoder(input_variable, lengths)

        # Create initial decoder input (start with ^ tokens for each sentence)
        batch_size, _ = input_variable.size()
        decoder_input = torch.zeros(batch_size, dtype=torch.long)
        decoder_input[:] = self.vocabulary["^"]
        decoder_input = decoder_input.view(batch_size, -1)
        decoder_input = decoder_input.to(device)

        target_variable = target_variable.transpose(0, 1)  # [seq_len, batch_size]

        # Forward batch of sequences through decoder one time step at a time
        for t in range(1, max_target_len):
            decoder_output, decoder_hidden = self.network.decoder(
                decoder_input, decoder_hidden, encoder_outputs, mask
            )

            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(batch_size, -1)

            # Calculate and accumulate loss
            logits = decoder_output.squeeze(1)
            log_probs = logits.log_softmax(dim=1)
            mask_loss += self._nll_loss(log_probs,
                                        target_variable[t])

        return mask_loss

    def greedy_sample(self, input_encoder, source_length, target, mask, device):
        with torch.no_grad():
            encoder_outputs, decoder_hidden = self.network.encoder(input_encoder, source_length,
                                                                )
            # encoder_output: [batch, seq_len, num_dir*hidden_size]
            # decoder_hidden: [batch, num_layer*num_dir, hidden_size]
            batch_size = target.shape[0]
            start_token = torch.zeros(batch_size, dtype=torch.long)
            start_token[:] = self.vocabulary["^"]
            decoder_input = start_token.to(device)

            sequences = []
            nlls = 0

            mask_temp = torch.ones(self.max_sequence_length, batch_size)
            temp = pad_sequence([mask_temp, target.transpose(0, 1)], batch_first=True)
            pad_target = temp[1].transpose(0, 1).long().to(device)

            for i in range(self.max_sequence_length - 1):
                logits, decoder_hidden = self.network.decoder(decoder_input.unsqueeze(1), decoder_hidden,
                                                              encoder_outputs, mask)
                logits = logits.squeeze(1)
                log_probs = logits.log_softmax(dim=1)

                _, topi = logits.topk(1, dim=1)
                decoder_input = topi.view(-1).detach()

                sequences.append(decoder_input.view(-1, 1))
                nlls += self._nll_loss(log_probs, pad_target[:, i + 1])
                if decoder_input.sum() == 0:
                    break

            sequences = torch.cat(sequences, 1)
        return sequences.data, nlls.cpu().numpy()
