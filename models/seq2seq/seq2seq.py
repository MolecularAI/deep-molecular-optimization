# coding=utf-8

"""
Implementation of the Seq2Seq model
"""
import math

import torch
import torch.nn as tnn
import torch.nn.functional as tnnf

class EncoderRNN(tnn.Module):
    """
    Encoder
    """

    def __init__(self, voc_size, layer_size=512, num_layers=5, cell_type='lstm', embedding_layer_size=256, dropout=0.3,
                 bidirectional=True, bidirect_model='addition'):
        """
        Init
        :param voc_size: number of unique tokens
        :param layer_size: hidden size
        :param num_layers: number of layers
        :param cell_type: RNN cell type
        :param embedding_layer_size: embedding size
        :param dropout: dropout
        :param bidirectional: if bidirectional or not
        :param bidirect_model: how to combine two directions
        """
        super(EncoderRNN, self).__init__()

        self._layer_size = layer_size
        self._embedding_layer_size = embedding_layer_size
        self._num_layers = num_layers
        self._cell_type = cell_type.lower()
        self._dropout = dropout
        self._bidirectional = bidirectional
        self._bidirect_model = bidirect_model

        # Embedding each token in the vocabulary
        self._embedding = tnn.Embedding(voc_size, self._embedding_layer_size)
        # RNN
        if self._cell_type == 'gru':
            self._rnn = tnn.GRU(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                                bidirectional=self._bidirectional,
                                dropout=self._dropout, batch_first=True)
        elif self._cell_type == 'lstm':
            self._rnn = tnn.LSTM(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                                 bidirectional=self._bidirectional,
                                 dropout=self._dropout, batch_first=True)
        else:
            raise ValueError('Value of the parameter cell_type should be "gru" or "lstm"')

        # if use bidirectional RNN and use "concat_linear" mode (concatenate both directions and transform back)
        if self._bidirect_model == 'concat_linear':
            self._linear_bidirect = tnn.Linear(self._layer_size*2, self._layer_size)

    def forward(self, input_vector, src_len, hidden_state=None):
        """
        Performs a forward pass on the model. 
        :param input_vector: Input tensor (batch_size, seq_size).
        :param src_len: [seq len] used for pack padded sequence
        :param hidden_state: Hidden state tensor.
        :return:
        """
        batch_size, seq_size = input_vector.size()

        # embedding
        embedded_data = self._embedding(input_vector)  # (batch,seq,embedding)
        packed_embedded = tnn.utils.rnn.pack_padded_sequence(embedded_data, src_len.cpu().numpy(), batch_first=True)
        packed_outputs, hidden_state = self._rnn(packed_embedded, hidden_state)
        output_vector, _ = tnn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        output_vector = output_vector.reshape(batch_size, seq_size, -1)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden_state is now from the final non-padded element in the batch
        # output_vector: (batch_size, seq_len, num_directions*hidden_size), from the last layer including all time t
        # hidden: (num_layers*num_directions, batch_size, hidden_size) 
        # hidden_state_out is a tuple for lstm (hidden, cell)

        # Bidirectional
        if self._bidirectional:
            if self._bidirect_model == 'concat':
                # Simply concatenate, dimension doubled

                # LSTM
                if isinstance(hidden_state, tuple):
                    h_n = torch.cat(
                        [hidden_state[0][0:hidden_state.size(0):2], hidden_state[0][1:hidden_state.size(0):2]], 2)
                    c_n = torch.cat(
                        [hidden_state[1][0:hidden_state.size(0):2], hidden_state[1][1:hidden_state.size(0):2]], 2)
                    hidden_state = (h_n, c_n)

                    output_vector = torch.cat(
                        [output_vector[:, :, :self._layer_size], output_vector[:, :, self._layer_size:]], 2
                    )
                else:
                    # GRU
                    hidden_state = torch.cat(
                        [hidden_state[0:hidden_state.size(0):2], hidden_state[1:hidden_state.size(0):2]], 2)
                    output_vector = torch.cat(
                        [output_vector[:, :, :self._layer_size], output_vector[:, :, self._layer_size:]], 2
                    )
            elif self._bidirect_model == 'addition':
                # LSTM
                if isinstance(hidden_state, tuple):
                    h_n = hidden_state[0][0:hidden_state[0].size(0):2] + hidden_state[0][1:hidden_state[0].size(0):2]
                    c_n = hidden_state[1][0:hidden_state[0].size(0):2] + hidden_state[1][1:hidden_state[0].size(0):2]
                    hidden_state = (h_n, c_n)

                    output_vector = output_vector[:, :, :self._layer_size] + output_vector[:, :, self._layer_size:]
                else:
                    # GRU
                    hidden_state = hidden_state[0:hidden_state.size(0):2] + hidden_state[1:hidden_state.size(0):2]
                    output_vector = output_vector[:, :, :self._layer_size] + output_vector[:, :, self._layer_size:]
            elif self._bidirect_model == 'concat_linear':
                # GRU
                hidden_state = self._linear_bidirect(torch.cat(
                    [hidden_state[0:hidden_state.size(0):2], hidden_state[1:hidden_state.size(0):2]], 2))
                output_vector = self._linear_bidirect(torch.cat(
                    [output_vector[:, :, :self._layer_size], output_vector[:, :, self._layer_size:]], 2))

        return output_vector, hidden_state

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        """
        return {
            'dropout': self._dropout,
            'layer_size': self._layer_size,
            'num_layers': self._num_layers,
            'cell_type': self._cell_type,
            'embedding_layer_size': self._embedding_layer_size,
            'bidirectional': self._bidirectional,
            'bidirect_model': self._bidirect_model
        }

# Luong attention layer
class Attn(tnn.Module):
    def __init__(self, method, layer_size=512):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.layer_size = layer_size
        if self.method == 'general':
            self.attn = tnn.Linear(self.layer_size, layer_size)
        elif self.method == 'concat':
            self.attn = tnn.Linear(self.layer_size * 2, layer_size)
            self.v = tnn.Parameter(torch.FloatTensor(layer_size))

    def dot_score(self, hidden, encoder_output):
        _, _, layer_size = hidden.shape
        return torch.sum(hidden * encoder_output, dim=2) / math.sqrt(layer_size)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output) 
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs, mask):
        '''
        Forward
        :param hidden:
            hidden state of decoder, in shape (B, layers*directions, H)
        :param encoder_outputs:
            outputs from encoder, in shape (B, seq_len, H) or directions*H?
        :mask:
            used for masking,
        :return:
            attention energies in shape (B, seq_len)
        '''
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        attn_energies = attn_energies.masked_fill(mask == 0, -1e10)

        # Return the softmax normalized probability scores (with added dimension)
        return tnnf.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(tnn.Module):
    def __init__(self, voc_size, layer_size=512, num_layers=5, cell_type='lstm', embedding_layer_size=256, dropout=0.3,
                 attn_model='dot', bidirect_model='addition'):
        super(LuongAttnDecoderRNN, self).__init__()

        self._attn_model = attn_model
        self._layer_size = layer_size
        self._num_layers = num_layers
        self._embedding_layer_size = embedding_layer_size
        self._cell_type = cell_type.lower()
        self._dropout = dropout
        self._bidirect_model = bidirect_model

        self._embedding = tnn.Embedding(voc_size, self._embedding_layer_size)

        self._layer_size = self._layer_size*2 if self._bidirect_model == 'concat' else self._layer_size


        if self._cell_type == 'gru':
            self._rnn = tnn.GRU(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                                dropout=self._dropout, batch_first=True)
        elif self._cell_type == 'lstm':
            self._rnn = tnn.LSTM(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                                 dropout=self._dropout, batch_first=True)
        else:
            raise ValueError('Value of the parameter cell_type should be "gru" or "lstm"')


        self.concat = tnn.Linear(self._layer_size * 2, self._layer_size)
        self._linear = tnn.Linear(self._layer_size, voc_size)
        self.attn = Attn(attn_model, self._layer_size)

    def forward(self, input_vector, hidden_state, encoder_outputs=None, mask=None):
        """

        :param input_vector: [batch, seq_len]
        :param hidden_state:
        :param encoder_outputs: [batch, src_seq_len, num_directions*hidden_size]
        :param mask:
        :return:
        """
        embedded_data = self._embedding(input_vector)  # (batch,seq,embedding)
        rnn_output, hidden_state = self._rnn(embedded_data, hidden_state)
        # rnn_output: [batch, seq, num_dir*hidden_size]
        # hidden_state_out: [batch, num_dir*num_layers, hidden_size]

        # Calculate attention weights from the current RNN output
        attn_weights = self.attn(rnn_output.transpose(0, 1), encoder_outputs.transpose(0, 1), mask)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs)  # [batch, seq_len, hidden_size]
        # Concatenate weighted context vector and RNN output using Luong eq. 5
        rnn_output = rnn_output.squeeze(1)  # [batch, hidden]
        context = context.squeeze(1)  # [batch, hidden]
        concat_input = torch.cat((rnn_output, context), 1)  # [batch, hidden_rnn_output + hidden_context]
        concat_output = torch.tanh(self.concat(concat_input))  # [batch, hidden]
        # Predict next word using Luong eq. 6
        output = self._linear(concat_output) 

        return output, hidden_state

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        """
        return {
            'dropout': self._dropout,
            'layer_size': self._layer_size,
            'num_layers': self._num_layers,
            'cell_type': self._cell_type,
            'embedding_layer_size': self._embedding_layer_size,
            'attn_model': self._attn_model,
            'bidirect_model': self._bidirect_model
        }


class Seq2Seq(tnn.Module):

    def __init__(self, vocabulary_size, encoder_params=None, decoder_params=None):
        super(Seq2Seq, self).__init__()

        # Encoder
        self.encoder = EncoderRNN(vocabulary_size, **encoder_params)
        # Decoder
        self.decoder = LuongAttnDecoderRNN(vocabulary_size, **decoder_params)

    def forward(self, input_vector, input_lengths, output_vector, mask):
        encoder_outputs, encoder_hidden = self.encoder(input_vector, input_lengths)
        output_data, hidden_state_out = self.decoder(output_vector, encoder_hidden, encoder_outputs=encoder_outputs,
                                                     mask=mask)
        return output_data, hidden_state_out
