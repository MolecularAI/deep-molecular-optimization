import torch
import torch.nn as nn
import copy

from torch.autograd import Variable
from models.transformer.module.subsequent_mask import subsequent_mask

from models.transformer.module.positional_encoding import PositionalEncoding
from models.transformer.module.positionwise_feedforward import PositionwiseFeedForward
from models.transformer.module.multi_headed_attention import MultiHeadedAttention
from models.transformer.module.embeddings import Embeddings
from models.transformer.encode_decode.encoder import Encoder
from models.transformer.encode_decode.decoder import Decoder
from models.transformer.encode_decode.encoder_layer import EncoderLayer
from models.transformer.encode_decode.decoder_layer import DecoderLayer
from models.transformer.module.generator import Generator

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, src_mask, **kwargs):
        if 'mode' in kwargs:
            if kwargs['mode'] == 'forward':
                trg = kwargs['trg']
                trg_mask = kwargs['trg_mask']
                "Take in and process masked src and target sequences."
                return self.generator(self.decode(self.encode(src, src_mask), src_mask,
                                   trg, trg_mask))
            elif kwargs['mode'] == 'sampling':
                max_len = kwargs['max_len']
                sample_type = kwargs['sample_type']

                ys = torch.zeros(src.shape[0], max_len).type_as(src.data)
                ys[:,0] = 1
                encoder_outputs = self.encode(src, src_mask)
                break_condition = torch.zeros(src.shape[0], dtype=torch.bool)
                for i in range(max_len - 1):
                    with torch.no_grad():
                        out = self.decode(encoder_outputs, src_mask, Variable(ys[:,:i+1]),
                                           Variable(subsequent_mask(ys[:,:i+1].size(1)).type_as(src.data)))

                        log_prob = self.generator(out[:, -1])
                        prob = torch.exp(log_prob)

                        if sample_type == 'greedy':
                            _, next_word = torch.max(prob, dim=1)
                            ys[:,i+1] = next_word
                        elif sample_type == 'multinomial':
                            next_word = torch.multinomial(prob, 1)
                            next_word = torch.squeeze(next_word)
                            ys[:,i+1] = next_word

                        break_condition = (break_condition | (next_word.to('cpu') == 2))
                        if all(break_condition):  # end token
                            break

                return ys

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    @classmethod
    def make_model(cls, src_vocab, tgt_vocab, N=6,
                   d_model=256, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

        return model

    @classmethod
    def load_from_file(cls, file_path):
        # Load model
        checkpoint = torch.load(file_path, map_location='cuda:0')
        para_dict = checkpoint['model_parameters']
        vocab_size = para_dict['vocab_size']
        model = EncoderDecoder.make_model(vocab_size, vocab_size, para_dict['N'],
                                  para_dict['d_model'], para_dict['d_ff'],
                                  para_dict['H'], para_dict['dropout'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model