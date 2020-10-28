import os
import pickle as pkl

import torch
import torch.nn.utils as tnnu

import utils.log as ul
import utils.file as uf
import utils.torch_util as ut
import configuration.config_default as cfgd
import preprocess.vocabulary as mv
from models.seq2seq.model import Model
from trainer.base_trainer import BaseTrainer


class Seq2SeqTrainer(BaseTrainer):

    def __init__(self, opt):
        super().__init__(opt)

    def get_model(self, opt, vocab, device):
        # Train from scratch or resume training from a given epoch
        if opt.starting_epoch == 1:
            model = Model.make_model(opt.num_layers, opt.layer_size, opt.cell_type, opt.embedding_layer_size, opt.dropout,
               opt.bidirectional, opt.bidirect_model, opt.attn_model, cfgd.DATA_DEFAULT['max_sequence_length'],
                                     vocab, mv.SMILESTokenizer(), self.LOG)

        else:
            file_name = os.path.join(self.save_path, f'checkpoint/model_{opt.starting_epoch-1}.pt')
            model = Model.load_from_file(file_name)
        # move to GPU
        model.network.encoder.to(device)
        model.network.decoder.to(device)
        return model

    def _initialize_optimizer(self, model, learning_rate):
        optimizer_encoder = torch.optim.Adam(model.network.encoder.parameters(),
                                                   lr=learning_rate)
        optimizer_decoder = torch.optim.Adam(model.network.decoder.parameters(),
                                                   lr=learning_rate)

        return optimizer_encoder, optimizer_decoder

    def _load_optimizer_from_epoch(self, optimizer_encoder, optimizer_decoder, file_name):
        save_dict = torch.load(file_name)
        optimizer_encoder.load_state_dict(save_dict['optimizer_encoder'])
        optimizer_decoder.load_state_dict(save_dict['optimizer_decoder'])

    def get_optimization(self, model, opt):
        # optimization
        optimizer_encoder, optimizer_decoder = self._initialize_optimizer(model, opt.learning_rate)
        if opt.starting_epoch > 1:
            file_name = os.path.join(self.save_path, f'checkpoint/optimizer_{opt.starting_epoch-1}.pt')
            self._load_optimizer_from_epoch(optimizer_encoder, optimizer_decoder, file_name)
        return optimizer_encoder, optimizer_decoder

    def train_epoch(self, data_loader, model, optimizer_encoder, optimizer_decoder, clip_gradient_norm, device):
        model.network.encoder.train()
        model.network.decoder.train()
        pad = cfgd.DATA_DEFAULT['padding_value']
        total_loss = 0
        total_tokens = 0
        for i, batch in enumerate(ul.progress_bar(data_loader, total=len(data_loader))):
            encoder_input, source_length, decoder_output, mask, _, max_length_target, _ = batch
            # Move to GPU
            encoder_input = encoder_input.to(device)
            decoder_output = decoder_output.to(device)
            source_length = source_length.to(device)
            mask = torch.squeeze(mask, 1).to(device)
            loss_b_sq = model.loss_step(encoder_input, source_length, decoder_output, mask, max_length_target, device)

            ntokens = (decoder_output != pad).data.sum()
            loss = loss_b_sq.sum()/ntokens

            # Backprop
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward()

            if clip_gradient_norm > 0:
                tnnu.clip_grad_norm_(model.network.encoder.parameters(), clip_gradient_norm)
                tnnu.clip_grad_norm_(model.network.decoder.parameters(), clip_gradient_norm)
            # Update weights
            optimizer_encoder.step()
            optimizer_decoder.step()
            # loss
            total_tokens += ntokens
            total_loss += loss_b_sq.sum()

        loss_epoch = total_loss / total_tokens
        return loss_epoch

    def validation_stat(self, dataloader, model, device, vocab):
        pad = cfgd.DATA_DEFAULT['padding_value']
        total_loss = 0
        total_tokens = 0
        n_correct = 0
        total_n_trg = 0
        tokenizer = mv.SMILESTokenizer()
        model.network.encoder.eval()
        model.network.decoder.eval()
        for _, batch in enumerate(ul.progress_bar(dataloader, total=len(dataloader))):
            encoder_input, source_length, decoder_output, mask, _, max_length_target, _ = batch

            # Move to GPU
            encoder_input = encoder_input.to(device)
            decoder_output = decoder_output.to(device)
            source_length = source_length.to(device)
            mask = torch.squeeze(mask, 1).to(device)
            # Loss
            with torch.no_grad():
                loss_b_sq = model.loss_step(encoder_input, source_length, decoder_output, mask, max_length_target, device)
            ntokens = (decoder_output != pad).data.sum()
            total_tokens += ntokens
            total_loss += loss_b_sq.sum()

            # Sample using greedy, compute accuracy
            predicted_seqs, predicted_nlls = model.greedy_sample(encoder_input, source_length, decoder_output,
                                                         mask, device)
            for j, seq in enumerate(predicted_seqs):
                target = tokenizer.untokenize(vocab.decode(decoder_output[j].cpu().numpy()))
                smi = tokenizer.untokenize(vocab.decode(seq.cpu().numpy()))
                if smi == target:
                    n_correct += 1
            total_n_trg += decoder_output.shape[0]
        accuracy = n_correct*1.0 / total_n_trg
        loss = total_loss/total_tokens
        return loss, accuracy

    def save(self, model, optimizer_encoder, optimizer_decoder, epoch):
        file_name = os.path.join(self.save_path, f'checkpoint/model_{epoch}.pt')
        uf.make_directory(file_name, is_dir=False)
        model.save(file_name)
        self._save_training_parameters(optimizer_encoder, optimizer_decoder, epoch)

    def _save_training_parameters(self, optimizer_encoder, optimizer_decoder, epoch):
        state = {'optimizer_encoder': optimizer_encoder.state_dict(),
                 'optimizer_decoder': optimizer_decoder.state_dict()
                 }
        file_name = os.path.join(self.save_path, f'checkpoint/optimizer_{epoch}.pt')
        uf.make_directory(file_name, is_dir=False)
        torch.save(state, file_name)

    def train(self, opt):
        # Load vocabulary
        with open(os.path.join(opt.data_path, 'vocab.pkl'), "rb") as input_file:
            vocab = pkl.load(input_file)

        # Data loader
        dataloader_train = self.initialize_dataloader(opt.data_path, opt.batch_size, vocab, 'train')
        dataloader_validation = self.initialize_dataloader(opt.data_path, opt.batch_size, vocab, 'validation')

        device = ut.allocate_gpu()

        model = self.get_model(opt, vocab, device)
        optimizer_encoder, optimizer_decoder = self.get_optimization(model, opt)

        # Train epoch
        for epoch in range(opt.starting_epoch, opt.starting_epoch + opt.num_epoch):
            self.LOG.info("Starting EPOCH #%d", epoch)

            self.LOG.info("Training start")
            model.network.encoder.train()
            model.network.decoder.train()
            loss_epoch_train = self.train_epoch(dataloader_train, model, optimizer_encoder,
                                                       optimizer_decoder, opt.clip_gradient_norm, device)

            self.LOG.info("Training end")

            self.LOG.info("Validation start")
            model.network.encoder.eval()
            model.network.decoder.eval()
            with torch.no_grad():
                loss_epoch_validation, accuracy = self.validation_stat(dataloader_validation, model, device,
                                                                                  vocab)

            self.LOG.info("Validation end")

            self.LOG.info(
                "Train loss, Validation loss, accuracy: {}, {}, {}".format(loss_epoch_train, loss_epoch_validation,
                                                                           accuracy))

            self.to_tensorboard(loss_epoch_train, loss_epoch_validation, accuracy, epoch)
            self.save(model, optimizer_encoder, optimizer_decoder, epoch)