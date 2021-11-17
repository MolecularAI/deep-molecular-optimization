import os
import pickle as pkl

import torch
import torch.nn as nn

import configuration.config_default as cfgd
import utils.log as ul
import utils.file as uf
import utils.torch_util as ut
import utils.chem as uc
import preprocess.vocabulary as mv
from models.transformer.encode_decode.model import EncoderDecoder
from models.transformer.module.noam_opt import NoamOpt as moptim
from models.transformer.module.decode import decode
from trainer.base_trainer import BaseTrainer
from models.transformer.module.label_smoothing import LabelSmoothing
from models.transformer.module.simpleloss_compute import SimpleLossCompute


class TransformerTrainer(BaseTrainer):

    def __init__(self, opt):
        super().__init__(opt)

    def get_model(self, opt, vocab, device):
        vocab_size = len(vocab.tokens())

        if opt.use_data_parallel:
            self.LOG.info(f'Using DataParallel with {torch.cuda.device_count()} GPUs')

        # build a model from scratch or load a model from a given epoch
        if opt.starting_epoch == 1:
            # define model
            model = EncoderDecoder.make_model(vocab_size, vocab_size, N=opt.N,
                                          d_model=opt.d_model, d_ff=opt.d_ff, h=opt.H, dropout=opt.dropout)
        else:
            # Load model
            file_name = os.path.join(self.save_path, f'checkpoint/model_{opt.starting_epoch-1}.pt')
            model= EncoderDecoder.load_from_file(file_name)

        if opt.use_data_parallel:
            model = nn.DataParallel(model)
        # move to GPU
        model.to(device)
        return model

    def _initialize_optimizer(self, model, opt):
        optim = moptim(opt.d_model, opt.factor, opt.warmup_steps,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(opt.adam_beta1, opt.adam_beta2),
                                        eps=opt.adam_eps))
        return optim

    def _load_optimizer_from_epoch(self, model, file_name):
        # load optimization
        checkpoint = torch.load(file_name, map_location='cuda:0')
        optim_dict = checkpoint['optimizer_state_dict']
        optim = moptim(optim_dict['model_size'], optim_dict['factor'], optim_dict['warmup'],
                       torch.optim.Adam(model.parameters(), lr=0))
        optim.load_state_dict(optim_dict)
        return optim

    def get_optimization(self, model, opt):
        # optimization
        if opt.starting_epoch == 1:
            optim = self._initialize_optimizer(model, opt)
        else:
            # load optimization
            file_name = os.path.join(self.save_path, f'checkpoint/model_{opt.starting_epoch-1}.pt')
            optim = self._load_optimizer_from_epoch(model, file_name)

        return optim

    def train_epoch(self, dataloader, model, loss_compute, device):

        pad = cfgd.DATA_DEFAULT['padding_value']
        total_loss = 0
        total_tokens = 0
        for i, batch in enumerate(ul.progress_bar(dataloader, total=len(dataloader))):
            src, source_length, trg, src_mask, trg_mask, _, _ = batch
            trg_y = trg[:, 1:].to(device)  # skip start token

            # number of tokens without padding
            ntokens = float((trg_y != pad).data.sum())

            # Move to GPU
            src = src.to(device)
            trg = trg[:, :-1].to(device)  # save start token, skip end token
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)

            # Compute loss
            out = model(src, src_mask, mode='forward', trg=trg, trg_mask=trg_mask)
            loss = loss_compute(out, trg_y, ntokens)
            total_tokens += ntokens
            total_loss += float(loss)

        loss_epoch = total_loss / total_tokens

        return loss_epoch

    def validation_stat(self, dataloader, model, loss_compute, device, vocab, without_property=False):
        pad = cfgd.DATA_DEFAULT['padding_value']
        total_loss = 0

        n_correct = 0
        total_n_trg = 0
        total_tokens = 0
        n_correct_token = 0
        similarities = []

        tokenizer = mv.SMILESTokenizer()
        for i, batch in enumerate(ul.progress_bar(dataloader, total=len(dataloader))):

            src, source_length, trg, src_mask, trg_mask, _, _ = batch

            trg_y = trg[:, 1:].to(device)  # skip start token

            # number of tokens without padding
            ntokens = float((trg_y != pad).data.sum())

            # Move to GPU
            src = src.to(device)
            trg = trg[:, :-1].to(device)  # save start token, skip end token
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)

            with torch.no_grad():
                # Compute loss with teaching forcing
                out = model(src, src_mask, mode='forward', trg=trg, trg_mask=trg_mask)
                loss = loss_compute(out, trg_y, ntokens)
                total_loss += float(loss)
                total_tokens += ntokens

                # Decode
                max_length_target = cfgd.DATA_DEFAULT['max_sequence_length']
                smiles = model(src, src_mask, mode='sampling', max_len=max_length_target, sample_type='greedy')

                # Compute accuracy
                for j in range(trg.size()[0]):
                    seq = smiles[j, :]
                    target = trg[j]
                    target = tokenizer.untokenize(vocab.decode(target.cpu().numpy()))
                    seq = tokenizer.untokenize(vocab.decode(seq.cpu().numpy()))

                    # molecular accuracy
                    if seq == target:
                        n_correct += 1
                    # token accuracy
                    for k in range(len(target)):
                        if k < len(seq) and seq[k] == target[k]:
                            n_correct_token += 1

                    start_ind = 0 if without_property else len(cfgd.PROPERTIES)
                    source_seq = tokenizer.untokenize(vocab.decode(src[j].cpu().numpy()[start_ind:]))
                    sim = uc.tanimoto_similarity(seq, source_seq)
                    if sim:
                        similarities.append(sim)

            # number of samples in current batch
            n_trg = trg.size()[0]
            # total samples
            total_n_trg += n_trg

        # Accuracy
        accuracy = n_correct*1.0 / total_n_trg
        loss_epoch = total_loss / total_tokens
        token_accuracy = n_correct_token/total_tokens

        sim_avg = 0
        if len(similarities)>0:
            sim_avg = sum(similarities) / len(similarities)
        return loss_epoch, accuracy, token_accuracy, sim_avg

    def _get_model_parameters(self, vocab_size, opt):
        return {
            'vocab_size': vocab_size,
            'N': opt.N,
            'd_model': opt.d_model,
            'd_ff': opt.d_ff,
            'H': opt.H,
            'dropout': opt.dropout
        }

    def save(self, model, optim, epoch, vocab_size, opt):
        """
        Saves the model, optimizer and model hyperparameters
        """
        save_dict = {
            'model_state_dict': model.module.state_dict() if opt.use_data_parallel else model.state_dict(),
            'optimizer_state_dict': optim.save_state_dict(),
            'model_parameters': self._get_model_parameters(vocab_size, opt)
        }

        file_name = os.path.join(self.save_path, f'checkpoint/model_{epoch}.pt')
        uf.make_directory(file_name, is_dir=False)

        torch.save(save_dict, file_name)

    def train(self, opt):
        # Load vocabulary
        with open(os.path.join(opt.data_path, 'vocab.pkl'), "rb") as input_file:
            vocab = pkl.load(input_file)
        vocab_size = len(vocab.tokens())

        # Data loader
        dataloader_train = self.initialize_dataloader(opt.data_path, opt.batch_size, vocab, 'train',
                                                      opt.without_property)
        dataloader_validation = self.initialize_dataloader(opt.data_path, opt.batch_size, vocab, 'validation',
                                                           opt.without_property)

        device = ut.allocate_gpu()

        model = self.get_model(opt, vocab, device)
        optim = self.get_optimization(model, opt)

        total_params = sum(p.numel() for p in model.parameters())
        total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.LOG.info(f'Total number of parameters: {total_params}')
        self.LOG.info(f'Total number of trainable parameters: {total_params_trainable}')
        self.LOG.info(f'Number of gpus: {torch.cuda.device_count()}')

        pad_idx = cfgd.DATA_DEFAULT['padding_value']
        criterion = LabelSmoothing(size=len(vocab), padding_idx=pad_idx, smoothing=opt.label_smoothing)

        # Train epoch
        for epoch in range(opt.starting_epoch, opt.starting_epoch + opt.num_epoch):
            self.LOG.info("Starting EPOCH #%d", epoch)

            self.LOG.info("Training start")
            model.train()
            loss_epoch_train = self.train_epoch(dataloader_train,
                                                       model,
                                                       SimpleLossCompute(
                                                                 criterion,
                                                                 optim), device)

            self.LOG.info("Training end")
            self.save(model, optim, epoch, vocab_size, opt)

            self.LOG.info("Validation start")
            model.eval()
            loss_epoch_validation, accuracy, token_accuracy, similarities = self.validation_stat(
                dataloader_validation,
                model,
                SimpleLossCompute(criterion, None),
                device, vocab, opt.without_property)
            self.LOG.info("Validation end")

            self.LOG.info(
                "Train loss, Validation loss, molecular accuracy, "
                "token_accuracy, similarities: {}, {}, {}, {}, {}".format(loss_epoch_train, loss_epoch_validation,
                                                                           accuracy, token_accuracy, similarities))
            self.to_tensorboard(loss_epoch_train, loss_epoch_validation, accuracy, token_accuracy, similarities, epoch)
