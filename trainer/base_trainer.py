import os
import pandas as pd
from abc import ABC, abstractmethod

import torch
from tensorboardX import SummaryWriter

import utils.log as ul
import models.dataset as md
import preprocess.vocabulary as mv


class BaseTrainer(ABC):

    def __init__(self, opt):

        self.save_path = os.path.join('experiments', opt.save_directory)
        self.summary_writer = SummaryWriter(logdir=os.path.join(self.save_path, 'tensorboard'))
        LOG = ul.get_logger(name="train_model", log_path=os.path.join(self.save_path, 'train_model.log'))
        self.LOG = LOG
        self.LOG.info(opt)

    def initialize_dataloader(self, data_path, batch_size, vocab, data_type):
        # Read train or validation
        data = pd.read_csv(os.path.join(data_path, data_type + '.csv'), sep=",")
        dataset = md.Dataset(data=data, vocabulary=vocab, tokenizer=mv.SMILESTokenizer(), prediction_mode=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size,
                                                 shuffle=True, collate_fn=md.Dataset.collate_fn)
        return dataloader

    def to_tensorboard(self, train_loss, validation_loss, accuracy, epoch):

        self.summary_writer.add_scalars("loss", {
            "train": train_loss,
            "validation": validation_loss
        }, epoch)
        self.summary_writer.add_scalar("accuracy/validation", accuracy, epoch)

        self.summary_writer.close()

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_optimization(self):
        pass

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def validation_stat(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def train(self):
        pass

