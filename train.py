import argparse

import configuration.opts as opts
from trainer.transformer_trainer import TransformerTrainer
from trainer.seq2seq_trainer import Seq2SeqTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.train_opts(parser)
    opt = parser.parse_args()

    if opt.model_choice == 'transformer':
        trainer = TransformerTrainer(opt)
    elif opt.model_choice == 'seq2seq':
        trainer = Seq2SeqTrainer(opt)
    trainer.train(opt)
