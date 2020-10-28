""" Implementation of all available options """
from __future__ import print_function


def train_opts(parser):
    # Transformer or Seq2Seq
    parser.add_argument('--model-choice', required=True, help="transformer or seq2seq")
    # Common training options
    group = parser.add_argument_group('Training_options')
    group.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    group.add_argument('--num-epoch', type=int, default=60,
                       help='Number of training steps')
    group.add_argument('--starting-epoch', type=int, default=1,
                       help="Training from given starting epoch")
    # Input output settings
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data-path', required=True,
                       help="""Input data path""")
    group.add_argument('--save-directory', default='train',
                       help="""Result save directory""")

    subparsers = parser.add_subparsers()
    transformer_parser = subparsers.add_parser('transformer')
    train_opts_transformer(transformer_parser)

    seq2seq_parser = subparsers.add_parser('seq2seq')
    train_opts_seq2seq(seq2seq_parser)

def train_opts_transformer(parser):
    # Model architecture options
    group = parser.add_argument_group('Model')
    group.add_argument('-N', type=int, default=6,
                       help="number of encoder and decoder")
    group.add_argument('-H', type=int, default=8,
                       help="heads of attention")
    group.add_argument('-d-model', type=int, default=256,
                       help="embedding dimension, model dimension")
    group.add_argument('-d-ff', type=int, default=2048,
                       help="dimension in feed forward network")
    group.add_argument('--dropout', type=float, default=0.1,
                       help="Dropout probability; applied in LSTM stacks.")
    # Optimization options
    group = parser.add_argument_group('Optimization')
    group.add_argument('--learning-rate', type=float, default=1.0,
                       help="""Starting learning rate""")
    group.add_argument('--warmup-steps', type=int, default=4000,
                       help="""Number of warmup steps for custom decay.""")
    group.add_argument('--label-smoothing', type=float, default=0.0,
                       help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size - 1).
                       Set to zero to turn off label smoothing.
                       For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")

def train_opts_seq2seq(parser):
    # Model architecture options
    group = parser.add_argument_group('Model')
    group.add_argument("--num-layers", "-l", help="Number of RNN layers of the model",
                        default=5, type=int)
    group.add_argument("--layer-size", "-s", help="Size of each of the RNN layers",
                        default=512, type=int)
    group.add_argument("--cell-type", "-c",
                        help="Type of cell used in RNN [gru, lstm]",
                        default='lstm', type=str)
    group.add_argument("--embedding-layer-size", "-e", help="Size of the embedding layer",
                        default=256, type=int)
    group.add_argument("--dropout", "-d", help="Amount of dropout between layers ",
                        default=0.3, type=float)
    group.add_argument("--bidirectional", "--bi", help="Encoder bidirectional", action="store_false")
    group.add_argument("--bidirect-model",
                        help="Method to use encoder hidden state for initialising decoder['concat', 'addition', 'none']",
                        default='addition', type=str)
    group.add_argument("--attn-model", help="Attention model ['dot', 'general', 'concat']",
                        default='dot', type=str)
    # Optimization options
    group = parser.add_argument_group('Optimization')
    group.add_argument('--learning-rate', type=float, default=0.0001,
                       help="""Starting learning rate""")
    group.add_argument("--clip-gradient-norm", help="Clip gradients to a given norm",
                       default=1.0, type=float)


def generate_opts(parser):
    # Transformer or Seq2Seq
    parser.add_argument('--model-choice', required=True, help="transformer or seq2seq")
    """Input output settings"""
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data-path', required=True,
                       help="""Input data path""")
    group.add_argument('--test-file-name', required=True, help="""test file name without .csv,
        [test, test_not_in_train, test_unseen_L-1_S01_C10_range]""")
    group.add_argument('--save-directory', default='evaluation',
                       help="""Result save directory""")
    # Model to be used for generating molecules
    group = parser.add_argument_group('Model')
    group.add_argument('--model-path', help="""Model path""", required=True)
    group.add_argument('--epoch', type=int, help="""Which epoch to use""", required=True)
    # General
    group = parser.add_argument_group('General')
    group.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    group.add_argument('--num-samples', type=int, default=10,
                       help='Number of molecules to be generated')
    group.add_argument('--decode-type',type=str, default='multinomial',help='decode strategy')


def evaluation_opts(parser):
    """Evaluation options (compute properties)"""
    group = parser.add_argument_group('General')
    group.add_argument('--data-path', required=True,
                       help="""Input data path for generated molecules""")
    group.add_argument('--num-samples', type=int, default=10,
                       help='Number of molecules generated')
    group = parser.add_argument_group('Evaluation')
    group.add_argument('--range-evaluation', default='',
                       help='[ , lower, higher]; set lower when evaluating test_unseen_L-1_S01_C10_range')
    group = parser.add_argument_group('MMP')
    group.add_argument('--mmpdb-path', help='mmpdb path; download from https://github.com/rdkit/mmpdb')
    group.add_argument('--train-path', help='Training data path')
    group.add_argument('--only-desirable', help='Only check generated molecules with desirable properties',
                       action="store_true")