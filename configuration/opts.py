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
    group.add_argument('--use-data-parallel', help='Use pytorch DataParallel',
                       action="store_true")
    # Input output settings
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data-path', required=True,
                       help="""Input data path""")
    group.add_argument('--save-directory', default='train',
                       help="""Result save directory""")
    # Training mode
    group = parser.add_argument_group('Training mode')
    group.add_argument('--without-property', help="""Training without property tokens as input""",
                       action="store_true")

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
    # Regularization
    group.add_argument('--dropout', type=float, default=0.1,
                       help="Dropout probability; applied in LSTM stacks.")
    group.add_argument('--label-smoothing', type=float, default=0.0,
                       help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size - 1).
                       Set to zero to turn off label smoothing.
                       For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")
    # Optimization options
    group = parser.add_argument_group('Optimization')
    group.add_argument('--factor', type=float, default=1.0,
                       help="""Factor multiplied to the learning rate scheduler formula in NoamOpt. 
                       For more information about the formula, 
                       see paper Attention Is All You Need https://arxiv.org/pdf/1706.03762.pdf""")
    group.add_argument('--warmup-steps', type=int, default=4000,
                       help="""Number of warmup steps for custom decay.""")
    group.add_argument('--adam-beta1', type=float, default=0.9,
                       help="""The beta1 parameter for Adam optimizer""")
    group.add_argument('--adam-beta2', type=float, default=0.98,
                       help="""The beta2 parameter for Adam optimizer""")
    group.add_argument('--adam-eps', type=float, default=1e-9,
                       help="""The eps parameter for Adam optimizer""")


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
    group.add_argument('--vocab-path', required=True, help="""Vocabulary path""")
    group.add_argument('--test-file-name', default='test', help="""test file name without .csv""")
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
    group.add_argument('--without-property', help="""Without property tokens as input""",
                       action="store_true")


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
    group.add_argument('--without-property', help="""Draw molecules without property information""",
                       action="store_true")