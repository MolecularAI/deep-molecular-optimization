"""
Preprocess
- encode property change
- build vocabulary
- split data into train, validation and test
"""
import os
import argparse
import pickle

import preprocess.vocabulary as mv
import preprocess.data_preparation as pdp
import configuration.config_default as cfgd
import utils.log as ul
import utils.file as uf
import preprocess.property_change_encoder as pce

global LOG
LOG = ul.get_logger("preprocess", "experiments/preprocess.log")


def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

    parser.add_argument("--input-data-path", "-i", help=("Input file path"), type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # encode property change without adding property name
    property_change_encoder = pce.encode_property_change(args.input_data_path)

    # add property name before property change; save to file
    property_condition = []
    for property_name in cfgd.PROPERTIES:
        if property_name == 'LogD':
            intervals, _ = property_change_encoder[property_name]
            property_condition.extend(intervals)
        else:
            intervals = property_change_encoder[property_name]
            for name in intervals:
                property_condition.append("{}_{}".format(property_name, name))
    LOG.info("Property condition tokens: {}".format(len(property_condition)))

    encoded_file = pdp.save_df_property_encoded(args.input_data_path, property_change_encoder, LOG)

    LOG.info("Building vocabulary")
    tokenizer = mv.SMILESTokenizer()
    smiles_list = pdp.get_smiles_list(args.input_data_path)
    vocabulary = mv.create_vocabulary(smiles_list, tokenizer=tokenizer, property_condition=property_condition)
    tokens = vocabulary.tokens()
    LOG.info("Vocabulary contains %d tokens: %s", len(tokens), tokens)

    # Save vocabulary to file
    parent_path = uf.get_parent_dir(args.input_data_path)
    output_file = os.path.join(parent_path, 'vocab.pkl')
    with open(output_file, 'wb') as pickled_file:
        pickle.dump(vocabulary, pickled_file)
    LOG.info("Save vocabulary to file: {}".format(output_file))

    # Split data into train, validation, test
    train, validation, test = pdp.split_data(encoded_file, LOG)

