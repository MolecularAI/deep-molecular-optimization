"""
Preprocess
- build vocabulary
    - tokens based on SMILES
    - tokens based on property change
- split data into train, validation and test based on the year of publications
"""
import os
import argparse
import pickle
import pandas as pd

import preprocess.vocabulary as mv
import preprocess.data_preparation as pdp
import configuration.config_default as cfgd
import utils.log as ul
import utils.file as uf
import preprocess.property_change_encoder as pce


def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

    parser.add_argument("--input-data-path", "-i", help=("Input file path"), type=str, required=True)

    return parser.parse_args()

def encode_property_change(df):
    # encode property change without adding property name
    property_change_encoder = pce.encode_property_change(df)

    # add property name before property change
    property_condition = set()
    for property_name in cfgd.PROPERTIES:
        if property_name == 'LogD':
            intervals, _ = property_change_encoder[property_name]
        else:
            intervals = property_change_encoder[property_name]

        for name in intervals:
            property_condition.add("{}_{}".format(property_name, name))
    LOG.info("Property condition tokens: {}".format(len(property_condition)))

    return property_change_encoder, property_condition


if __name__ == "__main__":

    args = parse_args()
    parent_path = uf.get_parent_dir(args.input_data_path)
    global LOG
    LOG = ul.get_logger(name="Preprocess", log_path=os.path.join(parent_path, 'preprocess.log'))

    LOG.info("Building vocabulary")
    tokenizer = mv.SMILESTokenizer()
    # SMILES tokens
    smiles_list = pdp.get_smiles_list(args.input_data_path)
    vocabulary = mv.create_vocabulary(smiles_list, tokenizer=tokenizer)
    # Add property tokens
    df = pd.read_csv(args.input_data_path)
    property_change_encoder, property_condition = encode_property_change(df)
    vocabulary.update(list(property_condition))
    tokens = vocabulary.tokens()
    LOG.info("Vocabulary contains %d tokens: %s", len(tokens), tokens)

    # Save vocabulary to file
    output_file = os.path.join(parent_path, 'vocab.pkl')
    with open(output_file, 'wb') as pickled_file:
        pickle.dump(vocabulary, pickled_file)
    LOG.info("Save vocabulary to file: {}".format(output_file))

    # save df to file
    encoded_file = pdp.save_df_property_encoded(os.path.join(args.input_data_path), property_change_encoder, LOG)

    # Split data into train, validation, test
    train, validation, test = pdp.split_data_temporal(encoded_file, LOG)


