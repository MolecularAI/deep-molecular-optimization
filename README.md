![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)
# Transformer Neural Network-Based Molecular Optimization Using General Transformations
## Description
This repository holds the code for [Transformer Neural Network-Based Molecular Optimization Using General Transformations](todo). 

The same Transformer architecture is trained on different datasets. These datasets consist of a set of molecular pairs, and were prepared to reflect different types of transformations for molecular optimization. Below are the datasets used,
- **Matched molecule paris (MMPs)**: two molecules differ by a single transformation, and the transformation is not bigger than 33% of the whole molecule.
- **Tanimoto similarity >= 0.5**: the Tanimoto similarity between two molecules is >= 0.5. This allows for multiple modifications to a starting molecule but keeps Tanimoto similarity >= 0.5.
- **0.5 <= Tanimoto similarity < 0.7**: this allows for multiple modifications to a starting molecule but keeps 0.5 <= Tanimoto similarity < 0.7.
- **Tanimoto similarity >= 0.7**: this allows for multiple modifications to a starting molecule but keeps Tanimoto similarity >= 0.7.
- **Scaffold**: two molecules share the same scaffold (RDKit Murcko Scaffold). This allows for multiple modifications but keep the scaffold constant.
- **Scaffold generic**: two molecules share the same generic scaffold (RDKit Murcko Scaffold generic, topological scaffold). This allows for multiple modifications but keep the generic scaffold constant.

Each resulting model takes a starting molecule and user-specified desirable sets of property changes as input, and output a molecule. Multiple molecules can be generated using multinomial sampling. The way the input molecule is tranformed reflects the nature of the dataset used for training the model. 

## Usage
Create environment 

```
conda env create -f environment.yml
source activate molopt
```
The examples below illustrate the usage for the MMP dataset. Other datasets can be done similarly.

**1. Preprocess data**

 Encode property change, build vocabulary, and split data into train, validation and test. Outputs are saved in the same directory with input data path. We have provided the input data, and the preprocessed output data in [to be published](todo).

To preprocess the MMP dataset,
```
python preprocess.py --input-data-path data/MMP/pairs_prop.csv
```
**2. Train model**

 Train the model and save results and logs to `<save_directory>`; The model from each epoch is saved in 
`<save_directory>/checkpoint/`; The training loss, validation loss, validation molecular accuracy, validation token accuracy and validation Tanimoto similarity are saved in `<save_directory>/tensorboard/`.

To train a Transformer model on MMP dataset,
```
python train.py --data-path data/MMP --save-directory experiments/trained/Transformer/MMP --model-choice transformer transformer
```
where `--data-path` specifies the directory where train.csv and validation.csv are located. 

More options can be found in `configuration/opts.py`.

**3. Generate molecules**

Use the model saved at a given epoch to generate molecules for the given test filename, and save the results to `<save_directory>/test/evaluation_<epoch>/generated_molecules.csv`

To generate molecules for a test set specified with `--data-path` using the model specified with `--model-path` and `--epoch`.
```
python generate.py --model-choice transformer --data-path data/MMP --model-path experiments/trained/Transformer/MMP/checkpoint --save-directory experiments/evaluation/Transformer/MMP --epoch 60 --vocab-path data/MMP/vocab.pkl
```   
where `--data-path` specifies the directory where test.csv is located.

**4. Compute properties for generated molecules**

Since we build the property prediction model based on the in-house experimental data, we can't make it public. But we have provided the results from computing the properties, which can be found [to be published](here).

**5. Evaluate the generated molecules in term of satisfying the desirable properties and draw molecules**

```
python evaluate.py --data-path experiments/evaluation/Transformer/MMP/test/evaluation_60/generated_molecules_prop.csv
```
The outputs are saved in the directory where `generated_molecules_prop.csv` is located, in this case, `experiments/evaluation/MMP/test/evaluation_60/`. The outputs include `generated_molecules_prop_statistics.csv`, additional figures and logs.

**6. Matched molecular pair analysis between starting molecules and generated molecules**

- Download [mmpdb](https://github.com/rdkit/mmpdb) for matched molecular pair generation
- Parse the downloaded mmpdb path (i.e. path/mmpdb/) to --mmpdb-path of mmp_analysis.py

Between starting molecules and all the generated molecules
```
python mmp_analysis.py --data-path experiments/evaluation/MMP/test/evaluation_60/generated_molecules_prop_statistics.csv --mmpdb-path path/mmpdb/
```

Between starting molecules and all the generated molecules with desirable properties
```
python mmp_analysis.py --data-path experiments/evaluation/MMP/test/evaluation_60/generated_molecules_prop_statistics.csv --mmpdb-path path/mmpdb/ --only-desirable
```

### License
The code is copyright 2021 by Jiazhen He and distributed under the Apache-2.0 license. See [LICENSE](LICENSE) for details.



