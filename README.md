![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)
# Molecular Optimization by Capturing Chemist's Intuition Using Deep Neural Networks
## Description
Implementation of the Seq2Seq with attention and the Transformer used in [Molecular Optimization by Capturing Chemist's Intuition Using Deep Neural Networks](https://chemrxiv.org/articles/preprint/Molecular_Optimization_by_Capturing_Chemist_s_Intuition_Using_Deep_Neural_Networks/12941744).
Given a molecule and desirable property changes, the goal is to generate molecules with desirable property changes. This problem can be viewed as a machine translation problem in natural language processing. Property changes are incorporated into input together with SMILES. 

![Alt text](./data/input_representation.PNG)

## Usage
Create environment 

```
conda env create -f environment.yml
source activate molopt
```
1. Preprocess data

 Encode property change, build vocabulary, and split data into train, validation and test. Outputs are saved in the same directory with input data path.

```
python preprocess.py --input-data-path data/chembl_02/mmp_prop.csv
```
2. Train model

 Train the model and save results and logs to `experiments/save_directory/`; The model from each epoch is saved in 
`experiments/save_directory/checkpoint/`; The training loss, validation loss and validation accuracy are saved in `experiments/save_directory/tensorboard/`.
```
python train.py --data-path data/chembl_02 --save-directory train_transformer --model-choice transformer transformer
``` 
3. Generate molecules

Use the model saved at a given epoch (e.g. 60) to generate molecules for the given test filename, and save the results to `experiments/save_directory/test_file_name/evaluation_epoch/generated_molecules.csv`. The three test sets used in our paper can be found in `data/chembl_02/` as below,

- Test-Original ->` data/chembl_02/test.csv`
- Test-Molecule -> `data/chembl_02/test_not_in_train.csv`
- Test-Property -> `data/chembl_02/test_unseen_L-1_S01_C10_range.csv`

```
python generate.py --model-choice transformer --data-path data/chembl_02 --test-file-name test --model-path experiments/train_transformer/checkpoint --save-directory evaluation_transformer --epoch 60
python generate.py --model-choice transformer --data-path data/chembl_02 --test-file-name test_not_in_train --model-path experiments/train_transformer/checkpoint --save-directory evaluation_transformer --epoch 60
python generate.py --model-choice transformer --data-path data/chembl_02 --test-file-name test_unseen_L-1_S01_C10_range --model-path experiments/train_transformer/checkpoint --save-directory evaluation_transformer --epoch 60
```   
4. Compute properties for generated molecules

Since we build the property prediction model based on the in-house experimental data, we can't make it public. But the computed properties can be found in `experiments/evaluation_transformer/test_file_name/evaluation_60/generated_molecules_prop.csv`

5.Evaluate the generated molecules in term of satisfying the desirable properties and draw molecules
```
python evaluate.py --data-path experiments/evaluation_transformer/test/evaluation_60/generated_molecules_prop.csv
python evaluate.py --data-path experiments/evaluation_transformer/test_not_in_train/evaluation_60/generated_molecules_prop.csv
python evaluate.py --data-path experiments/evaluation_transformer/test_unseen_L-1_S01_C10_range/evaluation_60/generated_molecules_prop.csv --range-evaluation lower
```
6. Matched molecular pair analysis between starting molecules and generated molecules

- Download [mmpdb](https://github.com/rdkit/mmpdb) for matched molecular pair generation
- Parse the downloaded mmpdb path (i.e. path/mmpdb/) to --mmpdb-path of mmp_analysis.py

Between starting molecules and all the generated molecules
```
python mmp_analysis.py --data-path experiments/evaluation_transformer/test/evaluation_60/generated_molecules_prop.csv --train-path data/chembl_02/train.csv --mmpdb-path path/mmpdb/
python mmp_analysis.py --data-path experiments/evaluation_transformer/test_not_in_train/evaluation_60/generated_molecules_prop.csv --train-path data/chembl_02/train.csv --mmpdb-path path/mmpdb/
python mmp_analysis.py --data-path experiments/evaluation_transformer/test_unseen_L-1_S01_C10_range/evaluation_60/generated_molecules_prop.csv --train-path data/chembl_02/train.csv --mmpdb-path path/mmpdb/
```

Between starting molecules and all the generated molecules with desirable properties
```
python mmp_analysis.py --data-path experiments/evaluation_transformer/test/evaluation_60/generated_molecules_prop_statistics.csv --train-path data/chembl_02/train.csv --mmpdb-path path/mmpdb/ --only-desirable
python mmp_analysis.py --data-path experiments/evaluation_transformer/test_not_in_train/evaluation_60/generated_molecules_prop_statistics.csv --train-path data/chembl_02/train.csv --mmpdb-path path/mmpdb/ --only-desirable
python mmp_analysis.py --data-path experiments/evaluation_transformer/test_unseen_L-1_S01_C10_range/evaluation_60/generated_molecules_prop_statistics.csv --train-path data/chembl_02/train.csv --mmpdb-path path/mmpdb/ --only-desirable
```
### License
The code is copyright 2020 by Jiazhen He and distributed under the Apache-2.0 license. See [LICENSE](LICENSE.md) for details.