# Knowledge Base Completion (kbc)
This code reproduces results in [Tensor Decompositions for Temporal Knowledge Base Completion](https://arxiv.org/abs/2004.04926) (ICLR 2020).

## Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name tkbc_env python=3.7
source activate tkbc_env
conda install --file requirements.txt -c pytorch
```

Then install the kbc package to this environment
```
python setup.py install
```

## Datasets

Once the datasets are downloaded, add them to the package data folder by running :
```
python tkbc/process_icews.py
python tkbc/process_yago.py
python tkbc/process_wikidata.py  # about 3 minutes.
python tkbc/process_wikidata12k_yago11k.py
```

This will create the files required to compute the filtered metrics.

## Reproducing results

In order to reproduce the results on the smaller datasets in the paper, run the following commands

```
python tkbc/learner.py --dataset ICEWS14 --model BiTComplEx --rank 400 --emb_reg 1e-2 --time_reg 1e-1 --learning_rate 0.1

python tkbc/learner.py --dataset ICEWS05-15 --model BiTComplEx --rank 275 --emb_reg 1e-3 --time_reg 1e-2 --learning_rate 0.1

python tkbc/learner.py --dataset gdelt --model BiTComplEx --rank 275 --emb_reg 1e-2 --time_reg 1e-2 --learning_rate 0.1

python tkbc/learner.py --dataset yago11k --model BiTComplEx --rank 275 --emb_reg 1e-1 --time_reg 1e-2 --learning_rate 0.01

python tkbc/learner.py --dataset wikidata12k --model BiTComplEx --rank 400 --emb_reg 1e-3 --time_reg 1e-2 --learning_rate 0.01
```




## License
tkbc is CC-BY-NC licensed, as found in the LICENSE file.
