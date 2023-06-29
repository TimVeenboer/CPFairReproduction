# CPfair Reproduction

This is a reproduction of the CPfair paper published by Naghiaei et al. This repository contains both the means to reproduce the results presented in the paper, as well as an extension upon the original research. 

## Setup

A Gurobi license is **required** to run this repository. We refer the reader to [Gurobi](https://www.gurobi.com/) to request an Academic or Commercial license. Once obtained, place this license in your home folder. Thereafter you can run the setup by running the designated bash script `setup.sh` in the root folder of this repository. If this fails, separately install the python requirements listed in `environment.yml`, install `libpython3.7` and install gurobi in your conda environment by running the command `conda install -c gurobi gurobi`.

The original results can be produced by accessing the notebook run.ipynb, which utilizes the `Experiment` class and the `table_reproduction.yaml` config in the first cell. This will provide the user with the tables and boxplots presented in the paper. The results for the Variational AutoEncoder for Collaborative Filtering differ from the original paper; we're uncertain as to why these results deviate so significantly from the paper since the setup of the experiment has been identical to that of the authors. The results will appear in the results folder and the current datetimes, i.e. 'results/currentdatetime/results_Gowalla.csv'.

## Extensions

This repository contains two extensions upon the original paper, though the first extension is essentially repairing and restructuring the code of the original codebase. The initial optimisation of the authors contained quite a few mistakes; therefore it did not correspond with the mathematics and explanation of the code given in the paper. The reader can run this refactored experiment within `run.ipynb`, under the name of `ExperimentDCG`.

The second extension calculates the Deviation in Consumer Fairness (DCF) and Deviation in Producer Fairness (DPF) - both explained in the paper - proportional to the group size of the groups used in the optimisation. This can again be run with a cell in `run.ipynb`, which utilizes the class `ExperimentProportional`.

## Code Changes

First of all, the entire codebase was moved from a single notebook file to separate .py files throughout the repository. In optimisation.py, the changes for the fairness optimisation can be found -> the changes to fix the DCG calculations, relevance matrix sorting, etc. Each change is indicated by a block of ### Change - ### End Change. In metrics.py, a change is included for the calculation of the nDCG for a user, since this was done based on the natural logarithm. The files clean_results.py and boxplot.py are additions to the repository made by us to calculate and create the results displayed in the paper. In the file experiment.py, several changes were made to also change the way how results are written to the disk after running an experiment.

```
@inproceedings{naghiaei2022cpfairness,
  title={CPFair: Personalized Consumer and Producer Fairness Re-ranking for Recommender Systems},
  author={Mohammadmehdi Naghiaei, Hossein A. Rahmani, Yashar Deldjoo},
  booktitle={The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2022}
}
```

```
@article{Naghiaei2022PyCPFair,
title = {PyCPFair: A framework for consumer and producer fairness in recommender systems},
journal = {Software Impacts},
pages = {100382},
year = {2022},
issn = {2665-9638},
doi = {https://doi.org/10.1016/j.simpa.2022.100382},
url = {https://www.sciencedirect.com/science/article/pii/S2665963822000835},
author = {Mohammadmehdi Naghiaei and Hossein A. Rahmani and Yashar Deldjoo}
}
```