#!/bin/bash

conda env create -f environment.yml
source activate base
conda activate cpfairre
conda install -c gurobi gurobi
sudo apt-get install libpython3.7

echo "Do not forget to place a Gurobi license in your home folder!"
echo "The repository will not run without a license!"