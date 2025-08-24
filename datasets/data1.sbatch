#!/bin/bash
#
#SBATCH --job-name=L_prior # Job name
#SBATCH --output=outs/data1.txt
#SBATCH --time=200:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=30G

module load Miniconda/3           # Load module dependencies
# module load ompi-cpu
# Example Process

source activate defino-env
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZWU1MzcwYy0yYjJmLTRmNzYtOTJkYy02MWNlMmZmM2Y3ZDMifQ=="
python main.py --config configs/FIM_NS/ns_1.yaml