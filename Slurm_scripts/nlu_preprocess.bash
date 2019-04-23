#!/bin/bash -l
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bz957@nyu.edu
#
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=32
#
# project id
#SBATCH --job-name=slurm_%j 
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
module load hdf5/1.10.1 
module load cuda90/toolkit/9.0.176
module load miniconda3/4.5.1
cd /gpfs/scratch/bz957/nlu/LM_bias/model
source activate test
python -u preprocess.py ../data/sample_stories_cda ../data/preprocessed_cda .txt 50000 -n 32
