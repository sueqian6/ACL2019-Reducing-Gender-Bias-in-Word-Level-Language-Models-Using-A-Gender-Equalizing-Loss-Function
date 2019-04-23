#!/bin/bash -l
#
#BATCH --mail-type=ALL
#SBATCH --mail-user bz957@nyu.edu
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1

#SBATCH --job-name=slurm_%j 
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=3-0
#SBATCH --ntasks=1

module load hdf5/1.10.1 
module load cuda90/toolkit/9.0.176
module load miniconda3/4.5.1
cd /gpfs/scratch/bz957/nlu/LM_bias
source activate test
python -u  model/training_loss_cda.py --dropout 0.25 --tied --lr 20 --anneal 4 --cuda --lamda 0 --glove --batch_size 48 --data ./data/preprocessed_cda/data --vocab ./data/preprocessed_cda/VOCAB.txt --save ./savedmodel/model_cda.pt --epochs 100



