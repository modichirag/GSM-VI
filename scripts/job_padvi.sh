#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --constraint=a100
#SBATCH --time=4:00:00
#SBATCH -J padvi
#SBATCH -o logs/%x.o%j


module load cuda cudnn gcc 
source activate jaxlatest

D=$1
lr=$2

niter=50001
nprint=100
savepoint=100
store_params=100
batch=32
cond=5
schedule="linear"
#suffix='test'

python -u padvi_exp.py -D $D --niter $niter --nprint $nprint --lr $lr --batch $batch  --savepoint $savepoint --store_params_iter $store_params --cond $cond --schedule $schedule  #--suffix $suffix
