#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
# #SBATCH -p gpu
# #SBATCH --gpus-per-task=1
# #SBATCH --cpus-per-task=1
# #SBATCH --ntasks=1
# #SBATCH --mem=32G
# #SBATCH --constraint=a100
#SBATCH --time=4:00:00
#SBATCH -o logs/r%x.o%j


module load cuda cudnn gcc 
source activate jaxlatest

D=$1
reg=$2
badcond=$3

niter=50001
nprint=100
savepoint=100
store_params=100
#tolerance=0.
#suffix='creg'

python -u pbam_exp.py -D $D --niter $niter --nprint $nprint --reg $reg --savepoint $savepoint --store_params_iter $store_params --badcond $badcond --batch 4 --suffix B4 #--suffix $suffix
