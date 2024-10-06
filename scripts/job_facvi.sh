#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH -o logs/%x.o%j


module load cuda cudnn gcc 
source activate jaxlatest

D=$1
lr=$2

niter=50001
nprint=100
savepoint=100
store_params=100
batch=8
cond=5
schedule="linear"
#suffix='test'

echo "run"
python -u facvi_exp.py -D $D --niter $niter --nprint $nprint --lr $lr --batch $batch  --savepoint $savepoint --store_params_iter $store_params --cond $cond --schedule $schedule  #--suffix $suffix
