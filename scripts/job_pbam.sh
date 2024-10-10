#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -o logs/r%x.o%j

# #SBATCH -p gpu
# #SBATCH --gpus-per-task=1
# #SBATCH --cpus-per-task=1
# #SBATCH --ntasks=1
# #SBATCH --mem=32G
# #SBATCH --constraint=a100


module load cuda cudnn gcc 
#source activate jaxlatest
source activate jaxenv

D=$1
reg=$2
updateform=$3
updatemode=$4

batch=32
niter=50001
nprint=100
savepoint=100
store_params=100
cond=5
suffix='reset'
postmean=0
schedule=1.0

python -u pbam_exp.py -D $D --batch $batch --niter $niter --nprint $nprint --reg $reg --savepoint $savepoint --store_params_iter $store_params --cond $cond --updateform $updateform --updatemode $updatemode --suffix $suffix --postmean  $postmean --schedule $schedule
