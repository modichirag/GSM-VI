#!/bin/bash
#SBATCH -p ccm
#SBATCH -N 1
#SBATCH -C skylake
#SBATCH --time=6:00:00
#SBATCH -J frg
#SBATCH -o logs/frg.o%j

source activate jaxenv

D=$1
B=$2
alg=$3
niter=10000
echo $D $B $alg

#time python -u frgaussian.py --alg $alg -D $D --batch $B --niter $niter

#for lr in  0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5  ;
for lr in  0.005 0.01 0.02 0.05 0.1 0.2  ;
do
    echo $lr
    time python -u frgaussian.py --alg $alg -D $D --batch $B --niter $niter --lr $lr  & 
done
wait
