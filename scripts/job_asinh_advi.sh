#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=2:00:00
#SBATCH -J advi
#SBATCH -o logs/%x.o%j

module purge
module load cuda cudnn gcc 
source activate jaxenv


niter=10000
alg="advi"
D=10
s=1.0
t=1.0


for batch in 2 4 8 16 32 ;
do
    for lr  in 0.01 0.05 0.1 0.2 0.5 ;
    do
        for seed in 0 
        do 
            python  -u sinh-arcsinh.py -D $D -s $s -t $t  --alg $alg  --niter $niter --batch $batch --seed $seed  --lr $lr  &
        done
    done
    wait 
done
