#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=2:00:00
#SBATCH -J lsgsm
#SBATCH -o logs/%x.o%j

module purge
module load cuda cudnn gcc 
source activate jaxenv

#pathfinder = 0 23 30 40 52 81 14 64 68 3 48 49 31 85 20 51 11 69 33 44 99
#glm = 0 23 30 40 52 81
#gp = 14 64 68
#gaussmix = 3 48 49
#heirarchical = 31 85
#diffeq = 20 51
#hmm = 11 69
#time series model = 33 44 99
#toshow = 23 68 48 31 51 11 44


niter=10000
alg="lsgsm"
lambdat=0
batch=1
D=10
s=1.0
t=1.0


seed=0
for reg  in 0.1 1 10 100 1000 10000;
do
    python  -u sinh-arcsinh.py -D $D -s $s -t $t  --alg $alg  --niter $niter --batch $batch --seed $seed  --reg $reg --lambdat $lambdat  &
done
wait 

# for reg  in 0.1 1 10 100 1000 10000;
# do
#     for seed in 0 
#     do 
#         python  -u sinh-arcsinh.py -D $D -s $s -t $t  --alg $alg  --niter $niter --batch $batch --seed $seed  --reg $reg --lambdat $lambdat  &
#     done
#     wait 
# done

