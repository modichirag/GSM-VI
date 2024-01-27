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

#pathfinder = 0 23 30 40 52 81 14 64 68 3 48 49 31 85 20 51 11 69 33 44 99
#glm = 0 23 30 40 52 81
#gp = 14 64 68
#gaussmix = 3 48 49
#heirarchical = 31 85
#diffeq = 20 51
#hmm = 11 69
#time series model = 33 44 99
#toshow = 23 68 48 31 51 11 44


modeln=$1
niter=10000
alg="advi"

#for modeln in  $(seq  $n1 $n2)
#for modeln in  23 64 68 48 31 51 11 44 85

for batch in 2 4 8 16 32
do
    echo $modeln
    #for lr  in 0.01 0.05 0.1 0.2 0.5 ;
    for lr  in 0.001 0.005  0.01 0.02 0.05 0.1 0.2 ;
    do
        
        for seed in {0..5}
        do 
            echo "running model " $nmodel " with seed " $seed " and batch " $batch
            python  -u pdbexp.py --modeln $modeln  --alg $alg  --niter $niter --batch $batch --seed $seed  --lr $lr  &
        done
        wait 
    done
    wait
done
wait

