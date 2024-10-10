#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=0:30:00
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


modeln=$1
niter=10000
alg="lsgsm"
modeinit=0
lbfgsinit=0
scaleinit=1.0

#lambdat=1
#batch=16
#for modeln in  $(seq  $n1 $n2)
#for modeln in  23 64 68 48 31 51 11 44 85
for seed in {0..1}
do
    echo $modeln
    for batch in 2 4 8 16 32 64 128
    #for batch in 32 
    do        
        for reg  in 0.
        do
            for lambdat in 1 0 2 ;
            do
                echo "running model " $nmodel " with seed " $seed " and batch " $batch
                python  -u pdbexp.py --alg $alg  --modeln $modeln --niter $niter --batch $batch --seed $seed  --reg $reg --lambdat $lambdat  --modeinit $modeinit --lbfgsinit $lbfgsinit --scaleinit $scaleinit  &
            done
            wait
        done
        wait
    done
    wait
done
wait

