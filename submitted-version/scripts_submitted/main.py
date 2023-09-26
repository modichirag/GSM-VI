import os

#Figure 2
print("\nExecutre experiment for fig. 2\n")
command = 'python -u frgaussian.py -D 4 -r 4 --niter 100'
os.system(command)

#Figure 3
print("\nExecutre experiment for fig. 2\n")
command = 'python -u illcondfrg.py -D 4 -c 10 --niter 100'
os.system(command)

#Figure 4
print("\nExecutre experiment for fig. 2\n")
command = 'python -u nongaussian.py -D 4 --skewness 0.2 --tailw 0.9 --niter 100'
os.system(command)
