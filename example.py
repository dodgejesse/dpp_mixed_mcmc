from zero_one_cube_unif_sampler import Cube_Sampler
from rbf_kernel import RBF_Kernel
from dpp_mcmc_sampler import sample_k_disc_and_cont
import numpy as np


k = 10
d = 3

sampler = Cube_Sampler(d)
dist = RBF_Kernel()

#import pdb; pdb.set_trace()

avgs = {}
stds = {}
max_iters = [0, 20, 50, 100, 1000, 5000]
for m in max_iters: 
    avgs[m] = 0
    stds[m] = 0

for i in range(5000):
    for m in max_iters:
        B_Y, L_Y = sample_k_disc_and_cont(sampler, dist, k, m)
    
        
        
        avgs[m] = (avgs[m] * i + np.mean(L_Y)) / (1.0*i+1)
        stds[m] = (stds[m] * i + np.std(L_Y)) / (1.0*i+1)
    avg_string = ""
    std_string = ""
    for m in max_iters:
        avg_string = avg_string + "{}".format(round(avgs[m],4)) + "\t"
        std_string = std_string + "{}".format(round(stds[m],4)) + "\t"
    print avg_string
    print std_string
    print(i)
    
