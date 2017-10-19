import os
import sys

cur_dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append(cur_dir_path + '/..')

import zero_one_cube_unif_sampler
import rbf_kernel
import dpp_mcmc_sampler
import numpy


def DPPSampler(n, d):
    
    sampler = zero_one_cube_unif_sampler.Cube_Sampler(d)
    
    dist = rbf_kernel.RBF_Kernel()
    #dist = rbf_kernel.RBF_Clipped_Kernel()

        

    num_iters = int(max(1000, numpy.power(n,2) * d))
    
    B_Y, L_Y, time =  dpp_mcmc_sampler.sample_k_disc_and_cont(sampler, dist, n, num_iters)
    #print("{} iters took {} seconds".format(num_iters, time))
    return B_Y
