import os
import sys

cur_dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append(cur_dir_path + '/..')

import zero_one_cube_unif_sampler
import rbf_kernel
import dpp_mcmc_sampler
import numpy


def DPPClippedSampler(n,d):
    return DPPSampler(n,d,'k')

def DPPSampler(n, d, clip_type=None):
    
    sampler = zero_one_cube_unif_sampler.Cube_Sampler(d)
    
    
    if clip_type is None:
        dist = rbf_kernel.RBF_Kernel()
    else:
        dist = rbf_kernel.RBF_Clipped_Kernel(clip_type)

        

    num_iters = int(max(1000, numpy.power(n,2) * d))

    num_retries = 5
    # tries five times to get a sample
    for i in range(num_retries):
        try:
            unfeat_B_Y, B_Y, L_Y, time =  dpp_mcmc_sampler.sample_k_disc_and_cont(sampler, dist, n, num_iters)
            
            return B_Y
        except:
            pass


    print("FAILED!!!")
    return B_Y
