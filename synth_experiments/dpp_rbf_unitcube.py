import os
import sys

cur_dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append(cur_dir_path + '/..')

import zero_one_cube_unif_sampler
import rbf_kernel
import dpp_mcmc_sampler
import numpy as np


def DPPClippedSampler(n,d):
    return DPPSampler(n,d,clip_type='k')

def DPPNarrow(n,d):
    return DPPSampler(n,d,gamma=8)

def DPPVNarrow(n,d):
    return DPPSampler(n,d,gamma=20)

def DPPVVNarrow(n,d):
    return DPPSampler(n,d,gamma=50)

def DPPVVVNarrow(n,d):
    return DPPSampler(n,d,gamma=100)

def DPPNNarrow(n,d):
    g = int(1.0*n/2)
    return DPPSampler(n,d,gamma=g)
    
def DPPNNNarrow(n,d):
    g = n
    return DPPSampler(n,d,gamma=g)

def DPPSampler(n, d, clip_type=None,gamma=None):
    
    sampler = zero_one_cube_unif_sampler.Cube_Sampler(d)
    
    
    if clip_type is None:
        dist = rbf_kernel.RBF_Kernel(gamma)
    else:
        dist = rbf_kernel.RBF_Clipped_Kernel(clip_type)

    

    num_iters = int(max(1000, np.power(n,2) * d))

    num_retries = 1
    # tries five times to get a sample
    for i in range(num_retries):
        try:
            unfeat_B_Y, B_Y, L_Y, time =  dpp_mcmc_sampler.sample_k_disc_and_cont(sampler, dist, n, num_iters)
            print np.linalg.slogdet(L_Y)
            return B_Y
        except:
            pass


    print("FAILED!!!")
    return B_Y
