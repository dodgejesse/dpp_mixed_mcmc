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

def DPPNsquaredNarrow(n,d):
    g = n*n
    return DPPSampler(n,d,gamma=g)

def DPPNsquaredOverD(n,d):
    g = 1.0*(n*n)/(d*d*d)
    return DPPSampler(n,d,gamma=g)

def DPPSigma(n,d,sigma = 0.1):
    DPPSampler(n,d, sigma=sigma)
    
def DPPSearchSigma(n,d):
    unif_sampler = zero_one_cube_unif_sampler.Cube_Sampler(d)

    cur_det = 0
    cur_sigma = 1.0
    step_size = 2.0
    det_too_small = False
    det_too_big = False
    while True:
        if n == 1: 
            break
        #import pdb; pdb.set_trace()
        dist_comp = rbf_kernel.RBF_Kernel(sigma=cur_sigma)
        signs = []
        logdets = []
        for i in range(100):
            unfeat_B_Y, B_Y = unif_sampler(n)
            L_Y = dist_comp(B_Y, B_Y)
            sign, logdet = np.linalg.slogdet(L_Y)
            signs.append(sign)
            logdets.append(logdet)
        #print "n={}, d={}, cur_sigma={}, step_size={}, avg_signs={}, avg_logdets={}".format(n,d,cur_sigma, step_size, np.average(signs), np.average(logdets))

        if np.average(signs) < 1:
            cur_det = 0
        else:
            cur_det = 10**np.average(logdets)


        # check if the det is too small or large.
        # if it is, grow or shrink it. if it jumped over the good range, shrink the step size.
        if cur_det < 10**-50:
            det_too_small = True
            if det_too_big:
                step_size = np.sqrt(step_size)
                det_too_big = False
            cur_sigma = cur_sigma / step_size
        elif cur_det > 10**-45:
            det_too_big = True
            if det_too_small:
                step_size = np.sqrt(step_size)
                det_too_small = False
            cur_sigma = cur_sigma * step_size
        else:
            break
        if step_size == 1:
            break

    #print "FINAL ASSIGNMENTS: n={}, d={}, cur_sigma={}, step_size={}, avg_signs={}, avg_logdets={}".format(n,d,cur_sigma, step_size, np.average(signs), np.average(logdets))
    #print("")
    #return cur_sigma
    DPPSampler(n,d,sigma=cur_sigma)

def DPPSampler(n, d, clip_type=None,gamma=None, alpha=None, sigma=None):
    
    sampler = zero_one_cube_unif_sampler.Cube_Sampler(d)
    
    
    if clip_type is None:
        dist = rbf_kernel.RBF_Kernel(gamma, alpha, sigma)
    else:
        dist = rbf_kernel.RBF_Clipped_Kernel(clip_type)
        

    

    num_iters = int(max(1000, np.power(n,2) * d))

    #num_retries = 1

    #for i in range(num_retries):
        #try:
    unfeat_B_Y, B_Y, L_Y, time =  dpp_mcmc_sampler.sample_k_disc_and_cont(sampler, dist, n, num_iters)
    if n > 1:
        print("from this dpp sample, with d={}, n={}:".format(d,n))
        print sigma, np.linalg.det(L_Y)
        print("")
    
    return B_Y
        #except:
        #    pass


    print("FAILED!!!")
    return B_Y
