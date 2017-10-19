import dpp_rbf_unitcube
import rbf_kernel
import zero_one_cube_unif_sampler
import numpy
import dpp_mcmc_sampler

d = 25
k = 10
num_iters = int(max(1000, numpy.power(k,2) * d))


import pdb; pdb.set_trace()

sampler = zero_one_cube_unif_sampler.Cube_Sampler(d)
dist = rbf_kernel.RBF_Clipped_Kernel()


for i in [1,10,25]:

    B_Y, L_Y, time =  dpp_mcmc_sampler.sample_k_disc_and_cont(sampler, dist, k, num_iters)

    # compare uniform sampling vs dpp

    
    B_Y_u = sampler(k)
    L_Y_u = dist(B_Y_u,B_Y_u, gamma=1.0/i)




#tmp = dpp_rbf_unitcube.DPPSampler(10,25)

#tmp = dpp_rbf_unitcube.DPPSampler(5,4)
