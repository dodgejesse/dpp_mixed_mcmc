import dpp_rbf_unitcube
import rbf_kernel
import zero_one_cube_unif_sampler
import numpy as np
import dpp_mcmc_sampler

ds = [2]
ns = range(1,8,2)

#import pdb; pdb.set_trace()


dist = rbf_kernel.RBF_Clipped_Kernel('k')


#for i in [1,10,25]:
for d in ds:
    sampler = zero_one_cube_unif_sampler.Cube_Sampler(d)
    for n in ns:
        num_iters = int(max(1000, np.power(n,2) * d))
    
        B_Y, L_Y, time =  dpp_mcmc_sampler.sample_k_disc_and_cont(sampler, dist, n, num_iters)
        print n,d
        for L_Y_i in L_Y:
            print str(L_Y_i).replace('\n','')

        sign, logdet = np.linalg.slogdet(L_Y)
        print "log determinant: {}".format(logdet)
        print B_Y
        print("")

    # compare uniform sampling vs dpp

    
#    B_Y_u = sampler(k)
#    L_Y_u = dist(B_Y_u,B_Y_u, gamma=1.0/i)




#tmp = dpp_rbf_unitcube.DPPSampler(10,25)

#tmp = dpp_rbf_unitcube.DPPSampler(5,4)
