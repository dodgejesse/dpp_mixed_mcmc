import dpp_rbf_unitcube
import rbf_kernel
import zero_one_cube_unif_sampler
import numpy as np
import dpp_mcmc_sampler

ds = [1]
n_max = 55
ns = [int(np.exp(x)) for x in np.linspace(0, np.log(n_max), 20)]
ns = sorted(list(set(ns)))
ns = [25,75,150]
#import pdb; pdb.set_trace()





for d in ds:
    for n in ns:
        for g in [n]:#[1,2,5, 8, 20, 50, 75]:
            B_Y = dpp_rbf_unitcube.DPPSampler(n,d, gamma=g)
            print("success with d={}, n={}, g={}".format(d,n,g))
            print("")

exit()



dist = rbf_kernel.RBF_Clipped_Kernel('k')


num_influential = {}





#for i in [1,10,25]:
for d in ds:
    sampler = zero_one_cube_unif_sampler.Cube_Sampler(d)
    for n in ns:
        num_iters = int(max(1000, np.power(n,2) * d))


        
    
        B_Y, L_Y, time =  dpp_mcmc_sampler.sample_k_disc_and_cont(sampler, dist, n, num_iters)
        print n,d
        frac_of_k = 1.0/2
        epsilon = 1-np.sqrt(1-(1-frac_of_k**(1.0/d)))

        print "epsilon = {}".format(epsilon)
        #for L_Y_i in L_Y:
        #    print str(L_Y_i).replace('\n','')

        num_at_min = np.sum(L_Y == np.exp(-1))/2.0
        total_sims = (n-1)*n/2.0
        num_influential['{},{}'.format(n,d)] = num_at_min/total_sims
        
        print 'num at min sim: {}, or {} out of {}'.format(round(num_at_min/total_sims,3), num_at_min, total_sims)
        sign, logdet = np.linalg.slogdet(L_Y)
        print "log determinant: {}".format(logdet)
        #print B_Y
        print("")


for thing in sorted(num_influential):
    print thing, num_influential[thing]

    # compare uniform sampling vs dpp

    
#    B_Y_u = sampler(k)
#    L_Y_u = dist(B_Y_u,B_Y_u, gamma=1.0/i)




#tmp = dpp_rbf_unitcube.DPPSampler(10,25)

#tmp = dpp_rbf_unitcube.DPPSampler(5,4)
