import dpp_rbf_unitcube
import rbf_kernel
import zero_one_cube_unif_sampler
import numpy as np
import dpp_mcmc_sampler
from current_experiment import *
import dispersion
import time
import sequentially_sample_post_var






print get_ns()
exit()

def load_and_print_searched_sigmas():
    import pickle
    file_loc = 'pickled_data/searched_sigma/DPPPostVarSearchSigmaBySampling'
    sigmas = pickle.load(open(file_loc))
    for n in sorted(sigmas[1]):
        
        print '{}: {}, {}'.format(n, [round(x,5) for x in sigmas[1][n]], round(2**.5/n,5))
    exit()


load_and_print_searched_sigmas()



sequentially_sample_post_var.many_samples_search_sigma()
exit()





num_disps = 0
avg_disps = 0

ns = get_ns()
ds = get_ds()
ns = [100]
for d in ds:
    valid_ns = []
    for n in ns:
        #sigma = np.sqrt(2.0)/(n)
        
        #print('{}: {},'.format(n,sequentially_sample_post_var.find_largest_sigma(n,d)))
        while True:

            

            #sequentially_sample_post_var.draw_many_samples_search_sigma(n,d)
            sequentially_sample_post_var.main(d,n,0.25, '')

            continue
            
            



            #sigma = sigma / 10
            #sigma = 1.41421356237e-35
            
            sigma = d**0.5/12
            sigma = sigma * 1000
            
            #gamma = .5/(sigma*sigma)
            #kernel_min_val = np.exp(-gamma*d)
            #print kernel_min_val

            
            

            cur_sample = sequentially_sample_post_var.main(d, n, sigma, '')
            vor = dispersion.bounded_voronoi(cur_sample)
            cur_disp = dispersion.compute_dispersion(vor)
            
            avg_disps = (avg_disps * num_disps + cur_disp) / (num_disps + 1)
            num_disps += 1

            print avg_disps

            

            
        





exit()


ns = [5,25,50,75]
ds = [1,3,5,7,9]

ns = [3]
ds = [2]

sigmas={}
for d in ds:
    sigmas[d]={}
    for n in ns:
        
        
        #determinants[d][n] = 1.0*(n*n)/(d*d*d)
        #for i in range(1):
            
        sigmas[d][n] = dpp_rbf_unitcube.DPPSearchSigma(n,d)
        #print("done with d={}, n={}".format(d,n))

print ns
for d in ds:
    print d, [round(sigmas[d][n],5) for n in ns]

noverd = {}
for d in ds:
    noverd[d] = {}
    for n in ns:
        noverd[d][n] = 1.0*(n**2)/(d**3)
        noverd[d][n] = (1/2.0)*1.0/np.sqrt(noverd[d][n])

for d in ds:
    print d, [round(noverd[d][n],5) for n in ns]

#for d in ds:
#    #print determinants[d]
#    for n in ns:
#        print "d={}, n={}".format(d, n), determinants[d][n], np.mean(determinants[d][n]), np.var(determinants[d][n])

exit()


print get_ns()
print len(get_ns())
exit()

for i in range(500):
    i = i+1
    print i, np.sqrt(2.0)/i

exit()

num_points_per_dim = 5
d = 2
thing = [np.linspace(0,1,num_points_per_dim) for i in range(d)]

print thing
print("")

XXX = np.meshgrid(*thing, indexing='ij')
print XXX[0]

X = np.linspace(0,1,num=num_points_per_dim)
print X
X = np.array([np.array([xi]) for xi in X])
print X






exit()
draw_samples()
    



def draw_samples():
    ds = [1]
    n_max = 55
    ns = [int(np.exp(x)) for x in np.linspace(0, np.log(n_max), 20)]
    ns = sorted(list(set(ns)))
    #ns = [25,75,150]
    ns = [10]
    #import pdb; pdb.set_trace()
    # alpha scales the kernel
    
    
    for d in ds:
        for n in ns:
            for g in [20]:#[1,2,5, 8, 20, 50, 75]:
                import cProfile
                #cProfile.run('B_Y = dpp_rbf_unitcube.DPPSampler(n,d, gamma=g)', sort='cumtime')
                B_Y = dpp_rbf_unitcube.DPPSampler(n,d, gamma=g)
                import pdb; pdb.set_trace()
                print("success with d={}, n={}, g={}".format(d,n,g))
                print(sorted(B_Y))
                print("")
            
    exit()




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
