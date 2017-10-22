import sys
import numpy as np
import dpp_rbf_unitcube
import pickle






def draw_all_samples(ks, sample_num):
        samples = {}
        for k in ks:
                samples[k] = dpp_rbf_unitcube.DPPSampler(k, 2)
        pickle_loc = 'pickled_data/dpp_samples_d=2/rbf_g=noverd_k=odds_up_to_{}_samplenum={}'.format(ks[-1], sample_num)
        
        pickle.dump(samples, open(pickle_loc, 'wb'))
	
	
					

np.random.seed()
k_max = 41
ks = range(1,k_max+1,2)
draw_all_samples(ks, sys.argv[1])
