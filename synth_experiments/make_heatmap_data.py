import sys
import numpy as np
import dpp_rbf_unitcube
import pickle
import sobol_seq as sobol
import numpy as np



def SobolSampler(n, d):
	X = sobol.i4_sobol_generate(d, n)
	U = np.random.rand(d)
	X += U
	X -= np.floor(X)
	return X


def draw_all_dpp_samples(ks, clip_type, sample_num):
        samples = {}
        for k in ks:
                samples[k] = dpp_rbf_unitcube.DPPSampler(k, 2, clip_type)
        g = '1overd'

        pickle_loc = 'pickled_data/dpp_samples_d=2/rbf_g={}_cliptype={}_k=odds_up_to_{}_samplenum={}'.format(g, clip_type, 
                                                                                                                 ks[-1], sample_num)
        
        pickle.dump(samples, open(pickle_loc, 'wb'))
	
def sample_dpp():	
					
        
        np.random.seed()
        k_max = 23
        ks = range(1,k_max+1,2)
        draw_all_dpp_samples(ks, "k", sys.argv[1])




def draw_one_sobol_sample(ks, sample_num):
        samples = {}
        for k in ks:
                samples[k] = SobolSampler(k, 2)
        pickle_loc = 'pickled_data/dpp_samples_d=2/sobol_k=odds_up_to_{}_samplenum={}'.format(ks[-1], sample_num)
        
        pickle.dump(samples, open(pickle_loc, 'wb'))


def sample_sobol():
        np.random.seed()
        k_max = 17
        ks = range(1,k_max+1,2)
        for i in range(100):
                print i
                for j in range(1,51):
                        draw_one_sobol_sample(ks, '{}_{}'.format(i,j))
        
        

sample_dpp()
#sample_sobol()
