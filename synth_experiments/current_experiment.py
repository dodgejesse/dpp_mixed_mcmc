import numpy as np
import pickle
import dpp_rbf_unitcube
#import center_origin_plots
import sobol_seq as sobol
import functools
import sequentially_sample_post_var
import scipy
import dispersion

def get_samplers():
    samplers = {'SobolSampler':{'fn': SobolSampler,'color': 'g'},
                #'RecurrenceSampler': {'fn': RecurrenceSampler,'color': 'r'},
                #'SobolSamplerNoNoise': {'fn': SobolSamplerNoNoise,'color': 'b'},
                #'DPPnsquared': {'fn': dpp_rbf_unitcube.DPPSampler, 'color': 'k'},
                'UniformSampler': {'fn': np.random.rand, 'color': 'b'},
                #'DPPNarrow': {'fn': dpp_rbf_unitcube.DPPNarrow, 'color': 'm'},
                #'DPPVNarrow': {'fn': dpp_rbf_unitcube.DPPVNarrow, 'color': 'm'}
                #'DPPVVNarrow': {'fn': dpp_rbf_unitcube.DPPVVNarrow, 'color': 'm'},
                #'DPPVVVNarrow': {'fn': dpp_rbf_unitcube.DPPVVVNarrow, 'color': 'm'},
                #'DPPNNarrow': {'fn': dpp_rbf_unitcube.DPPNNarrow, 'color': 'm'},
                #'DPPNNNarrow': {'fn': dpp_rbf_unitcube.DPPNNNarrow, 'color': 'm'}
                #'DPPNsquaredNarrow': {'fn': dpp_rbf_unitcube.DPPNsquaredNarrow, 'color': 'm'}
                #'DPPClipped': {'fn': dpp_rbf_unitcube.DPPClippedSampler, 'color': 'm'}
                #'SobolSamplerHighD':{'fn': SobolSamplerHighD, 'color':'m'},
                #'DPPVVNarrow': {'fn': dpp_rbf_unitcube.DPPVVNarrow, 'color': 'm'},
                #'DPPNsquaredOverD': {'fn': dpp_rbf_unitcube.DPPNsquaredOverD, 'color': 'm'},
                #'DPPSearchSigma': {'fn': dpp_rbf_unitcube.DPPSearchSigma, 'color': 'm'},
                
                
                #'DPPSigma{}'.format(get_sigma()): {'fn':functools.partial(dpp_rbf_unitcube.DPPSigma, sigma=get_sigma()), 'color': 'm'},
                #'DPPPostVarSigmaSqrt2overN': {'fn':sequentially_sample_post_var.one_sample_sigma_sqrt2overN, 'color': 'm'},
                #'DPPPostVarSigmaSqrt2overND': {'fn':sequentially_sample_post_var.one_sample_sigma_sqrt2overND, 'color': 'k'},
                #'DPPPostVarSigmaSqrt2overNtosqrtD': {'fn':sequentially_sample_post_var.one_sample_sigma_sqrt2overNtosqrtD, 'color': 'y'},
                #'DPPPostVarSigmaDover45': {'fn':sequentially_sample_post_var.one_sample_sigma_dover45, 'color': 'c'},
                #'DPPPostVarSearchSigma': {'fn':sequentially_sample_post_var.one_sample_search_sigma, 'color': 'c'},
                'DPPPostVarSearchSigmaBySampling': {'fn':sequentially_sample_post_var.one_sample_search_sigma_by_sampling, 'color': 'm'},
                #'DPPSeqPostSigma{}'.format(str(get_sigma())[2:]): {'fn':sequentially_sample_post_var.draw_many_samples, 'color': 'm'},
    }
    return samplers

# useful for DPPSigma
def get_sigma():
    return 0.141

# returns the number of uniformly sampled points used when computing the average distance
# to a uniformly sampled point
def get_num_unif_eval():
    return 1000

def get_n_min():
    return 3

def get_n_max():
    return 500

def get_ns():
    n_max = get_n_max()
    n_min = get_n_min()
    ns = [int(np.exp(x)) for x in np.linspace(np.log(n_min), np.log(n_max), 30)]
    ns = sorted(list(set(ns)))
    return ns
    
def get_ds():
    ds = [2]#[40,100, 500]#[1,2,3,4]#[2,3,5,7]#[2,3,5,10,15,25,35]
    return ds


def get_eval_measures():
    eval_measures = {
        #'l2':get_min_l2_norm, 
        #'l1':get_min_l1_norm, 
        #'l2_cntr':get_min_l2_norm_center, 
        #'l1_cntr':get_min_l1_norm_center,
        #'discrep':get_discrepency,
        #'unif_point':get_min_to_uniformly_sampled_point
        'dispersion':get_dispersion,
        #'projected_1d_dispersion':get_projected_1d_dispersion,
    }

    return eval_measures

def get_num_samples():
    return 100

#if __name__ == "__main__":
#    discrepancy.draw_many_samples()
#    comp.compute_discrep_for_samples()
#    center_origin_plots.make_plots()



###################### the evaluation measures ######################

def get_discrepency(X):
        worse_value = 0.
	for x in X:
		p = np.exp(np.sum(np.log(x)))
		p_hat = 0.
		for y in X:
			p_hat += float(all(y < x))
		p_hat /= len(X)
		worse_value = max(worse_value, abs(p-p_hat))
		worse_value = max(worse_value, abs(p-p_hat-1./len(X)))
	return worse_value

# computes a bounded voronoi diagram, 
# then returns the smallest distance between the points X and the voronoi vertices
def get_dispersion(X):
    vor = dispersion.bounded_voronoi(X)
    return dispersion.compute_dispersion(vor)

def get_projected_1d_dispersion(X):
    return get_dispersion(np.array([X[:,0]]).T)
    

def get_min_norm(X, order):
	return min(np.linalg.norm(X, ord=order, axis=1))

def get_min_l2_norm(X):
	#smallest_value = float('inf')
	#for x in X:
	#	tmp = numpy.dot(x, x)
	#	smallest_value = min(smallest_value, tmp)
	#return smallest_value
	return get_min_norm(X, 2)

def get_min_l1_norm(X):
	#smallest_value = float('inf')
	#for x in X:
#		tmp = numpy.dot(x, numpy.sign(x))
#		smallest_value = min(smallest_value, tmp)0
#	return smallest_value

	return get_min_norm(X, 1)

def get_min_l2_norm_center(X):
	center = np.ones(X.shape)*.5
	return get_min_norm(X-center, 2)

def get_min_l1_norm_center(X):
	center = np.ones(X.shape)*.5
	return get_min_norm(X-center, 1)


def get_avg_min_to_uniformly_sampled_point(X):
    d = len(X[0])
    unif_points = pickle.load(open("pickled_data/unif_samples_for_eval/sampler=uniform_n={}_dim={}".format(get_num_unif_eval(), d)))

    all_dists = scipy.spatial.distance.cdist(X, unif_points)
    min_dists = np.min(all_dists, axis=0)
    assert len(min_dists) == len(unif_points)
    avg_min_dist = np.average(min_dists)
    return avg_min_dist


def get_min_to_uniformly_sampled_point(X):
    d = len(X[0])
    unif_points = pickle.load(open("pickled_data/unif_samples_for_eval/sampler=uniform_n={}_dim={}".format(get_num_unif_eval(), d)))

    all_dists = scipy.spatial.distance.cdist(X, unif_points)
    min_dists = np.min(all_dists, axis=0)
    assert len(min_dists) == len(unif_points)
    min_min_dist = np.min(min_dists)
    return min_min_dist
    

############################ the samplers ############################


# adds uniform noise in [0,1], then for those results above 1 it subtracts 1
def SobolSampler(n, d):
	X = sobol.i4_sobol_generate(d, n)
	U = np.random.rand(d)
	X += U
	X -= np.floor(X)
	return X

# this method is to try and get sobol in D > 40
# adds uniform noise in [0,1], then for those results above 1 it subtracts 1
def SobolSamplerHighD(n, d):
        assert d in [5, 50, 100, 500]
        assert n < 1025
        
        with open('/home/ec2-user/software/Sobol.jl/test/results/exp_results_{}'.format(d), 'r') as f:
                lines = f.readlines()
        in_data = []
        counter = 0
        for line in lines:
                counter += 1
                if counter == 1 or counter == 2:
                        continue


                in_data.append(line.split(" ")[:-1])
                if len(in_data) == n:
                        break
        X = np.asarray(in_data, dtype=float)
        

	U = np.random.rand(d)
	X += U
	X -= np.floor(X)
	return X

def SobolSamplerNoNoise(n, d):
	X = sobol.i4_sobol_generate(d, n)
	return X


PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

def RecurrenceSampler(n, d):
	X = np.zeros((n, d))
	X[0] = np.random.rand(d)
	for i in range(1,n):
		for k in range(d):
			X[i, k] = X[0,k] + i*np.sqrt(PRIMES[k])
			X[i, k] -= int(X[i, k])
	return np.array(X)
