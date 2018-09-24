import numpy as np
import os
import time
import scipy.linalg
import pickle
import sys
from current_experiment import *



def get_sigma(n,d):
    d_to_n_to_sigma = {
        2: 
        {
            3: 100,
            4: 90.9090909091,
            5: 90.9090909091,
            6: 75.1314800902,
            7: 68.3013455365,
            8: 56.4473930054,
            10: 51.3158118231,
            11: 51.3158118231,
            12: 51.3158118231,
            14: 1.1338191753,
            16: 1.1338191753,
            18: 1.1338191753,
            20: 0.298570025588,
            23: 0.203927344846,
            26: 0.203927344846,
            29: 0.139285120446,
            33: 0.139285120446,
            38: 0.095133611397,
            42: 0.0714752903058,
            48: 0.0649775366417,
            54: 0.0714752903058,
            61: 0.059070487856,
            69: 0.0443805318227,
            78: 0.0403459380207,
            88: 0.0403459380207,
            100: 0.0333437504303,
        }
    }


def one_multinom_sample(unnorm_probs):
    z = sum(unnorm_probs)
    if sum(unnorm_probs) < 0:
        # sometimes this can happen if sigma is too large
        raise np.linalg.linalg.LinAlgError

    sampled_unnorm_prob = np.random.uniform(0,z)
    cur_unnorm_prob = 0
    for i in range(len(unnorm_probs)):
        cur_unnorm_prob += unnorm_probs[i]
        if cur_unnorm_prob >= sampled_unnorm_prob:
            return i
    print("MADE IT TO A WEIRD PLACE WTF")
    print("sampled unnorm prob: {}".format(sampled_unnorm_prob))
    print('cur_unnorm_prob: {}'.format(cur_unnorm_prob))
    print('sum of unnorm_probs: {}'.format(z))
    print('unnorm_probs[0:10]: {}'.format(unnorm_probs[0:10]))
    sys.exit()
    return len(unnorm_probs)-1
    
    
def build_rbf_kernel(D, sigma):
	gamma = .5/(sigma*sigma)
        kernel_mtx = np.exp(-gamma*D)
	return kernel_mtx

def build_distance_sq_matrix(X, Z):
	return np.outer(np.sum(X**2, axis=1), np.ones(Z.shape[0])) -2*np.dot(X, Z.T) + np.outer(np.ones(X.shape[0]), np.sum(Z**2, axis=1))


def update_kernel_mtxs(train_kernel_mtx, test_to_train_kernel_mtx, X_train, X_test, new_point, sigma):
    new_point_val = X_test[np.array([new_point])]

    # add new point to train
    X_train = np.append(X_train, X_test[np.array([new_point])], axis=0)
    # remove new point from test
    X_test = np.delete(X_test, new_point, axis=0)

    # take row for new_point from test_to_train and put it in train_kernel
    new_point_to_train_kernel = np.asarray([test_to_train_kernel_mtx[new_point]])
    ## remove new_point distances from test_to_train
    test_to_train_kernel_mtx = np.delete(test_to_train_kernel_mtx, new_point, axis=0)
    new_col = np.append(new_point_to_train_kernel.T,[[1]], axis=0)
    with_new_row = np.concatenate((train_kernel_mtx, new_point_to_train_kernel))
    ## this guy is the new train_dist_mtx
    train_kernel_mtx = np.concatenate((with_new_row, new_col), axis=1)

    # compute distances between new point and test points
    test_to_new_point_dist_mtx = build_distance_sq_matrix(X_test, new_point_val)
    test_to_new_point_kernel_mtx = build_rbf_kernel(test_to_new_point_dist_mtx, sigma)

    # add new distances to test_to_train_dist_mtx
    test_to_train_kernel_mtx = np.concatenate((test_to_train_kernel_mtx, test_to_new_point_kernel_mtx), axis=1)

    return train_kernel_mtx, test_to_train_kernel_mtx, X_train, X_test


# to initialize the relevant matricies
def initialize_kernel_matrices(X_train, X_test, sigma):
    train_dist_mtx = build_distance_sq_matrix(X_train, X_train)
    train_kernel_mtx = build_rbf_kernel(train_dist_mtx, sigma)

    test_to_train_dist_mtx = build_distance_sq_matrix(X_test, X_train)
    test_to_train_kernel_mtx = build_rbf_kernel(test_to_train_dist_mtx, sigma)
    
    return train_kernel_mtx, test_to_train_kernel_mtx

def pickle_sample(X_train, sigma, n, d, sigma_name):
    if sigma_name == "Sqrt2overN":
        pickle_loc = 'pickled_data/dim=1/sampler=DPPPostVarSigma{}_n={}_d={}_samplenum={}'.format(sigma_name, n, d, sys.argv[1])
    else:
        pickle_loc = 'pickled_data/dim=1/sampler=DPPSeqPostSigma{}_n={}_d={}_samplenum={}'.format(str(sigma)[2:], n, d, sys.argv[1])
    pickle.dump(X_train, open(pickle_loc, 'wb'))


def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def get_grid(d):
    max_grid_size = 10000
    each_dim_size = int(max_grid_size**(1.0/(d)))

    dims = []
    for i in range(d):
        dims.append(np.linspace(0,1,num=each_dim_size))

    X = cartesian_product(dims)
    return X

def main(d, k, sigma, sigma_name):
    start_time = time.time()
    
    X = get_grid(d)
    new_point = np.random.randint(0,len(X))

    X_train = X[np.array([new_point])]
    X_test = np.delete(X, new_point,axis=0)    

    for i in range(k-1):
        start_iter_time = time.time()
        # update the kernel mtxs
        if i == 0:
            train_kernel_mtx, test_to_train_kernel_mtx = initialize_kernel_matrices(X_train, X_test, sigma)
        else:
            train_kernel_mtx, test_to_train_kernel_mtx, X_train, X_test = update_kernel_mtxs(train_kernel_mtx, 
                                                                           test_to_train_kernel_mtx, X_train, X_test, new_point, sigma)

        # compute unnorm probs        

        #new_L = use_eigendecomp(train_kernel_mtx)
        #new_v = np.linalg.solve(new_L, test_to_train_kernel_mtx.T)
        #new_unnorm_probs = 1-np.einsum('ij,ji->i', new_v.T,new_v)
        #unnorm_probs = new_unnorm_probs.real
        #print unnorm_probs[0:20]

        lower = True
        L = scipy.linalg.cholesky(train_kernel_mtx, lower)
        #print('old_L.shape: {}'.format(L.shape))
        v = np.linalg.solve(L, test_to_train_kernel_mtx.T)
        
        unnorm_probs = 1-np.einsum('ij,ji->i', v.T,v)
        
        # get new point        
        new_point = one_multinom_sample(unnorm_probs)
        
        if i > 100 or i == range(k-1)[-1]:
            print("iteration: {}. this iter time: {}. total elapsed time: {}".format(i, 
                                                            round(time.time() - start_iter_time,2), round(time.time() - start_time,2)))

    
    if k > 1:
        X_train = np.append(X_train, X_test[np.array([new_point])], axis=0)

    print "finished sampling n={} d={} sigma={}".format(k, d, sigma)
    print("")

    return X_train
    

def use_eigendecomp(train_kernel_mtx):
    w, v = np.linalg.eig(train_kernel_mtx)
    w = np.maximum(w, np.zeros(w.shape))
    new_L = np.dot(np.dot(v, np.diag(np.sqrt(w))), v.T)
    print new_L.shape
    return new_L

        
def draw_many_samples():
    ds = get_ds()
    ns = get_ns()
    sigma = get_sigma()
    for d in ds:
        for n in ns:
            main(d, n, sigma, str(sigma))

def one_sample_sigma_sqrt2overN(n,d):
    sigma = np.sqrt(2.0)/n
    return main(d, n, sigma, "Sqrt2overN")

def one_sample_sigma_sqrt2overND(n,d):
    sigma = np.sqrt(2.0)/(n*d)
    return main(d, n, sigma, "Sqrt2overND")

def one_sample_sigma_sqrt2overNtosqrtD(n,d):
    sigma = np.sqrt(2.0)/(n**(d**(.5)))
    return main(d, n, sigma, "Sqrt2overND")

def one_sample_sigma_dover45(n,d):
    sigma = d/45.0
    return main(d, n, sigma, "Dover45")

def draw_many_samples_sigma_sqrt2overn():
    ds = get_ds()
    ns = get_ns()
    
    for d in ds:
        for n in ns:
            sigma = np.sqrt(2.0)/n
            main(d, n, sigma, "Sqrt2overN")


def find_largest_sigma_for_worst_matrix(n,d):
    # idea: find largest sigma s.t. cholesky decomposition works for the most degenerate possible matrix
    # turns out, this is too conservative.
    X = get_grid(d)

    new_point = 0
    X_train = X[np.array([new_point])]
    X_test = np.delete(X, new_point,axis=0)
    test_to_train_dist_mtx = build_distance_sq_matrix(X_test, X_train)

    for i in range(n-1):
        
        new_point_dist, new_point = min((val, idx) for (idx, val) in enumerate(test_to_train_dist_mtx))
        # add new point to train
        X_train = np.append(X_train, X_test[np.array([new_point])], axis=0)
        # remove new point from test
        X_test = np.delete(X_test, new_point, axis=0)
        test_to_train_dist_mtx = np.delete(test_to_train_dist_mtx, new_point, axis=0)
        

    train_dist_mtx = build_distance_sq_matrix(X_train, X_train)
    
    sigma = 100
    cond_nums = []
    while True:

        train_kernel_mtx = build_rbf_kernel(train_dist_mtx, sigma)
        cond_nums.append((np.linalg.cond(train_kernel_mtx),sigma))
        # try cholesky
        try:
            L = scipy.linalg.cholesky(train_kernel_mtx, True)
            break
        except:
            sigma = sigma / 1.1
            
    return sigma


def one_sample_search_sigma_by_sampling(n,d):
    print("you have to use the many_samples_search_sigma() function, not this one")
    #raise NotImplementedError


def many_samples_search_sigma():
    from current_experiment import *
    ns = get_ns()
    ds = get_ds()
    sampler = 'DPPPostVarSearchSigmaBySampling'

    for d in ds:
        for n in ns:
            # if samples exist, skip
            dir_path = 'pickled_data/dim={}/'.format(d)
            largest_sample_path = dir_path + 'sampler={}_n={}_d={}_samplenum={}'.format(sampler,n,d,'{}_1'.format(get_num_samples()-1))
            
            if os.path.exists(largest_sample_path):
                print('found sample at {}'.format(largest_sample_path))
                continue
            sigma = 100
            cur_samples = {}
            
            # while we can't draw a full set of samples without crashing, reduce sigma and try again
            while len(cur_samples) < get_num_samples():
                try:
                    cur_samples = many_samples_search_sigma_helper(n, d, sigma, get_num_samples())
                except np.linalg.linalg.LinAlgError:
                    print('sigma too large. reducing from {} to {}'.format(sigma, sigma / 1.1))
                    sigma = sigma / 1.1

            # save samples
            for i in range(get_num_samples()):
                pickle_loc = dir_path + 'sampler={}_n={}_d={}_samplenum={}'.format(sampler,n,d,'{}_1'.format(i))
                if os.path.exists(pickle_loc):
                    continue
                else:
                    pickle.dump(cur_samples[i], open(pickle_loc, 'wb'))
            
            # save the learned sigmas
            save_sigma_loc = 'pickled_data/searched_sigma/{}'.format(sampler)
            if os.path.exists(save_sigma_loc):
                sigmas = pickle.load(open(save_sigma_loc))
                if not d in sigmas:
                    sigmas[d] = {}
                if not n in sigmas[d]:
                    sigmas[d][n] = []
                sigmas[d][n].append(sigma)
            else:
                sigmas = {}
                sigmas[d] = {}
                sigmas[d][n] = [sigma]
            pickle.dump(sigmas, open(save_sigma_loc, 'wb'))
                    
                

def many_samples_search_sigma_helper(n,d, sigma, num_samples):
    samples = {}
    for snum in range(num_samples):
        samples[snum] = main(d,n, sigma, '')
    return samples


def load_and_print_searched_sigmas():
    import pickle
    file_loc = 'pickled_data/searched_sigma/DPPPostVarSearchSigmaBySampling'
    sigmas = pickle.load(open(file_loc))
    for d in sorted(sigmas):
        for n in sorted(sigmas[d]):
            print '{}: {}, {}'.format(n, [round(x,5) for x in sigmas[d][n]], round(2**.5/n,5))
        print('')




if __name__ == "__main__":
    #draw_many_samples_sigma_sqrt2overn()
    
    many_samples_search_sigma()


