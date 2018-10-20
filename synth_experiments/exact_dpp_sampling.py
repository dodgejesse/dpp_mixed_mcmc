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

# to make this faster, precompute M_i_ab
def new_main(D=8,k=50,sigma=.01):
    import time
    epsilon = 0.0001
    #import pdb; pdb.set_trace()    
    X = np.random.random((1,D))

    
    for i in range(k-1):
        global time_spent_in_methods
        time_spent_in_methods = [[],[],[]]
        start_time = time.time()
        distances = build_distance_sq_matrix(X,X)
        post_distance_time = time.time()
        K = build_rbf_kernel(distances, sigma)
        post_rbf_time = time.time()
        K_inv = np.linalg.inv(K) # THIS IS GOING TO BE PROBLEMATIC!
        post_inv_time = time.time()
        x_i = np.asarray([])
        for d in range(D):
            d_start_time = time.time()
            scaling_factor = get_prob(x_i, 1, X, D, sigma, K_inv)
            d_post_scaling_factor = time.time()
            u = scaling_factor * np.random.random()
            I = [0,1]
            while I[1] - I[0] > epsilon:
                avg = .5*(I[0]+I[1])
                if get_prob(x_i, avg, X, D, sigma, K_inv) < u:
                    I = [avg, I[1]]
                else:
                    I = [I[0], avg]
            x_i = np.append(x_i, .5*(I[0] + I[1]))
            d_post_search = time.time()
            print("\t one_get_prob={:.6f}, one_search={:.6f}, total_{:.0f}th_dim={:.6f}".format(d_post_scaling_factor - d_start_time, d_post_search - d_post_scaling_factor, d, d_post_search-d_start_time))
        #import pdb; pdb.set_trace()
        print("build_distance_mtx={:.6f}, build_rbf_from_distance={:.6f}, inverse={:.6f}, total_for_{:.0f}th_sample={:.6f}".format(post_distance_time-start_time, post_rbf_time - post_distance_time, post_inv_time - post_rbf_time, i, d_post_search-start_time))
        print("{}, {}, {}".format(sum(time_spent_in_methods[0]),sum(time_spent_in_methods[1]),sum(time_spent_in_methods[2])))
        X = np.vstack((X, x_i))
        print(d, i+1)
    print(X)
    return X
    
def get_prob(x_i, x_i_d, X, D, sigma, K_inv):
    global time_spent_in_methods
    a_b_sum = 0
    for a in range(len(X)):
        for b in range(len(X)):
            first_start = time.time()
            first_equation_line = get_first_line_of_eq(x_i, a, b, X, D, sigma, K_inv)
            second_start = time.time()
            second_equation_line = get_second_line_of_eq(x_i_d, a, b, X, len(x_i), sigma)
            third_start = time.time()
            third_equation_line = get_third_line_of_eq(a, b, X, len(x_i), D, sigma)
            third_end = time.time()
            
            a_b_sum += first_equation_line * second_equation_line * third_equation_line

            
            time_spent_in_methods[0].append(second_start - first_start)
            time_spent_in_methods[1].append(third_start - second_start)
            time_spent_in_methods[2].append(third_end - third_start)
            
            
    return x_i_d - a_b_sum
    
def get_first_line_of_eq(x_i, a, b, X, D, sigma, K_inv):
    sum_up_to_d_minus_one = 0
    for r in range(len(x_i)):
        sum_up_to_d_minus_one += (x_i[r] - m_a_b(a,b,X,r))**2
    expd_sum = np.exp(-sum_up_to_d_minus_one/sigma**2)
    M_i_ab = get_M_i_ab(a,b,X,D,sigma)
    return expd_sum * M_i_ab * K_inv[a][b]

def get_second_line_of_eq(x_i_d, a,b,X,d,sigma):

    first_term = scipy.special.erf((x_i_d - m_a_b(a,b,X,d))/sigma)
    second_term = scipy.special.erf(m_a_b(a,b,X,d)/sigma)
    third_term = np.sqrt(np.pi)*sigma/2.0
    return (first_term + second_term) * third_term

def get_third_line_of_eq(a,b,X,d,D,sigma):
    prod_val = 1
    for l in range(d+1,D):
        first_term = scipy.special.erf((1 - m_a_b(a,b,X,l))/sigma)
        second_term = scipy.special.erf(m_a_b(a,b,X,l)/sigma)
        third_term = np.sqrt(np.pi)*sigma/2.0
        prod_val = prod_val * (first_term + second_term) * third_term
    return prod_val
    
def get_M_i_ab(a,b,X,D,sigma):
    s = 0
    for d in range(D):
        s += (X[a][d] - X[b][d])**2
    return np.exp(-s/4.0*sigma**2)

def m_a_b(a,b,X,d):
    return .5*(X[a][d]+X[b][d])
        



def sigma_sqrt2overN(n,d):
    sigma = np.sqrt(2.0)/n
    return new_main(d, n, sigma)








def old_main(d, k, sigma, sigma_name):
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


def old_many_samples_search_sigma():
    #from current_experiment import *
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
    new_main()
    


