import numpy as np
import os
import time
import scipy.linalg
import pickle
import sys
from current_experiment import *
import predicted_sigmas

    
    
def build_rbf_kernel(D, sigma):
    gamma = .5/(sigma*sigma)
    kernel_mtx = np.exp(-gamma*D)
    return kernel_mtx

def build_distance_sq_matrix(X, Z):
    return np.outer(np.sum(X**2, axis=1), np.ones(Z.shape[0])) -2*np.dot(X, Z.T) + np.outer(np.ones(X.shape[0]), np.sum(Z**2, axis=1))


def set_global_storage_objects():
    global first_line_stored
    global third_line_stored
    third_line_stored = {}
    first_line_stored = {}


# to make this faster, precompute M_i_ab
def new_main(D=5,k=100,sigma=0.19814):
    debug_print = False
    import time
    epsilon = 0.0001
    max_cond_num = 10**6
    X = np.random.random((1,D))
    actual_start_time = time.time()

    set_global_storage_objects()
    for i in range(k-1):
        global time_spent_in_methods
        time_spent_in_methods = [[],[],[]]
        start_time = time.time()
        distances = build_distance_sq_matrix(X,X)
        post_distance_time = time.time()
        K = build_rbf_kernel(distances, sigma)
        post_rbf_time = time.time()
        K_inv = np.linalg.inv(K) # THIS IS GOING TO BE PROBLEMATIC!

        cond_num = np.linalg.cond(K_inv)
        if debug_print:
            print("condition number of K_inv: {:.6f}".format(cond_num))
        if cond_num > max_cond_num:
            raise np.linalg.linalg.LinAlgError
        post_inv_time = time.time()
        
        M_i = np.exp(-distances / (4.0*sigma**2))

        M_iK_inv = np.multiply(M_i, K_inv)

        #import pdb; pdb.set_trace()
        
        m_ab = compute_m_ab(X, sigma)
        
        x_i = np.asarray([])
        for d in range(D):
            d_start_time = time.time()
            scaling_factor = get_prob(x_i, 1, X, D, sigma, M_iK_inv, m_ab)
            d_post_scaling_factor = time.time()
            u = scaling_factor * np.random.random()
            I = [0,1]
            while I[1] - I[0] > epsilon:
                avg = .5*(I[0]+I[1])
                if get_prob(x_i, avg, X, D, sigma, M_iK_inv, m_ab) < u:
                    I = [avg, I[1]]
                else:
                    I = [I[0], avg]
            x_i = np.append(x_i, np.random.uniform(I[0], I[1]))
            d_post_search = time.time()
            if debug_print:
                print("\t one_get_prob={:.6f}, one_search={:.6f}, total_{:.0f}th_dim={:.6f}".format(d_post_scaling_factor - d_start_time, d_post_search - d_post_scaling_factor, d, d_post_search-d_start_time))

        if debug_print:
            print("build_distance_mtx={:.6f}, build_rbf_from_distance={:.6f}, inverse={:.6f}, total_for_{:.0f}th_sample={:.6f}".format(post_distance_time-start_time, post_rbf_time - post_distance_time, post_inv_time - post_rbf_time, i, d_post_search-start_time))
            print("{}, {}, {}".format(sum(time_spent_in_methods[0]),sum(time_spent_in_methods[1]),sum(time_spent_in_methods[2])))
        X = np.vstack((X, x_i))

    if debug_print:
        #print(X)
        print("final condition number: {}".format(np.linalg.cond(K_inv)))
        print("total time: {}".format(time.time() - actual_start_time))
    return X
    
def get_prob(x_i, x_i_d, X, D, sigma, M_iK_inv, m_ab):
    global time_spent_in_methods
    a_b_sum = 0
    # resetting the storage for a new x_i_d
    global second_line_stored
    second_line_stored = {}

    for a in range(len(X)):
        for b in range(len(X)):
            first_start = time.time()
            first_equation_line = get_first_line_of_eq(x_i, a, b, D, sigma, M_iK_inv, m_ab)
            second_start = time.time()
            second_equation_line = get_second_line_of_eq(x_i_d, a, b, len(x_i), sigma, m_ab)
            third_start = time.time()
            third_equation_line = get_third_line_of_eq(a, b, len(x_i), D, sigma, m_ab)
            third_end = time.time()
            
            a_b_sum += first_equation_line * second_equation_line * third_equation_line

            
            time_spent_in_methods[0].append(second_start - first_start)
            time_spent_in_methods[1].append(third_start - second_start)
            time_spent_in_methods[2].append(third_end - third_start)
            
            
    return x_i_d - a_b_sum
    
def get_first_line_of_eq(x_i, a, b, D, sigma, M_iK_inv, m_ab):
    global first_line_stored
    d = len(x_i)

    #import pdb; pdb.set_trace()
    if d not in first_line_stored:
        #sum_up_to_d_minus_one = 0
        #for r in range(d):
        #    sum_up_to_d_minus_one += (x_i[r] - m_ab[a][b][r])**2
        #expd_sum = np.exp(-sum_up_to_d_minus_one/sigma**2)
        #return_val = expd_sum * M_iK_inv[a][b]

        diffs = (x_i - m_ab[:,:,0:d])**2
        sum_diffs = diffs.sum(axis=2)
        exp_sum = np.exp(-sum_diffs/sigma**2)
        final = np.multiply(exp_sum,M_iK_inv)

        first_line_stored[d] = final

        #diff_sq = (x_i - m_ab[:,:,0:d])**2
        #exp_sum = np.exp(diff_sq/sigma**2)
        #final = exp_sum.multiply(M_iK_inv)

        #if stored_val:
        #    #print("made it here!")
        #    if not first_line_stored[a][b][d] == return_val:
        #        import pdb; pdb.set_trace()
        #        #assert first_line_stored[a][b][d] == return_val

        #first_line_stored[a][b][d] = return_val
        
        #assert return_val == final[a][b]




    #if len(m_ab) > 3 and len(x_i) > 1:
    #    import pdb; pdb.set_trace()
    return first_line_stored[d][a][b]

def get_second_line_of_eq(x_i_d, a,b,d,sigma, m_ab):
    #if a > 3 and b > 3:
    #    import pdb; pdb.set_trace()
        
    global second_line_stored
    if len(second_line_stored) == 0:
        first_part = (x_i_d - m_ab[:,:,d])/sigma
        erf_first_part = scipy.special.erf(first_part)
        # first_part.shape should be something like (a,b)
        
        global second_line_second_part_stored

        third_term = np.sqrt(np.pi)*sigma/2.0
        second_line_stored = (erf_first_part + second_line_second_part_stored[:,:,d]) * third_term


    new_computation =  second_line_stored[a][b]
    return new_computation




def get_third_line_of_eq(a,b,d,D,sigma, m_ab):
    global third_line_stored
    stored_val = get_stored_value(a,b,d, third_line_stored)
    if not stored_val:
        prod_val = 1
        for l in range(d+1,D):
            first_term = scipy.special.erf((1 - m_ab[a][b][l])/sigma)
            second_term = scipy.special.erf(m_ab[a][b][l]/sigma)
            third_term = np.sqrt(np.pi)*sigma/2.0
            prod_val = prod_val * (first_term + second_term) * third_term


        third_line_stored[a][b][d] = prod_val

        return prod_val
    else:
        return stored_val

def get_stored_value(a,b,d, storage):
    if a not in storage:
        storage[a] = {}
    if b not in storage[a]:
        storage[a][b] = {}
    if d not in storage[a][b]:
        return None
    # else, storage[a][b][d] exists, and we retrieve it
    return storage[a][b][d]
        
            
def compute_m_ab(X, sigma):
    m_ab = np.zeros((X.shape[0], X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            m_ab[i][j] = .5*(X[i] + X[j])

    global second_line_second_part_stored
    second_line_second_part_stored = scipy.special.erf(m_ab/sigma)
    global first_line_stored
    first_line_stored = {}
    return m_ab

def predicted_sigmas_tentosix(n,d):
    sigma = predicted_sigmas.get_sigma(n,d)
    
    if sigma is None:
        return None
    else:
        to_return = None
        while to_return is None:
            try:
                to_return = new_main(d,n,sigma)
            except np.linalg.linalg.LinAlgError:
                print("sigma too big! trying again.")
            

        return to_return

def sigma_sqrt2overN(n,d):
    sigma = np.sqrt(2.0)/n
    return new_main(d, n, sigma)
    

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


def many_samples_search_sigma():
    from current_experiment import *
    ns = get_ns()
    ds = get_ds()
    sampler = 'DPPExactSearchSigmaBySamplingCondTenToSix'


    for n in ns:
        for d in ds:
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
                    # might crash if sigma is too large
                    cur_samples = {}
                    for snum in range(get_num_samples()):
                        cur_samples[snum] = new_main(d,n, sigma)

                except np.linalg.linalg.LinAlgError:
                    print('sigma too large. reducing from {:.6f} to {:.6f} after {} being drawn'.format(float(sigma), sigma / 1.1, len(cur_samples)))
                    sigma = sigma / 1.1
                
            print('')
            print('-------------------------------------------------------------------------------------------------------------------')
            print('successfully drew {} samples, with sigma={}, d={}, n={}.'.format(get_num_samples(), round(sigma,5), d, n))
            print('-------------------------------------------------------------------------------------------------------------------')
            print('')
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
                    
                
def load_and_print_searched_sigmas():
    import pickle
    file_loc = 'pickled_data/searched_sigma/DPPExactSearchSigmaBySamplingCondTenToSix'
    sigmas = pickle.load(open(file_loc))
    for d in sorted(sigmas):
        for n in sorted(sigmas[d]):
            print 'd={},k={}: {}'.format(d,n, [round(x,5) for x in sigmas[d][n]])
        print('')

    for d in sorted(sigmas):
        
        out_string = "d={}, [".format(d)
        for n in sorted(sigmas[d]):
            out_string += "[{},{}]".format(n,round(sigmas[d][n][0],5)) + ","
        out_string += "]"
        print(out_string)




if __name__ == "__main__":
    #draw_many_samples_sigma_sqrt2overn()

    #for i in range(20):
    #    sigma_sqrt2overN(i+2,3)
    many_samples_search_sigma()
    #load_and_print_searched_sigmas()
    #new_main()


