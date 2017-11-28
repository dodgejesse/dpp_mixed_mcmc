"""
Taken from https://github.com/mehdidc/dpp!
Don't forget to cite
DEBUGGING
"""

import numpy as np
import time
from itertools import product
"""
Determinantal point process sampling procedures based
on  (Fast Determinantal Point Process Sampling with
     Application to Clustering, Byungkon Kang, NIPS 2013)
"""


def sample(items, L, max_nb_iterations=1000, rng=np.random):
    """
    Sample a list of items from a DPP defined
    by the similarity matrix L. The algorithm
    is iterative and runs for max_nb_iterations.
    The algorithm used is from
    (Fast Determinantal Point Process Sampling with
    Application to Clustering, Byungkon Kang, NIPS 2013)
    """
    Y = rng.choice((True, False), size=len(items))
    L_Y = L[Y, :]
    L_Y = L_Y[:, Y]
    L_Y_inv = np.linalg.inv(L_Y)

    for i in range(max_nb_iterations):
        u = rng.randint(0, len(items))

        c_u = L[u:u+1, :]
        c_u = c_u[:, u:u+1]
        b_u = L[Y, :]
        b_u = b_u[:, u:u+1]
        if Y[u] == False:
            p_include_U = min(1, c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u))
            if p_include_U > 0:
                print "p_include_U: {}".format(p_include_U)

            if rng.uniform() <= p_include_U:
                d_u = (c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u))
                upleft = (L_Y_inv +
                          np.dot(np.dot(np.dot(L_Y_inv, b_u), b_u.T),
                                 L_Y_inv) / d_u)
                upright = -np.dot(L_Y_inv, b_u) / d_u
                downleft = -np.dot(b_u.T, L_Y_inv) / d_u
                downright = d_u
                L_Y_inv = np.bmat([[upleft, upright], [downleft, downright]])
                Y[u] = True
                L_Y = L[Y, :]
                L_Y = L_Y[:, Y]
        else:
            p_remove_U = min(1, 1./(c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u)))
            if p_remove_U > 0:
                print "p_remove_U: {}".format(p_remove_U)
            if rng.uniform() <= p_remove_U:
                l = L_Y_inv.shape[0] - 1
                D = L_Y_inv[0:l, :]
                D = D[:, 0:l]
                e = L_Y_inv[0:l, :]
                e = e[:, l:l+1]
                f = L_Y_inv[l:l+1, :]
                f = f[:, l:l+1]
                L_Y_inv = D - np.dot(e, e.T) / f
                Y[u] = False
                L_Y = L[Y, :]
                L_Y = L_Y[:, Y]
    return np.array(items)[Y]


def print_unif_samps_dets(L, k, num_samps, rng):
    """
    prints the determinants of num_samps uniform samples
    """
    dets = []
    for i in range(num_samps):
        initial = rng.choice(range(len(L)), size=k, replace=False)
        X = [False] * len(L)
        for j in initial:
            X[j] = True
        X = np.array(X)
        L_Y_cur = L[X,:]
        L_Y_cur = L_Y_cur[:,X]
        dets.append(np.linalg.det(L_Y_cur))
    print("about to sort determinants...")
    sorted_dets = np.sort(dets)
    print(sorted_dets[1:500])
    print(sorted_dets[len(sorted_dets)-500:len(sorted_dets)-1])
    print('avg: {}'.format(np.average(dets)))
     

# returns alpha s.t. logdet(alpha*L_Y) > -500
def get_kernel_multiplier(L_Y):
    cur_alpha = 1
    np.linalg.slogdet(L_Y)
    

# returns a sample from a DPP defined over a mixed discrete and contiuous space
# requires unif_sampler, which is an object initialized with a dimension d,
#   that contains a function draw_sample() which returns a 1 dim feature vector
# requires dist_comp, which takes a dxk matrix (k stacked feature vectors) 
#   and returns the similarity matrix representing the similarity between all pairs
#   this is L_Y

def sample_k_disc_and_cont(unif_sampler, dist_comp, k, max_iter=None, rng=np.random):

    start_time = time.time()
    print_debug = False
    use_log_dets = k > 20

    # should figure out the best thing to do here
    if max_iter is None:
        import math
        max_iter = 5*int(len(L)*math.log(len(L)))

    #import pdb; pdb.set_trace()

    unfeat_B_Y, B_Y, L_Y = sample_initial(unif_sampler, k, dist_comp, use_log_dets)

    if print_debug:
        numerator_counter = 0
        denom_counter = 0
        num_neg_counter = 0
        denom_neg_counter = 0
        p_neg_counter = 0
        both_neg_counter = 0
    
    steps_taken = 0
    num_Y_not_invert = 0
    proposal_with_neg_det = 0
    for i in range(max_iter):
        if i % 10000 == 0 and not i == 0:
            print "iter {} out of {}, after {} seconds".format(i, max_iter, 
                                                               time.time() - start_time)
        
        u_index = rng.choice(range(len(L_Y)))
        v_unfeat_vect, v_feat_vect = unif_sampler(1)
        
        det_ratio, B_Y_prime, L_Y_prime = get_det_ratio(u_index, v_unfeat_vect, v_feat_vect, L_Y, B_Y, use_log_dets, dist_comp)
        
        det_ratio_s, B_Y_prime, L_Y_prime = get_simplified_det_ratio(B_Y, L_Y, u_index, v_feat_vect, dist_comp)

        print det_ratio
        print det_ratio_s
        print('')
        
        if det_ratio < 0:
            proposal_with_neg_det += 1

        # taken from alireza's paper
        p = .5 * min(1, det_ratio)
        

        if rng.uniform() <= p:
            steps_taken += 1
            del unfeat_B_Y[u_index]
            unfeat_B_Y.append(v_unfeat_vect)
            B_Y = B_Y_prime
            L_Y = L_Y_prime
        
        
        
        #c_v = dist_comp(cur_out, cur_out)
        #b_v = dist_comp(cur_out, cur_out)
        #c_u = L_Y[cur_in_index:cur_in_index+1,cur_in_index:cur_in_index+1]
        #b_u = L_Y[:,cur_in_index:cur_in_index+1]
    
    end_time = time.time()
    total_time = end_time - start_time
    if proposal_with_neg_det > 0:
        print "{} out of {} iters ({}%) the proposal had neg det. k={}, d={}".format(
            proposal_with_neg_det, max_iter, round(100.0*proposal_with_neg_det/max_iter,2),
            k, len(B_Y[0]))
    return unfeat_B_Y, B_Y, L_Y, total_time

    
def sample_initial(unif_sampler,k,dist_comp, use_log_dets):
    # number of resamples to try:
    resample_limit = 10000
    unfeat_B_Y, B_Y = unif_sampler(k)
    
    num_Y_resampled = 0
    L_Y = dist_comp(B_Y, B_Y)
    if use_log_dets:
        tolerance = 0
        sign, logdet = np.linalg.slogdet(L_Y)
        cur_det = sign
    else:
        tolerance = 10**-200
        cur_det = np.linalg.det(L_Y)
    best_found_cur_det = cur_det
    while cur_det < tolerance:
        unfeat_B_Y, B_Y= unif_sampler(k)
        L_Y = dist_comp(B_Y, B_Y)
        if use_log_dets:
            sign, logdet = np.linalg.slogdet(L_Y)
            cur_det = sign
        else:
            cur_det = np.linalg.det(L_Y)
        num_Y_resampled += 1
        if cur_det > best_found_cur_det:
            best_found_cur_det = cur_det
        if num_Y_resampled > resample_limit:
            import pdb; pdb.set_trace()
            out_string = "We've tried to sample Y such that L_Y is invertible (has det(L_Y) > 0)" + " but after {} samples we didn't find any with det(L_Y) > {}. The best " + "found determinant was {}."
            print(out_string.format(resample_limit,tolerance, best_found_cur_det))
            raise ZeroDivisionError("The matrix L is likely low rank => det(L_Y) = 0.")
    return unfeat_B_Y, B_Y, L_Y


def get_simplified_det_ratio(B_Y, L_Y, u_index, v_feat_vect, dist_comp):
    import pdb; pdb.set_trace()

    B_Y_minus_u = np.delete(B_Y, u_index, 0)

    L_Y_minus_u = np.delete(np.delete(L_Y, u_index, 0), u_index, 1)
    L_Y_minus_u_inv = np.linalg.inv(L_Y_minus_u)
    
    c_v = dist_comp(v_feat_vect, v_feat_vect)
    c_u = L_Y[u_index,u_index]
    
    b_v = dist_comp(v_feat_vect, B_Y_minus_u)
    b_u = np.delete(L_Y[u_index],u_index, 0)


    numer = c_v - np.dot(np.dot(b_v,L_Y_minus_u_inv),np.transpose(b_v))
    denom = c_u - np.dot(np.dot(b_u,L_Y_minus_u_inv),np.transpose(b_u))



    # to return:
    B_Y_prime = np.vstack([B_Y_minus_u, v_feat_vect])
    L_Y_prime = dist_comp(B_Y_prime, B_Y_prime)
    if numer < 0:
        det_ratio = -1
    else:
        det_ratio = (numer/denom)[0][0]
    if denom < 0:
        print("Problems in using log det for ratio!")
        print("second sign: {}, logdet: {}, k: {}, d: {}".format(
                second_sign, second_logdet,k, len(B_Y[0])))
        raise ZeroDivisionError("The matrix L is likely low rank => det(L_Y) = 0.")
    
    return det_ratio, B_Y_prime, L_Y_prime

def get_det_ratio(u_index, v_unfeat_vect, v_feat_vect, L_Y, B_Y, use_log_dets, dist_comp):
    B_Y_minus_u = np.delete(B_Y, u_index,0)
    
    
    # L_Y_prime is L_Y minus u plus v
    B_Y_prime = np.vstack([B_Y_minus_u, v_feat_vect])
        
    L_Y_prime = dist_comp(B_Y_prime, B_Y_prime)
    
    if not use_log_dets:
        numerator = np.linalg.det(L_Y_prime)
        if numerator < 0:
            numerator = -1
        denom = np.linalg.det(L_Y)
        det_ratio = numerator/denom
    else:
        # using det(L_Y_prime) / det(L_Y) = 
        # exp(log(det(L_Y_prime)) - log(det(L_Y)))
        (first_sign, first_logdet) = np.linalg.slogdet(L_Y_prime)
        (second_sign, second_logdet) = np.linalg.slogdet(L_Y)
        det_ratio = np.exp(first_logdet - second_logdet)
        # diagnostic things
        if first_sign < 0:
            #print("Problems in using log det for ratio!")
            #print("first sign: {}, logdet: {}, k: {}, d: {}".format(
            #        first_sign, first_logdet, k, len(B_Y[0])))
            det_ratio = -1
        if second_sign < 0:
            print("Problems in using log det for ratio!")
            print("second sign: {}, logdet: {}, k: {}, d: {}".format(
                    second_sign, second_logdet,k, len(B_Y[0])))
            raise ZeroDivisionError("The matrix L is likely low rank => det(L_Y) = 0.")
    return det_ratio, B_Y_prime, L_Y_prime



# returns the determinant of the submatrix of L defined by X
def det_X(X,L):
    L_Y_cur = L[X,:]
    L_Y_cur = L_Y_cur[:,X]
    return np.linalg.det(L_Y_cur)


def sample_discrete_L(L,k,rng,items):
    initial = rng.choice(range(len(items)), size=k, replace=False)
    X = [False] * len(items)
    for i in initial:
        X[i] = True
    X = np.array(X)
    return X


def sample_k(items, L, k, max_nb_iterations=None, rng=np.random):
    """
    Sample a list of k items from a DPP defined
    by the similarity matrix L. The algorithm
    is iterative and runs for max_nb_iterations.
    The algorithm used is from
    (Fast Determinantal Point Process Sampling with
    Application to Clustering, Byungkon Kang, NIPS 2013)
    """
    #import pdb; pdb.set_trace()
    # if L is infinite (some dims of the space are continuous)
    sample_continuous = type(L) == type({})

    print_debug = False

    if max_nb_iterations is None:
        import math
        max_nb_iterations = 5*int(len(L)*math.log(len(L)))
    
    if not sample_continuous:        
        X = sample_discrete_L(L,k,rng,items)
    else:
        initial = sample_continuous_L(L,k)


    # if Y has very close to zero determinant, resample it
    num_Y_resampled = 0
    tolerance = 10**-100
    while det_X(X, L) < tolerance:
        initial = rng.choice(range(len(items)), size=k, replace=False)
        X = [False] * len(items)
        for i in initial:
            X[i] = True
        X = np.array(X)
        num_Y_resampled += 1
        if num_Y_resampled > (1.0/2)*len(L):
            print("We've tried to sample Y such that L_Y is invertible (has det(L_Y) > 0)" + 
                  " but after {} samples we didn't find any with det(L_Y) > {}.".format(
                      (1.0/2)*len(L),tolerance))
            raise ZeroDivisionError("The matrix L is likely low rank => det(L_Y) = 0.")

    if print_debug:
        numerator_counter = 0
        denom_counter = 0
        num_neg_counter = 0
        denom_neg_counter = 0
        p_neg_counter = 0
        both_neg_counter = 0
        
    steps_taken = 0
    num_Y_not_invert = 0
    for i in range(max_nb_iterations):
        
        u = rng.choice(np.arange(len(items))[X])
        v = rng.choice(np.arange(len(items))[~X])
        Y = X.copy()
        Y[u] = False
        L_Y = L[Y, :]
        L_Y = L_Y[:, Y]

        # to check determinants
        if print_debug:
            Y_cur = X.copy()
            L_Y_cur = L[Y_cur,:]
            L_Y_cur = L_Y_cur[:,Y_cur]
            
            Y_next = X.copy()
            Y_next[u] = False
            Y_next[v] = True
            L_Y_next = L[Y_next,:]
            L_Y_next = L_Y_next[:,Y_next]

                  
        try:
            L_Y_inv = np.linalg.inv(L_Y)
        except:
            num_Y_not_invert += 1
            continue
            #import pdb; pdb.set_trace()



        c_v = L[v:v+1, :]
        c_v = c_v[:, v:v+1]
        b_v = L[Y, :]
        b_v = b_v[:, v:v+1]
        c_u = L[u:u+1, :]
        c_u = c_u[:, u:u+1]
        b_u = L[Y, :]
        b_u = b_u[:, u:u+1]


        numerator = c_v - np.dot(np.dot(b_v.T, L_Y_inv.T), b_v)
        denom = c_u - np.dot(np.dot(b_u.T, L_Y_inv.T), b_u)

        if print_debug:
            if numerator < 0 and denom > 0:
                num_neg_counter += 1
            if numerator < 10**-9:
                numerator_counter += 1
            if denom < 0 and numerator > 0:
                denom_neg_counter += 1
            if denom < 10**-9:
                denom_counter += 1
        
            if numerator < 0 and denom < 0:
                both_neg_counter += 1


        p = 0.5 * min(1,  numerator/denom)
        
        # to print if we have some problems with small or zero determinants / eigenvalues
        if print_debug:
            if numerator < 0 or denom < 0 or p < 0:
                print i, p, numerator, denom#u, v, [j for j, b_var in enumerate(Y) if b_var]
                print("{}\t->\t{}".format(np.linalg.det(L_Y_cur), np.linalg.det(L_Y_next)))
                print("steps taken so far: {}, {}%".format(steps_taken, round(100.0*steps_taken/i,3)))
                #import pdb; pdb.set_trace()

        if rng.uniform() <= p:
            steps_taken += 1
            X = Y[:]
            X[v] = True
            
            if print_debug:
                
                print("{}\t->\t{}".format(np.linalg.det(L_Y_cur),np.linalg.det(L_Y_next)))

    if print_debug:
        print("num numerators that would be rounded to zero: {}".format(numerator_counter))
        print("num denoms that would be rounded to zero: {}".format(denom_counter))
        print("num_neg_counter: {}".format(num_neg_counter))
        print("denom_neg_counter: {}".format(denom_neg_counter))
        print("both_neg_counter: {}".format(both_neg_counter))
        print("steps taken: {}".format(steps_taken))
        

    if num_Y_not_invert > .5 * max_nb_iterations:
        print("We've tried to sample Y such that L_Y is invertible (has det(L_Y) > 0)" + 
              " but after {} potential mcmc steps, we found L_Y not invertible {} times.".format(
                  .5 * max_nb_iterations, num_Y_not_invert))
        raise ZeroDivisionError("The matrix L is likely low rank => det(L_Y) = 0.")

    if steps_taken == 0:
        print("We ran the MCMC algorithm for {} steps, but it never accepted a metropolis-hastings " + 
              "proposal, so this is just a uniform sample.".format(steps_taken))
        raise ZeroDivisionError("It's likely the matrix L is bad. The MCMC algorithm failed.")

    print("{} steps taken by mcmc algorithm, out of {} possible steps. {}%".format(steps_taken, 
                                    max_nb_iterations, 100.0*steps_taken/max_nb_iterations))
    return np.array(items)[X]





# taken from dpp.py. it's for debugging.
def construct_L_debug(num_points):
    step_size = 1.0/(num_points-1)
    L_debug = []
    for i in range(num_points):
        cur_row = []
        for j in range(num_points):
            cur_row.append(1-abs(step_size*(i-j)))
        L_debug.append(cur_row)
        
    items = []
    for i in range(num_points):
        items.append(i)
    return items, np.asarray(L_debug)
    


# this was taken from dpp.py. it's for debugging
def debug_mcmc(d_space, L):
    np.set_printoptions(linewidth=20000)
    import dpp_mcmc_sampler

    items, L_debug = construct_L_debug(10)
    
    #things = dpp_mcmc_sampler.sample_k(items, L_debug, 3)


    items, L_debug = construct_L_debug(4080)
    #dpp_mcmc_sampler.sample_k(items, L, 3)
    #dpp_mcmc_sampler.sample_k(items, L, 4)
    dpp_mcmc_sampler.sample_k(items, L, 7)
    import pdb; pdb.set_trace()
    #L_small = np.array([[1,.6,.3],[.6,1,.6],[.3,.6,1]])
    #items = ['1','2','3']

    #things = dpp_mcmc_sampler.sample_k(items, L_small, 2)
    
    #L_s = np.array([[1,.9,.8,.7],[.9,1,.9,.8],[.8,.9,1,.9],[.7,.8,.9,1]])
    #items = ['1','2','3','4']

    
    
    #things = dpp_mcmc_sampler.sample_k(items, L_s, 2)


    #things = dpp_mcmc_sampler.sample_k(d_space, L, 5)
