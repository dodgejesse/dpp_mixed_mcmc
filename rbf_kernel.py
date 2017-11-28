import sklearn.metrics
import numpy as np

class RBF_Kernel():
    # gamma is the bandwidth parameter, alpha is the scaling parameter
    def __init__(self, gamma = None, alpha = None):
        self.g = gamma
        self.alpha = alpha

    def __call__(self,B,B_prime,g=None):
        #g=1.0/(B.shape[1]*B.shape[0])
        # default for gamma is 1/d
        if not self.g is None:
            L_Y = sklearn.metrics.pairwise.rbf_kernel(B,B_prime,gamma=self.g)
        else:
            L_Y = sklearn.metrics.pairwise.rbf_kernel(B,B_prime,gamma=g)



        #for i in range(15):
        #    print np.linalg.slogdet(L_Y*(10**i)), np.linalg.det(L_Y*(10**i)), 10**i
        #import pdb; pdb.set_trace()
        #if not self.alpha is None:
        #    L_Y = self.alpha * L_Y
        return L_Y



class RBF_Clipped_Kernel():
    def __init__(self, clip_type):
        self.clip_type=clip_type


    def __call__(self, B, B_prime, gamma=None):
        #import pdb; pdb.set_trace()
        d = len(B[0])
        B = np.array(B)
        B_prime = np.array(B_prime)
        L_Y = []

        assert self.clip_type == 'k' or self.clip_type == 'd'
        if self.clip_type == "k":
            # epsilon = 1-sqrt(1-(fraction of n points we want this to depend on)^d)
            # if we want each point to have non-zero distance with half the others, assuming uniform:
            # epsilon = 1-sqrt(1-(1/2)^d)
            frac_of_k = 1.0/2 # could also try sqrt(n)/n = 1/sqrt(n)
            epsilon = 1-np.sqrt(1-(1-frac_of_k**(1.0/d)))

        elif self.clip_type == "d":
            # epsilon = 1-1.0/len(B)
            # to keep expected number of non-zero dimension distances to p:
            # 1/2 +- sqrt(1+p/d)/2  <--- might be wrong
            # to keep the expected number of non-zero distances to 2:
            # 1-sqrt(1-2/d)
            # another idea: set so only half of the points influence a given point, in expect
            frac_of_d = 2.0/d
            epsilon = 1-np.sqrt(1-frac_of_d)

        
        if gamma is None:
            gamma = 1.0/d
        for i in range(len(B_prime)):
            to_be_summed = np.square((B-B[i])*(abs(B-B[i])<epsilon)+(abs(B-B[i])>epsilon))
            L_Y.append(np.exp(-gamma*np.sum(to_be_summed,axis=1)))
            
        return L_Y

