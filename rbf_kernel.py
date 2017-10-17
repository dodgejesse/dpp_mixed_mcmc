import sklearn.metrics

class RBF_Kernel():
    def __call__(self,L,L_prime,g=1):
        return sklearn.metrics.pairwise.rbf_kernel(L,L_prime,gamma=10)
