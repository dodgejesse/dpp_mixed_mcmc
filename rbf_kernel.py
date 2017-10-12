import sklearn.metrics

class RBF_Kernel():
    def __call__(self,L,L_prime):
        return sklearn.metrics.pairwise.rbf_kernel(L,L_prime)
