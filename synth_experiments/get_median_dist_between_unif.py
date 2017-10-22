import numpy as np
import sklearn.metrics


ns = range(1,56,2)
ds = [2, 5, 10]
num_resamples=500



for d in ds:
    for n in ns:
        medians = []
        for resamp in range(num_resamples):

            medians.append(np.median(sklearn.metrics.pairwise.euclidean_distances(np.random.rand(n,d))))
        print n, np.mean(medians)*np.mean(medians)
                        
        
