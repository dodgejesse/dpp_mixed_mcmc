import numpy
import pickle, glob


def get_fname(sampler, n, d, sample_num):
    return 'pickled_data/all_samples/sampler={}_n={}_d={}_samplenum={}'.format(sampler,n,d,sample_num)
    

def get_out_name(sampler, eval_measure):
        return 'pickled_data/all_samples_errors/sampler={}_eval={}'.format(sampler, eval_measure)

        

def remove_errors(samplers, eval_measures, ns, ds):
        for sampler in samplers:
                for eval_measure in eval_measures:
                        with open(get_out_name(sampler, eval_measure), 'rb') as pickle_file:
                                cur_evals = pickle.load(pickle_file)
                                pickle_file.close()
                        if ds == 'all':
                                ds = cur_evals.keys()
                        for d in ds:
                                if d not in cur_evals: 
                                        continue

                                if ns == 'all':
                                        ns = cur_evals[d].keys()
                                for n in ns:
                                        if n not in cur_evals[d]:
                                                continue
                                        num_del = len(cur_evals[d][n])
                                        del cur_evals[d][n]
                                        print('deleted {}, eval={}, d={}, n={}. num_deleted={}'.format(sampler, eval_measure, d, n, num_del))
                                if len(eval_measure[d]) == 0:
                                        del cur_evals[d]
                                        print('deleted {}, eval={}, d={}'.format(sampler, eval_measure, d))



                        with open(get_out_name(sampler, eval_measure), 'wb') as pickle_file:
                                pickle.dump(cur_evals, pickle_file)

                                                
                                                
                        #print cur_evals.keys()






samplers = {#'SobolSampler',
	    #'RecurrenceSampler',
	    #'SobolSamplerNoNoise': {'fn': SobolSamplerNoNoise,'color': 'b'},
	    #'DPPnsquared',#: {'fn': dpp_rbf_unitcube.DPPSampler, 'color': 'k'},
	    #'UniformSampler',
            #'DPPNarrow',
            #'DPPVNarrow',
            'DPPVVNarrow',
            'DPPVVVNarrow',
            #'DPPNNarrow',
            #'DPPNNNarrow'
            #'DPPNsquaredNarrow'
            #'DPPClipped': {'fn': dpp_rbf_unitcube.DPPClippedSampler, 'color': 'm'}
    }

eval_measures = ['l2', 'l2_cntr','discrep']


n_max = 150
#ns = [int(numpy.exp(x)) for x in numpy.linspace(0, numpy.log(n_max), 20)]
#ns = sorted(list(set(ns)))
ns = range(n_max+1)
ns = 'all'
ds = [1]#,2,3,4,5]


#compute_errors(samplers, eval_measures, ns, ds)
remove_errors(samplers, eval_measures, ns, ds)
