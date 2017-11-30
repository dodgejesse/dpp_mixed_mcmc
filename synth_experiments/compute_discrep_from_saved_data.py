import numpy
import pickle, glob

# i don't think i care about this
def get_discrepency(X):
        worse_value = 0.
	for x in X:
		p = numpy.exp(numpy.sum(numpy.log(x)))
		p_hat = 0.
		for y in X:
			p_hat += float(all(y < x))
		p_hat /= len(X)
		worse_value = max(worse_value, abs(p-p_hat))
		worse_value = max(worse_value, abs(p-p_hat-1./len(X)))
	return worse_value

def get_min_norm(X, order):
	return min(numpy.linalg.norm(X, ord=order, axis=1))

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
	center = numpy.ones(X.shape)*.5
	return get_min_norm(X-center, 2)

def get_min_l1_norm_center(X):
	center = numpy.ones(X.shape)*.5
	return get_min_norm(X-center, 1)

def get_fname(sampler, n, d, sample_num):
    return 'pickled_data/all_samples/sampler={}_n={}_d={}_samplenum={}'.format(sampler,n,d,sample_num)
    

def compute_errors(samplers, eval_measures, ns, ds):
        #import pickle
    for sample_counter in range(50):
        print(sample_counter)
        for sample_subcounter in range(1,4):
            sample_num = '{}_{}'.format(sample_counter, sample_subcounter)

            sampler_to_n_err = {}
            for sampler in samplers:
		n_to_d_err = {}
		for n in ns:
                    d_to_measure_err = {}
                    for d in ds:
                        measure_to_err = {}
                        file_name = get_fname(sampler, n, d, sample_num)
                        print file_name
                        pkl_file = open(file_name)
                        X = pickle.load(pkl_file)

                        
                        for measure in eval_measures:
                            measure_to_err[measure] = eval_measures[measure](X)
                        d_to_measure_err[d] = measure_to_err
                    n_to_d_err[n] = d_to_measure_err
                    #print("finishd n={}, sampler={}".format(n, sampler))
		sampler_to_n_err[sampler] = n_to_d_err


            #import pdb; pdb.set_trace()


            pickle_loc = 'pickled_data/errors_from_samples/nmax={}_nsamplers={}_neval={}_d={}_samplenum={}'.format(n_max,
			  len(samplers), len(eval_measures), ''.join(str(e)+',' for e in ds)[:-1], sample_num)
	
            pickle.dump(sampler_to_n_err, open(pickle_loc, 'wb'))



def get_out_name(sampler, eval_measure):
        return 'pickled_data/all_samples_errors/sampler={}_eval={}'.format(sampler, eval_measure)

def get_in_name(sampler, n, d):
        return './pickled_data/all_samples/sampler={}_n={}_d={}_samplenum=*'.format(sampler, n,d)


def find_actual_samples(ns, sampler, n_or_d):
        sampled_ns = set()
        
        for f_name in glob.glob('./pickled_data/all_samples/sampler={}_*'.format(sampler)):

                n_and_stuff = f_name.split(n_or_d + '=')[-1]
                n = int(n_and_stuff.split('_')[0])
                
                if n in ns and n not in sampled_ns:
                        sampled_ns.add(n)


        return sampled_ns

        
        

def compute_individual_errors(samplers, eval_measures, ns_try, ds_try):
        for sampler in samplers:
                ns = find_actual_samples(ns_try, sampler, 'n')
                ds = find_actual_samples(ds_try, sampler, 'd')


                for eval_measure in eval_measures:
                        # if saved evaluations exist
                        try:
                                with open(get_out_name(sampler, eval_measure), 'rb') as pickle_file:
                                        cur_evals = pickle.load(pickle_file)
                                        pickle_file.close()
                        except:
                                cur_evals = {}



                        
                        for d in ds:
                                for n in ns:

                                        in_file_names = glob.glob(get_in_name(sampler, n,d))
                                        print "sampler={}, eval={}, n={}, d={}, samples={}".format(sampler, eval_measure, n, d, len(in_file_names))

                                        for in_file_name in in_file_names:


                                                if d not in cur_evals:
                                                        cur_evals[d] = {}
                                                if n not in cur_evals[d]:
                                                        cur_evals[d][n] = {}

                                                
                                                sample_num = in_file_name.split("=")[-1]
                                                if sample_num not in cur_evals[d][n]:
                                                        pkl_file = open(in_file_name)
                                                        cur_sample = pickle.load(pkl_file)
                                                        # compute eval
                                                        cur_evals[d][n][sample_num] = eval_measures[eval_measure](cur_sample)

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
            #'DPPVVVNarrow',
            #'DPPNNarrow',
            #'DPPNNNarrow'
            #'DPPNsquaredNarrow'
            #'DPPClipped': {'fn': dpp_rbf_unitcube.DPPClippedSampler, 'color': 'm'}
    }

eval_measures = {'l2':get_min_l2_norm, 
		 #'l1':get_min_l1_norm, 
		 'l2_cntr':get_min_l2_norm_center, 
		 #'l1_cntr':get_min_l1_norm_center,
                 'discrep':get_discrepency}
n_max = 150
#ns = [int(numpy.exp(x)) for x in numpy.linspace(0, numpy.log(n_max), 20)]
#ns = sorted(list(set(ns)))
ns = range(n_max+1)

ds = [1]#,2,3,4,5]


#compute_errors(samplers, eval_measures, ns, ds)
compute_individual_errors(samplers, eval_measures, ns, ds)
