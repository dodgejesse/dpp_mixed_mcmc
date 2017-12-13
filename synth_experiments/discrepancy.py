import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)
import sobol_seq as sobol
import dpp_rbf_unitcube


PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

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



# adds uniform noise in [0,1], then for those results above 1 it subtracts 1
def SobolSampler(n, d):
	X = sobol.i4_sobol_generate(d, n)
	U = numpy.random.rand(d)
	X += U
	X -= numpy.floor(X)
	return X

def SobolSamplerNoNoise(n, d):
	X = sobol.i4_sobol_generate(d, n)
	return X

def RecurrenceSampler(n, d):
	X = numpy.zeros((n, d))
	X[0] = numpy.random.rand(d)
	for i in range(1,n):
		for k in range(d):
			X[i, k] = X[0,k] + i*numpy.sqrt(PRIMES[k])
			X[i, k] -= int(X[i, k])
	return numpy.array(X)


def HyperRectangleTarget(d, vol):
	sides = numpy.array([2]*d)
	counter = 0
	while sum(sides > 1):

		sides_unnorm = numpy.random.rand(d)
		scaling_factor = numpy.power(vol / numpy.prod(sides_unnorm), 1.0/d)
		sides = sides_unnorm * scaling_factor
		

		counter += 1
		if counter % 1000 == 0:
			print ("{} rectangles sampled and thrown out".format(counter))
	max_rect_fits = [1]*d - sides
	min_corner = numpy.random.uniform([0]*d, max_rect_fits)
	max_corner = min_corner + sides
	return min_corner, max_corner
	


	
	
	
# returns the fraction of rectangles which contain at least one point in X
def FracInRects(X, rects):
	
	num_successes = 0
	for rect in rects:
		above_min = numpy.sum(rect[0] > X, axis=1) == 0
		below_max = numpy.sum(rect[1] < X, axis=1) == 0
		num_successes += sum(numpy.logical_and(above_min, below_max)) > 0
	return num_successes * 1.0 / len(rects)

def plot_stuff(cur_ax, alg_to_successes, num_rects, d, max_k, vol, samplers):
	unif_success = {}
	for k in range(max_k):
		unif_success[k] = 1-numpy.power(1-vol,k)

	for sampler in samplers:
		cur_ax.plot(alg_to_successes[sampler].keys(), alg_to_successes[sampler].values(), color=samplers[sampler]['color'])

	
	cur_ax.plot(unif_success.keys(), unif_success.values(), color='y')
	cur_ax.set_title('d={} vol={}'.format(d, vol))

# these values will plot until random has a > max_unif_prob chance of having one success
def GetMaxKs(vols, max_unif_prob):
	max_ks = {}
	for vol in vols:
		max_ks[vol] = int(numpy.ceil(numpy.log(1-max_unif_prob)/numpy.log(1-vol)))
	return max_ks


def random_rect(samplers):
	num_rects = 200
	ds = [3,5,10]
	vols = [0.05]
	maxp_unif = 0.90


	import sys
	pickle_result = len(sys.argv) > 1
	if pickle_result:
		num_rects = int(sys.argv[1])
		ds = [int(sys.argv[2])]
		vols = [float(sys.argv[3])]
		maxp_unif = float(sys.argv[4])


	max_ks = GetMaxKs(vols, maxp_unif)


	d_to_vol = {}

	for d in ds:
		vol_to_alg = {}
		for vol in vols:

			rects = []
			for i in range(num_rects):
				rects.append(HyperRectangleTarget(d, vol))

			alg_to_successes = {}

			for sampler in samplers:

				expected_success = {}
				#import pdb; pdb.set_trace()
				for i in range(1,max_ks[vol]):
					X = samplers[sampler]['fn'](i, d)
					e_s = FracInRects(X, rects)
					expected_success[i] = e_s
					print "{} {} {} {} {}".format(d, vol, sampler, i, e_s)
				alg_to_successes[sampler] = expected_success
			vol_to_alg[vol] = alg_to_successes
		d_to_vol[d] = vol_to_alg


	if pickle_result:
		import pickle
		pickle.dump(d_to_vol, open('pickled_data/rects={}_d={}_vol={}_maxp={}'.format(num_rects, ds[0], vols[0], maxp_unif), 'wb'))
		exit()



	matplotlib.rcParams.update({'font.size':4})
	fig = plt.figure()
	counter = 0
	for d in ds:
		for vol in vols: 
			counter = counter + 1
			cur_ax = fig.add_subplot(len(ds),len(vols),counter)#, adjustable='box', aspect=100)
			cur_data = d_to_vol[d][vol]
			print("{} {}".format(d, vol))
			plot_stuff(cur_ax, cur_data, num_rects, d, max_ks[vol], vol, samplers)

	plt.tight_layout()
	plt.savefig('dpp_nrects={}_{}dims_maxp={}_{}vols.pdf'.format(num_rects, len(ds), maxp_unif, len(vols)))



	exit()









# compute L1, L2 dist from origin and center
def origin_center_data(samplers, eval_measures, ns, ds, sample_num):
        n_max = ns[-1]


	sampler_to_n_err = {}
	for sampler in samplers:
		n_to_d_err = {}
		for n in ns:
			d_to_measure_err = {}
			for d in ds:
				measure_to_err = {}
				X = samplers[sampler]['fn'](n, d)
				
				for measure in eval_measures:
					measure_to_err[measure] = eval_measures[measure](X)
				d_to_measure_err[d] = measure_to_err
			n_to_d_err[n] = d_to_measure_err
			print("finishd n={}, sampler={}".format(n, sampler))
		sampler_to_n_err[sampler] = n_to_d_err


	#import pdb; pdb.set_trace()

	import pickle
	pickle_loc = 'pickled_data/origin_center_data/nmax={}_nsamplers={}_neval={}_d={}_samplenum={}'.format(n_max,
			  len(samplers), len(eval_measures), ''.join(str(e)+',' for e in ds)[:-1], sample_num)
	
	pickle.dump(sampler_to_n_err, open(pickle_loc, 'wb'))
	exit()
	
					


def draw_samples(samplers, ns, ds, sample_num):
        import pickle
        for sampler in samplers:
                print sampler
                for n in ns:
                        for d in ds:
                                X = samplers[sampler]['fn'](n,d)
                                #print X
                                pickle_loc = 'pickled_data/all_samples/sampler={}_n={}_d={}_samplenum={}'.format(sampler,n,d,sample_num)

        
                                pickle.dump(X, open(pickle_loc, 'wb'))


samplers = {#'SobolSampler':{'fn': SobolSampler,'color': 'g'},
	    #'RecurrenceSampler': {'fn': RecurrenceSampler,'color': 'r'},
	    #'SobolSamplerNoNoise': {'fn': SobolSamplerNoNoise,'color': 'b'},
	    #'DPPnsquared': {'fn': dpp_rbf_unitcube.DPPSampler, 'color': 'k'},
	    #'UniformSampler': {'fn': numpy.random.rand, 'color': 'c'},
            'DPPNarrow': {'fn': dpp_rbf_unitcube.DPPNarrow, 'color': 'm'},
            #'DPPVNarrow': {'fn': dpp_rbf_unitcube.DPPVNarrow, 'color': 'm'}
            #'DPPVVNarrow': {'fn': dpp_rbf_unitcube.DPPVVNarrow, 'color': 'm'},
            #'DPPVVVNarrow': {'fn': dpp_rbf_unitcube.DPPVVVNarrow, 'color': 'm'},
            #'DPPNNarrow': {'fn': dpp_rbf_unitcube.DPPNNarrow, 'color': 'm'},
            #'DPPNNNarrow': {'fn': dpp_rbf_unitcube.DPPNNNarrow, 'color': 'm'}
            #'DPPNsquaredNarrow': {'fn': dpp_rbf_unitcube.DPPNsquaredNarrow, 'color': 'm'}
            #'DPPClipped': {'fn': dpp_rbf_unitcube.DPPClippedSampler, 'color': 'm'}
        #'DPPVVNarrow': {'fn': dpp_rbf_unitcube.DPPVVNarrow, 'color': 'm'},
    }

#eval_measures = {'l2':get_min_l2_norm, 
		 #'l1':get_min_l1_norm, 
#		 'l2_cntr':get_min_l2_norm_center, 
		 #'l1_cntr':get_min_l1_norm_center,
#                 'discrep':get_discrepency}

#random_rect(samplers)



numpy.random.seed()
n_max = 150
ns = [int(numpy.exp(x)) for x in numpy.linspace(0, numpy.log(n_max), 20)]
ns = sorted(list(set(ns)))
#ns = [115]
ds = [2]#,2,3,4,5]


draw_samples(samplers, ns, ds, sys.argv[1])
#origin_center_data(samplers, eval_measures, n_max, ds, sys.argv[1])






exit()





def simulator(eval, sampler, d, n_max, num_samples_per_index=10):
	ns = [int(numpy.exp(x)) for x in numpy.linspace(0, numpy.log(n_max), 20)]
	ns = sorted(list(set(ns)))
	s_hist = [[] for _ in range(len(ns))]
	for it in range(num_samples_per_index*len(ns)):
		idx = (it % len(ns))
		n = ns[idx]
		X = sampler(n, d)

		val = eval(X)
		s_hist[idx].append(val)

		if it % num_samples_per_index == 0:
			print '%d/%d' % (it, num_samples_per_index*len(ns))

	vs = [numpy.mean(tmp) for tmp in s_hist]
	err_us = [sorted(s_hist[i])[int(.75*len(s_hist[i]))] for i in range(len(s_hist))]
	err_ls = [sorted(s_hist[i])[int(.25*len(s_hist[i]))] for i in range(len(s_hist))]
	return ns, vs, err_ls, err_us

NUM_SAMPLES_PER_INDEX = 300
samplers = [{'name': 'RandomSampler',
						 'fn': numpy.random.rand,
						 'color': 'b'},
						 {'name': 'SobolSampler',
						 'fn': SobolSampler,
						 'color': 'g'},
						 {'name': 'RecurrenceSampler',
						 'fn': RecurrenceSampler,
						 'color': 'r'}]

simulators = [{'name': '$\min_{i=1,\dots,n} ||x_i||_2^2$',
				 'filename_prefix': 'l2_sq_d',
				 'fn': get_min_l2_norm_squared},
				 {'name': 'Star Discrepency',
				 'filename_prefix': 'discrepency_d',
				 'fn': get_discrepency},
				{'name': '$\min_{i=1,\dots,n} ||x_i||_1$',
				 'filename_prefix': 'l1_d',
				 'fn': get_min_l1_norm}]

n_max = 1000
for sim in simulators:
	for d in range(1,7):
		for sampler in samplers:
			print 'Starting %s with %s with d=%d' %(sim['name'], sampler['name'], d)
			# ns, vs, err_ls, err_us = discrepency_simulator(sampler['fn'], d, n_max, num_samples_per_index=NUM_SAMPLES_PER_INDEX)

			ns, vs, err_ls, err_us = simulator(sim['fn'], sampler['fn'], d, n_max, num_samples_per_index=NUM_SAMPLES_PER_INDEX)

			print ns, vs
			plt.plot(ns, vs, '.', color=sampler['color'], label=sampler['name'])
			plt.fill_between(ns, err_ls, err_us, alpha=.1, color=sampler['color'])

		plt.xscale('log')
		plt.yscale('log')
		plt.legend()
		plt.title('Dimension d=%d' % d)
		plt.xlabel('Number of points')
		plt.ylabel(sim['name'])
		ax = plt.gca()
		ax.grid(True,which="both")
                save_name = sim['filename_prefix']+'%d' % d
                print("save_name:", save_name)
		plt.savefig(save_name)
		plt.close()



# experiments: 
# as n varies
# as d varies
# as d varies, where some fraction of dimensions are noise (e.g. we can ignore them)
# using distance to randomly sampled points
