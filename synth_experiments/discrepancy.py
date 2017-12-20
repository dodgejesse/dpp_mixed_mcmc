import sys
import os
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

# adds uniform noise in [0,1], then for those results above 1 it subtracts 1
def SobolSampler(n, d):
	X = sobol.i4_sobol_generate(d, n)
        print X
        return
        sys.exit()
	U = numpy.random.rand(d)
	X += U
	X -= numpy.floor(X)
	return X

# this method is to try and get sobol in D > 40
# adds uniform noise in [0,1], then for those results above 1 it subtracts 1
def SobolSamplerHighD(n, d):
        assert d in [5, 50, 100, 500]
        assert n < 1025
        
        with open('/home/ec2-user/software/Sobol.jl/test/results/exp_results_{}'.format(d), 'r') as f:
                lines = f.readlines()
        in_data = []
        counter = 0
        for line in lines:
                counter += 1
                if counter == 1 or counter == 2:
                        continue


                in_data.append(line.split(" ")[:-1])
                if len(in_data) == n:
                        break
        X = numpy.asarray(in_data, dtype=float)
        

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








	
					


def draw_samples(samplers, ns, ds, sample_num):
        import pickle
        for sampler in samplers:
                print sampler
                for n in ns:
                        for d in ds:
                                X = samplers[sampler]['fn'](n,d)
                                #print X
                                dir_path = 'pickled_data/dim={}/'.format(d)
                                if not os.path.isdir(dir_path):
                                        os.makedirs(dir_path)
                                pickle_loc = dir_path + 'sampler={}_n={}_d={}_samplenum={}'.format(sampler,n,d,sample_num)

        
                                pickle.dump(X, open(pickle_loc, 'wb'))


samplers = {#'SobolSampler':{'fn': SobolSampler,'color': 'g'},
	    #'RecurrenceSampler': {'fn': RecurrenceSampler,'color': 'r'},
	    #'SobolSamplerNoNoise': {'fn': SobolSamplerNoNoise,'color': 'b'},
	    #'DPPnsquared': {'fn': dpp_rbf_unitcube.DPPSampler, 'color': 'k'},
	    'UniformSampler': {'fn': numpy.random.rand, 'color': 'c'},
            #'DPPNarrow': {'fn': dpp_rbf_unitcube.DPPNarrow, 'color': 'm'},
            #'DPPVNarrow': {'fn': dpp_rbf_unitcube.DPPVNarrow, 'color': 'm'}
            #'DPPVVNarrow': {'fn': dpp_rbf_unitcube.DPPVVNarrow, 'color': 'm'},
            #'DPPVVVNarrow': {'fn': dpp_rbf_unitcube.DPPVVVNarrow, 'color': 'm'},
            #'DPPNNarrow': {'fn': dpp_rbf_unitcube.DPPNNarrow, 'color': 'm'},
            #'DPPNNNarrow': {'fn': dpp_rbf_unitcube.DPPNNNarrow, 'color': 'm'}
            #'DPPNsquaredNarrow': {'fn': dpp_rbf_unitcube.DPPNsquaredNarrow, 'color': 'm'}
            #'DPPClipped': {'fn': dpp_rbf_unitcube.DPPClippedSampler, 'color': 'm'}
            'SobolSamplerHighD':{'fn': SobolSamplerHighD, 'color':'m'},
            #'DPPVVNarrow': {'fn': dpp_rbf_unitcube.DPPVVNarrow, 'color': 'm'},
    }


#print SobolSamplerHighD(750, 500)[0]
#sys.exit()
#for i in range(100):
#        #SobolSampler(7,5)
#        print SobolSamplerHighD(7, 5)[0]
#
#        sys.exit()


numpy.random.seed()
n_max = 750
ns = [int(numpy.exp(x)) for x in numpy.linspace(0, numpy.log(n_max), 20)]
ns = sorted(list(set(ns)))
#ns = [750]
#ns = [750]
ds = [100,500]#,2,3,4,5]


draw_samples(samplers, ns, ds, sys.argv[1])
#origin_center_data(samplers, eval_measures, n_max, ds, sys.argv[1])








# experiments: 
# as n varies
# as d varies
# as d varies, where some fraction of dimensions are noise (e.g. we can ignore them)
# using distance to randomly sampled points
