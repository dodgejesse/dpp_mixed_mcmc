import sys
import os
import numpy
#import matplotlib
#import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('text', usetex=True)
import sobol_seq as sobol
import dpp_rbf_unitcube
from current_experiment import *




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
                for d in ds:
                        
                        for n in ns:
                                dir_path = 'pickled_data/dim={}/'.format(d)
                                pickle_loc = dir_path + 'sampler={}_n={}_d={}_samplenum={}'.format(sampler,n,d,sample_num)
                                if os.path.exists(pickle_loc):
                                        continue
                                X = samplers[sampler]['fn'](n,d)
                                #print X

                                if not os.path.isdir(dir_path):
                                        os.makedirs(dir_path)
                                pickle_loc = dir_path + 'sampler={}_n={}_d={}_samplenum={}'.format(sampler,n,d,sample_num)

        
                                pickle.dump(X, open(pickle_loc, 'wb'))





def draw_many_samples():
        numpy.random.seed()
        samplers = get_samplers()
        ns = get_ns()
        ds = get_ds()
        num_samples = get_num_samples()
        
        for i in range(num_samples):
                draw_samples(samplers, ns, ds, "{}_1".format(i))
                print "finished {}".format(i)



if __name__ == "__main__":
        
        
        
        draw_many_samples()
        
        
        #origin_center_data(samplers, eval_measures, n_max, ds, sys.argv[1])
        







# experiments: 
# as n varies
# as d varies
# as d varies, where some fraction of dimensions are noise (e.g. we can ignore them)
# using distance to randomly sampled points
