import pickle, pprint
import numpy as np

samplers = {'SobolSampler': 'g',
	    'RecurrenceSampler': 'r',
	    'SobolSamplerNoNoise': 'b',
	    'DPPnsquared': 'k',
	    'UniformSampler': 'c'}

eval_measures = ['l2', 'l1', 'l2_cntr', 'l1_cntr']

n_max = 40
ns = [int(np.exp(x)) for x in np.linspace(0, np.log(n_max), 20)]
ns = sorted(list(set(ns)))
ds = [2,5,10,15,25,35]
example_filename = 'nmax={}_nsamplers={}_neval={}_d={}_samplenum={}'


data = {}

for s_num_1 in range(1,6):
    for s_num_2 in range(1,6):
        sample_num = str(s_num_1) + '_' + str(s_num_2)
        fname = example_filename.format(n_max, len(samplers), len(eval_measures), ''.join(str(e)+',' for e in ds)[:-1], sample_num)
        try:
            pkl_file = open('pickled_data/origin_center_data/' + fname, 'rb')
            data[sample_num] = pickle.load(pkl_file)
        except:
            pass
            




def compute_averages(data):
    avgs = {}
    stds = {}
    for s_num in data:
        for sampler in samplers:
            if sampler not in avgs:
                avgs[sampler] = {}
                stds[sampler] = {}
            for n in ns:
                if n not in avgs[sampler]:
                    avgs[sampler][n] = {}
                    stds[sampler][n] = {}
                for d in ds:
                    if d not in avgs[sampler][n]:
                        avgs[sampler][n][d] = {}
                        stds[sampler][n][d] = {}
                    for measure in eval_measures:
                        if measure not in avgs[sampler][n][d]:
                            avgs[sampler][n][d][measure] = []
                        avgs[sampler][n][d][measure].append(data[s_num][sampler][n][d][measure])
    
    for sampler in samplers:
        for n in ns:
            for d in ds:
                for measure in eval_measures:
                    cur_std = np.std(avgs[sampler][n][d][measure])
                    #print "stds:"
                    #print stds
                    #print "stds[sampler]:"
                    #print stds[sampler]
                
                    #print("{}, {}".format(n,d))

                    
                    stds[sampler][n][d][measure] = cur_std
                    avgs[sampler][n][d][measure] = np.average(avgs[sampler][n][d][measure])
    print_averages(avgs, stds)
    


def print_averages(avgs, stds):
    for thing in sorted(avgs['DPPnsquared'][22]):
        print thing, avgs['DPPnsquared'][22][thing]
    for thing in sorted(avgs['UniformSampler'][22]):
        print thing, avgs['UniformSampler'][22][thing]

compute_averages(data)
 
