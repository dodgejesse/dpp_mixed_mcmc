import pickle, pprint


samplers = {'SobolSampler': 'g',
	    'RecurrenceSampler': 'r',
	    'SobolSamplerNoNoise': 'b',
	    'DPPnsquared': 'k',
	    'UniformSampler': 'c'}

eval_measures = ['l2', 'l1', 'l2_cntr', 'l1_cntr']

n_max = 25
ds = [1,5,10,15]
example_filename = 'nmax={}_nsamplers={}_neval={}_d={}_samplenum={}'

fname = example_filename.format(n_max, len(samplers), len(eval_measures), ''.join(str(e)+',' for e in ds)[:-1], 1)

pkl_file = open('pickled_data/origin_center_data/' + fname, 'rb')
data1 = pickle.load(pkl_file)
pprint.pprint(data1)
