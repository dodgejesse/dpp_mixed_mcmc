import pickle, pprint
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


            
def get_samplers():
    samplers = {'SobolSampler': 'g',
                'RecurrenceSampler': 'r',
                'SobolSamplerNoNoise': 'b',
                'DPPnsquared': 'k',
                'UniformSampler': 'c'}
    return samplers

def get_n_max():
    return 55

def get_ns():
    n_max = get_n_max()
    ns = [int(np.exp(x)) for x in np.linspace(0, np.log(n_max), 20)]
    ns = sorted(list(set(ns)))
    return ns
    
def get_ds():
    ds = [2,3,5,10,15,25,35]
    return ds

def get_eval_measures():
    eval_measures = ['l2', 'l1', 'l2_cntr', 'l1_cntr']
    return eval_measures


def get_filename():
    example_filename = 'nmax={}_nsamplers={}_neval={}_d={}'
    fname = example_filename.format(get_n_max(), len(get_samplers()), len(get_eval_measures()), ''.join(str(e)+',' for e in get_ds())[:-1])
    return fname

def get_data():
    


    data = {}

    for sample_num in range(301):
        fname = get_filename() + '_samplenum={}'.format(sample_num)
        try:
            pkl_file = open('pickled_data/origin_center_data/' + fname, 'rb')
            data[sample_num] = pickle.load(pkl_file)
        except:
            pass
    return data


def compute_averages(data):
    avgs = {}
    stds = {}
    print data.keys()
    for s_num in data:
        for sampler in get_samplers():
            if sampler not in avgs:
                avgs[sampler] = {}
                stds[sampler] = {}
            for n in get_ns():
                if n not in avgs[sampler]:
                    avgs[sampler][n] = {}
                    stds[sampler][n] = {}
                for d in get_ds():
                    if d not in avgs[sampler][n]:
                        avgs[sampler][n][d] = {}
                        stds[sampler][n][d] = {}
                    for measure in get_eval_measures():
                        if measure not in avgs[sampler][n][d]:
                            avgs[sampler][n][d][measure] = []
                        avgs[sampler][n][d][measure].append(data[s_num][sampler][n][d][measure])
    
    for sampler in get_samplers():
        for n in get_ns():
            for d in get_ds():
                for measure in get_eval_measures():
                    cur_std = np.std(avgs[sampler][n][d][measure])
                    stds[sampler][n][d][measure] = cur_std
                    avgs[sampler][n][d][measure] = np.average(avgs[sampler][n][d][measure])
    
    #print_averages(avgs, stds)
    multiplot_measure_by_d(avgs, stds, len(data))

def get_one_plot_data(data, measure, d):
    sampler_to_n = {}
    for sampler in get_samplers():
        n_to_err = {}
        for n in get_ns():
            n_to_err[n] = data[sampler][n][d][measure]
        sampler_to_n[sampler] = n_to_err
    return sampler_to_n
    


def multiplot_measure_by_d(avgs, stds, num_samples):
    matplotlib.rcParams.update({'font.size':4})
    fig = plt.figure()
    counter = 0
    measures = ['l2', 'l2_cntr']
    ds = [get_ds()[0], get_ds()[1], get_ds()[2]]
    for d in ds:
        for measure in measures:
            counter = counter + 1
            cur_ax = fig.add_subplot(len(ds),len(measures),counter)#, adjustable='box', aspect=100)
            cur_avgs = get_one_plot_data(avgs, measure, d)
            cur_stds = get_one_plot_data(stds, measure, d)

            one_plot(cur_ax, cur_avgs, cur_stds, measure, d)

    plt.tight_layout()
    out_fname = get_filename() + '_nsamples={}.pdf'.format(num_samples)
    plt.savefig('plots/' + out_fname)


# takes cur_avgs which is sampler -> n -> err
def one_plot(cur_ax, cur_avgs, cur_stds, measure, d):
    
    cur_samplers = {'SobolSampler':-1.0/4, 'DPPnsquared':0, 'UniformSampler':1.0/4}
    
    for sampler in cur_samplers:
        ns = sorted(cur_avgs[sampler].keys())
        ns_offset = [cur_samplers[sampler] + n for n in ns]
        errs = [cur_avgs[sampler][n] for n in ns]
        err_stds = [cur_stds[sampler][n] for n in ns]
        #cur_ax.plot(ns, errs, color=get_samplers()[sampler])
        #cur_ax.set_xscale('log')
        #cur_ax.set_yscale('log')        
        line,_,_ = cur_ax.errorbar(ns_offset, errs, yerr=err_stds, color=get_samplers()[sampler], elinewidth=0.5)
        line.set_label(sampler)
        cur_ax.legend()
        
        
	

    cur_ax.set_title('measure={} d={}'.format(measure, d))



def print_averages(avgs, stds):
    for thing in sorted(avgs['DPPnsquared'][22]):
        print thing, avgs['DPPnsquared'][22][thing]
    for thing in sorted(avgs['UniformSampler'][22]):
        print thing, avgs['UniformSampler'][22][thing]

data = get_data()
compute_averages(data)
 
