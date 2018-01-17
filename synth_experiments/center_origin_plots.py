import pickle, pprint
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from current_experiment import *

            
def get_sampler_names():
    sampler_names = {'SobolSampler': 'Sobol',
                     'SobolSamplerHighD': 'Sobol',
                'RecurrenceSampler': 'Add Recurrence',
                'SobolSamplerNoNoise': 'Sobol No Noise',
                'DPPnsquared': 'DPP-rbf-wide',
                'UniformSampler': 'Uniform',
                     'DPPNarrow': 'DPP-rbf-g=8',
                     'DPPVNarrow': 'DPP-rbf-g=20',
                     'DPPVVNarrow': 'DPP-rbf-g=50',
                     'DPPVVVNarrow': 'DPP-rbf-g=100',
                     'DPPNNarrow': 'DPP-rbf-g=n/2',
                     'DPPNNNarrow': 'DPP-rbf-g=n',
                     'DPPNsquaredNarrow': 'DPP-rbf-g=n*n',
                     'DPPSeqPostSigma001': 'DPP-rbf-sigma=0.001',
                     'DPPSeqPostSigma003': 'DPP-rbf-sigma=0.003'}
    return sampler_names

def get_measure_names():
    measure_names = {'l2':'distance from origin', 'l2_cntr':'distance from center', 'l1':'L1_from_origin', 'l1_cntr':'L1_from_center', 'discrep': 'star discrepancy'}
    return measure_names


def get_filename(ds, measures, samplers, min_sample_num):
    example_filename = 'ds={}_measures={}_samplers={}_nminandmax={}-{}_minnumsamples={}'
    fname = example_filename.format(''.join(str(e)+',' for e in ds)[:-1],
                                    ''.join(str(e)+',' for e in measures)[:-1], 
                                    ''.join(str(e)+',' for e in samplers)[:-1], 
                                    get_n_min(), get_n_max(), min_sample_num)
    return fname


def load_errors():
    data = {}
    for sampler in get_samplers():
        for eval_measure in get_eval_measures():
            try:

                pkl_file = open('pickled_data/all_samples_errors/sampler={}_eval={}'.format(sampler, eval_measure))
                if not sampler in data:
                    data[sampler] = {}
                data[sampler][eval_measure] = pickle.load(pkl_file)
            except:
                print("tried {}, {}, but doesn't exist".format(sampler, eval_measure))

    return data
            
        


def compute_averages(data):
    avgs = {}
    stds = {}
    #print data.keys()
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
                    #print data[sampler][measure].keys()
                    #print sampler, measure, n, d
                    #print sorted(data[sampler][measure][d].keys())

                    for sample_num in data[sampler][measure][d][n]:

                        # there's some data which was generated with n^3 iters of mcmc, but we want to exclude it.
                        #subsample_num = sample_num[len(sample_num)-1]
                        #bad_sample_nums = ['4','5','6']#['1','2','3']
                        #bad_sample = subsample_num in bad_sample_nums
                        #if "PNNarrow" in sampler and d == 1 and n == 40 and bad_sample:
                        #    #import pdb; pdb.set_trace()
                        #    continue

                        avgs[sampler][n][d][measure].append(data[sampler][measure][d][n][sample_num])

                        
    for sampler in get_samplers():
        for n in get_ns():
            for d in get_ds():
                for measure in get_eval_measures():
                    if measure == 'l2' or measure == 'l2_cntr':
                        avgs[sampler][n][d][measure] = np.array(avgs[sampler][n][d][measure])
                        avgs[sampler][n][d][measure] = avgs[sampler][n][d][measure]*avgs[sampler][n][d][measure]
                    err_us = sorted(avgs[sampler][n][d][measure])[int(.75*len(avgs[sampler][n][d][measure]))]
                    err_ls = sorted(avgs[sampler][n][d][measure])[int(.25*len(avgs[sampler][n][d][measure]))]
                    cur_std = np.std(avgs[sampler][n][d][measure])
                    stds[sampler][n][d][measure] = [cur_std, err_ls, err_us, len(avgs[sampler][n][d][measure])]

                    avgs[sampler][n][d][measure] = np.median(avgs[sampler][n][d][measure])                                                                  
                                                                  
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
    matplotlib.rcParams.update({'font.size':8})
    fig = plt.figure(figsize=(9,3))
    #fig.suptitle("Columns, left to right: Star discrepancy, squared distance from the origin, and squared distance from the center.\n" + 
    #             "K between 1 and 55. Shaded is 45th to 55th percentile.\n" +
    #             "DPPs are using an RBF kernel: DPP-rbf-narrow has variance 1/10, DPP-rbf-wide has variance d/2.", 
    #             fontsize=8)

    counter = 0
    measures = ['discrep','l2_cntr', 'l2']#, 'l1', 'l1_cntr']
    ds = get_ds()
    #ds = [get_ds()[0], get_ds()[1], get_ds()[2], get_ds()[3], get_ds()[6]]

    samplers = get_samplers()#['SobolSampler','UniformSampler', 'DPPVVNarrow']
    

    min_samples = []

    for d in ds:
        for measure in measures:
            counter = counter + 1
            cur_ax = fig.add_subplot(len(ds),len(measures),counter, adjustable='box')#adjustable='box', aspect='equal')#adjustable='box', aspect=1)#, adjustable='box', aspect=100)
            #cur_ax.set_aspect('equal', 'box')
            cur_avgs = get_one_plot_data(avgs, measure, d)
            cur_stds = get_one_plot_data(stds, measure, d)

            # to get the minimum samples used to make one of the plotted points
            #print cur_stds
            cur_min = [None, None, None, float('inf')]
            for cur_sampler in cur_stds:
                
                cur_cur_min = min([cur_stds[cur_sampler][i][3] for i in cur_stds[cur_sampler]])
                if cur_min[3] > cur_cur_min:
                    cur_min = [d, measure, cur_sampler, cur_cur_min]
            min_samples.append(cur_min)

            one_plot(cur_ax, cur_avgs, cur_stds, measure, d, samplers)
            #cur_ax.set_ylabel(get_measure_names()[measure])
            if d == ds[0] and measure == 'discrep':
                cur_ax.set_title('star discrepancy')
            elif d == ds[0] and measure == 'l2':
                cur_ax.set_title('distance from origin')
            elif d == ds[0] and measure == 'l2_cntr':
                cur_ax.set_title('distance from center')
            if measure == 'discrep' and d == ds[-1]:
                cur_ax.set_xlabel('k, between {} and {}'.format(get_n_min(), get_n_max()))
            if measure == 'discrep':
                cur_ax.set_ylabel('d={}'.format(d))



    plt.tight_layout()


    print("the min samples: ", min_samples)
    min_samp_num = min([x[3] for x in min_samples])
    
    #print("the number of d=1,n=40,DPPNNarrow samps:", 

    out_fname = 'plots/sobol_worse_than_unif/' + get_filename(ds, measures, samplers, min_samp_num) + '.pdf'
    plt.savefig(out_fname)
    print("saving to {}".format(out_fname))




# takes cur_avgs which is sampler -> n -> err
def one_plot(cur_ax, cur_avgs, cur_stds, measure, d, samplers):
    cur_samplers = {}
    for sampler in samplers:
        cur_samplers[sampler] = 0
    for sampler in cur_samplers:
        ns = sorted(cur_avgs[sampler].keys())
        ns_offset = [cur_samplers[sampler] + n for n in ns]
        errs = [cur_avgs[sampler][n] for n in ns]
        err_stds = [cur_stds[sampler][n] for n in ns]
        err_ls = [err_stds[i][1] for i in range(len(err_stds))]
        err_us = [err_stds[i][2] for i in range(len(err_stds))]
        #cur_ax.plot(ns, errs, color=get_samplers()[sampler])
        
        #line,_,_ = cur_ax.errorbar(ns_offset, errs, yerr=err_stds, color=get_samplers()[sampler], elinewidth=0.5, linewidth=0.5)
        
        #line.set_label(sampler)
        
        sampler_label = get_sampler_names()[sampler]
        
        
        #del errs[len(errs)-1]
        #del err_ls[len(err_ls)-1]
        #del err_us[len(err_us)-1]
        #del ns_offset[len(ns_offset)-1]

        cur_ax.plot(ns_offset, errs, '.', color=get_samplers()[sampler]['color'], label=sampler_label)
        cur_ax.fill_between(ns_offset, err_ls, err_us, alpha=.1, color=get_samplers()[sampler]['color'])
        cur_ax.set_xscale('log')
        cur_ax.set_yscale('log')
        cur_ax.grid(True, which="both")
        cur_ax.legend()
    

	






def print_averages(avgs, stds):
    print avgs['UniformSampler'][40][1]
    print avgs['DPPNNarrow'][40][1]
    print stds['DPPNNarrow'][40][1]
    #for thing in sorted():
    #    print thing
    for thing in avgs:
        print thing
    sys.exit()


def make_plots():

    data = load_errors()
    #data = get_data()
    
    compute_averages(data)

if __name__ == "__main__":
    make_plots()
 
