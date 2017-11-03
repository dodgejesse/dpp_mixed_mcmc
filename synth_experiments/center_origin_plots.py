import pickle, pprint
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


            
def get_samplers():
    samplers = {'SobolSampler': 'g',
                #'RecurrenceSampler': 'k',
                #'SobolSamplerNoNoise': 'c',
                #'DPPnsquared': 'r',
                'UniformSampler': 'b',
                'DPPNarrow': 'm',
                #'DPPVNarrow': 'm',
                'DPPVVNarrow': 'k'}
                #'DPPVVVNarrow': 'c',
                #'DPPNNarrow': 'r'}
    return samplers

def get_sampler_names():
    sampler_names = {'SobolSampler': 'Sobol',
                'RecurrenceSampler': 'Add Recurrence',
                'SobolSamplerNoNoise': 'Sobol No Noise',
                'DPPnsquared': 'DPP-rbf-wide',
                'UniformSampler': 'Uniform',
                     'DPPNarrow': 'DPP-rbf-g=8',
                     'DPPVNarrow': 'DPP-rbf-g=20',
                     'DPPVVNarrow': '$k$-DPP-rbf',
                     'DPPVVVNarrow': 'DPP-rbf-g=100',
                     'DPPNNarrow': 'DPP-rbf-g=n'}
    return sampler_names

def get_n_max():
    return 150

def get_ns():
    n_max = get_n_max()
    ns = [int(np.exp(x)) for x in np.linspace(0, np.log(n_max), 20)]
    ns = sorted(list(set(ns)))
    return ns
    
def get_ds():
    ds = [2,3,4,5]#[2,3,5,7]#[2,3,5,10,15,25,35]
    return ds

def get_eval_measures():
    eval_measures = ['l2', 'l2_cntr', 'discrep']#['l2', 'l1', 'l2_cntr', 'l1_cntr', 'discrep']
    return eval_measures

def get_measure_names():
    measure_names = {'l2':'distance from origin', 'l2_cntr':'distance from center', 'l1':'L1_from_origin', 'l1_cntr':'L1_from_center', 'discrep': 'star discrepancy'}
    return measure_names


def get_filename():
    example_filename = 'nmax={}_nsamplers={}_neval={}_d={}'
    fname = example_filename.format(get_n_max(), len(get_samplers()), len(get_eval_measures()), ''.join(str(e)+',' for e in get_ds())[:-1])
    return fname

# WARNING THIS FUNCTION IS DEPRECATED!
def get_data():
    print("THIS IS DEPRECATED! DON'T USE!")
    sys.exit()
    data = {}

    for sample_counter in range(301):
        for sample_subcounter in range(1,11):
            sample_num = '{}_{}'.format(sample_counter, sample_subcounter)
            fname = get_filename() + '_samplenum=' + sample_num
            try:
                #pkl_file = open('pickled_data/origin_center_data/' + fname, 'rb')
                pkl_file = open('pickled_data/errors_from_samples/' + fname, 'rb')

                data[sample_num] = pickle.load(pkl_file)
            except:
                pass
    return data

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
    print data.keys()
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
                    for sample_num in data[sampler][measure][d][n]:
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
                    stds[sampler][n][d][measure] = [cur_std, err_ls, err_us]

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
    matplotlib.rcParams.update({'font.size':11})
    fig = plt.figure(figsize=(10,10))
    #fig.suptitle("Columns, left to right: Star discrepancy, squared distance from the origin, and squared distance from the center.\n" + 
    #             "K between 1 and 55. Shaded is 45th to 55th percentile.\n" +
    #             "DPPs are using an RBF kernel: DPP-rbf-narrow has variance 1/10, DPP-rbf-wide has variance d/2.", 
    #             fontsize=8)

    counter = 0
    measures = ['discrep','l2_cntr', 'l2']#, 'l1', 'l1_cntr']
    ds = [2,3,4]#get_ds()
    #ds = [get_ds()[0], get_ds()[1], get_ds()[2], get_ds()[3], get_ds()[6]]
    
    
    
    #for d in ds:
    #    for measure in measures:
    #counter = counter + 1
    #        cur_ax = fig.add_subplot(len(ds),len(measures),counter)#, adjustable='box', aspect=100)


    for d in ds:
        for measure in measures:
            counter = counter + 1
            cur_ax = fig.add_subplot(len(ds),len(measures),counter)#, adjustable='box', aspect=100)

            cur_avgs = get_one_plot_data(avgs, measure, d)
            cur_stds = get_one_plot_data(stds, measure, d)

            one_plot(cur_ax, cur_avgs, cur_stds, measure, d)
            #cur_ax.set_ylabel(get_measure_names()[measure])
            if d == 4:
                cur_ax.set_xlabel('k, between 1 and 150')
            if d == 2 and measure == 'discrep':
                cur_ax.set_title('star discrepancy')
            elif d == 2 and measure == 'l2':
                cur_ax.set_title('distance from origin')
            elif d == 2 and measure == 'l2_cntr':
                cur_ax.set_title('distance from center')
            if measure == 'discrep':
                cur_ax.set_ylabel('d={}'.format(d))


    plt.tight_layout()
    out_fname = get_filename() + '_nsamples=something.pdf'
    plt.savefig('plots/' + out_fname)
    print("saving to plots/{}".format(out_fname))


# takes cur_avgs which is sampler -> n -> err
def one_plot(cur_ax, cur_avgs, cur_stds, measure, d):

    #cur_samplers = {'SobolSampler':-1.0/4, 'DPPnsquared':0, 'UniformSampler':1.0/4}
    #cur_samplers = {'SobolSampler':-1.0/4, 'DPPnsquared':-1.0/12, 'RecurrenceSampler':1.0/12, 'UniformSampler':1.0/4}

    #cur_samplers = {'SobolSampler':0, 'RecurrenceSampler':0, 'UniformSampler':0, 'DPPnsquared':0, 'DPPNarrow':0}
    #samplers = get_samplers()
    samplers = ['SobolSampler','UniformSampler', 'DPPVVNarrow']
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

        del errs[len(errs)-1]
        del err_ls[len(err_ls)-1]
        del err_us[len(err_us)-1]
        del ns_offset[len(ns_offset)-1]

        cur_ax.plot(ns_offset, errs, '.', color=get_samplers()[sampler], label=sampler_label)
        cur_ax.fill_between(ns_offset, err_ls, err_us, alpha=.1, color=get_samplers()[sampler])
        cur_ax.set_xscale('log')
        cur_ax.set_yscale('log')
        cur_ax.grid(True, which="both")
        cur_ax.legend()
    
        
	






def print_averages(avgs, stds):
    for thing in sorted(avgs['DPPnsquared'][22]):
        print thing, avgs['DPPnsquared'][22][thing]
    for thing in sorted(avgs['UniformSampler'][22]):
        print thing, avgs['UniformSampler'][22][thing]

data = load_errors()
#data = get_data()


compute_averages(data)
 
