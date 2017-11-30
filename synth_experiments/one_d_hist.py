import pickle, pprint
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

            
def get_sampler():
    sampler = 'UniformSampler'
                #'DPPApproxVVNarrow'
                #'DPPVVNarrow'
                #'SobolSampler': 'g',
                #'RecurrenceSampler': 'k',
                #'SobolSamplerNoNoise': 'c',
                #'DPPnsquared': 'r',
                #'UniformSampler': 'b',
                #'DPPNarrow': 'm',
                #'DPPVNarrow': 'm',
                #'DPPVVNarrow': 'k',
                #'DPPVVVNarrow': 'c',
                #'DPPNNarrow': 'r',
                #'DPPNsquaredNarrow': 'm'}
                #'DPPNNNarrow': 'c'}
    return sampler

def get_sampler_names():
    sampler_names = {'SobolSampler': 'Sobol',
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
                     'DPPNsquaredNarrow': 'DPP-rbf-g=n*n'}
    return sampler_names

def get_n():
    return 115


def get_out_filename(sampler, n):
    example_filename = 'plot_sample_d=1_sampler={}_n={}'
    fname = example_filename.format(sampler, n)
    return fname

def get_in_filenames(sampler, n, d):
    for sample_counter in range(50):
        print(sample_counter)
        for sample_subcounter in range(1,4):
            sample_num = '{}_{}'.format(sample_counter, sample_subcounter)
            file_name = get_fname(sampler, n, d, sample_num)

    return 'pickled_data/all_samples/sampler={}_n={}_d={}_samplenum={}'.format(sampler,n,d,sample_num)


def load_samples(sampler, n):
    data = {}
    for f_name in glob.glob('./pickled_data/all_samples/sampler={}_n={}_d=1_*'.format(sampler,n)):
        sample_num = f_name.split("=")[-1]
        pkl_file = open(f_name)
        data[sample_num] = pickle.load(pkl_file)
        
    return data


def multiplot_samples(data, sampler, n):
    matplotlib.rcParams.update({'font.size':6})
    fig = plt.figure()#figsize=(10,10))


    num_rows = 5
    num_cols = 2
    counter = 0
    for sample_num in data:
        counter += 1
        if counter > num_rows * num_cols:
            break

        cur_ax = fig.add_subplot(num_rows,num_cols,counter)
        one_plot_lines(cur_ax, data[sample_num], sampler, n)
    


    #plt.tight_layout()



    out_fname = 'plots/samples_plotted/' + get_out_filename(sampler, n) + '.pdf'
    plt.savefig(out_fname)
    print("saving to {}".format(out_fname))




# takes cur_avgs which is sampler -> n -> err
def one_plot(cur_ax, cur_data, sampler, n):
    ys = range(len(cur_data))
    
    cur_ax.scatter(sorted(cur_data), ys, marker='x')

def one_plot_lines(cur_ax, cur_data, sampler, n):

    cur_data_formatted = [i[0] for i in cur_data]
    cur_ax.hlines(1,0,1)  # Draw a horizontal line
    cur_ax.eventplot(cur_data_formatted, orientation='horizontal', colors='b')
    cur_ax.axis('off')


sampler = get_sampler()
n = get_n()


data = load_samples(sampler, n)
if len(data) == 0:
    'sorry, no samples found for sampler={}, n={}'.format(sampler, n)
    exit()

multiplot_samples(data, sampler, n)


 
