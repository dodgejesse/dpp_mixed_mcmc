import pickle, pprint
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_n_max():
    return 23
            
# returns odd numbers between 1 and n_max
def get_ns():
    return range(1,get_n_max()+1, 2)
    

def get_filename():
    example_filename = 'rbf_g=1overd_cliptype=k_k=odds_up_to_{}'
    fname = example_filename.format(get_n_max())
    return fname

def get_data():
    


    data = {}

    for sample_counter in range(101):
        for sample_subcounter in range(1,51):
            sample_num = '{}_{}'.format(sample_counter, sample_subcounter)
            fname = get_filename() + '_samplenum=' + sample_num
        
            try:
                pkl_file = open('pickled_data/dpp_samples_d=2/' + fname, 'rb')
                data[sample_num] = pickle.load(pkl_file)
            except:
                pass
    return data


                                                                  

def get_one_plot_data(data, n, n_bins):
    # need one vector for x, one vector for y
    first_dim = []
    second_dim = []
    for snum in data:

        first_dim += [data[snum][n][k][0] for k in range(len(data[snum][n]))]
        second_dim += [data[snum][n][k][1] for k in range(len(data[snum][n]))]

    heatmap, xedge, yedge = np.histogram2d(first_dim, second_dim, bins=(n_bins,n_bins))
    
                   
    return heatmap, [0,1,0,1]
    


def multiplot_by_n(data, ns, bins):
    matplotlib.rcParams.update({'font.size':7})
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Heatmap of 5000 samples from sobol sequence with uniform noise.\n{}x{} bins, plotted with bicubic interpoliation.".format(bins,bins), fontsize=8)
    #fig.suptitle("Heatmap of 4900 samples drawn from a DPP with kernel = exp(-g||x-y||^2), with g=n/d, d=2.\n{}x{} bins, plotted with bicubic interpoliation.".format(bins,bins), fontsize=8)

    num_plots_x = 3
    num_plots_y = 3
    all_plot_data = {}

    max_value = 0
    min_value = float('inf')
    for i in range(num_plots_x*num_plots_y):
        n = ns[i]
        heatmap, edges = get_one_plot_data(data, n, bins)
        heatmap = heatmap/(1.0*n)
        all_plot_data[i] = heatmap
        if max_value < np.amax(heatmap):
            max_value = np.amax(heatmap)
        if min_value > np.amin(heatmap):
            min_value = np.amin(heatmap)
        print min_value, max_value
            
    for i in range(num_plots_x*num_plots_y):
        n = ns[i]
        cur_ax = fig.add_subplot(num_plots_y, num_plots_x, i+1)
        

        one_plot(cur_ax, all_plot_data[i], edges, n, min_value, max_value)


    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    out_fname = 'heatmap_' + get_filename() + '_nsamples={}.pdf'.format(len(data))
    plt.savefig('plots/' + out_fname)
    print("saving to {}".format('plots/' + out_fname))



def one_plot(cur_ax, heatmap, extent, n, min_value, max_value):
    cur_ax.imshow(heatmap, extent=extent, interpolation='bicubic', cmap='BuPu')#,vmin=min_value,vmax=max_value)


    cur_ax.set_title('n={}'.format(n))




data = get_data()
if len(data) == 0:
    print 'did not read data, probably filename incorrect'
    exit
ns = get_ns()
bins = 20
multiplot_by_n(data, ns, bins)
