import numpy
import pickle, glob
from current_experiment import *


def get_fname(sampler, n, d, sample_num):
    return 'pickled_data/all_samples/sampler={}_n={}_d={}_samplenum={}'.format(sampler,n,d,sample_num)
    



def get_out_name(sampler, eval_measure):
        return 'pickled_data/all_samples_errors/sampler={}_eval={}'.format(sampler, eval_measure)

def get_in_name(sampler, n, d):
        if 'SeqPostSigma' in sampler:
                return './pickled_data/dim={}/sampler={}_n=*_d={}_samplenum=*'.format(d, sampler, d)
        else:
                return './pickled_data/dim={}/sampler={}_n={}_d={}_samplenum=*'.format(d, sampler, n,d)


def find_actual_samples(ns, sampler, n_or_d):
        sampled_ns = set()

        if 'SeqPostSigma' in sampler:
                loc = './pickled_data/dim=1/sampler={}_*'.format(sampler)
        else:
                loc = './pickled_data/all_samples/sampler={}_*'.format(sampler)

        for f_name in glob.glob(loc):

                n_and_stuff = f_name.split(n_or_d + '=')[-1]
                n = int(n_and_stuff.split('_')[0])
                
                if n in ns and n not in sampled_ns:
                        sampled_ns.add(n)


        return sampled_ns

        
        

def compute_individual_errors(samplers, eval_measures, ns_try, ds_try):
        for sampler in samplers:
                #ns = find_actual_samples(ns_try, sampler, 'n')
                #ds = find_actual_samples(ds_try, sampler, 'd')
                ns = ns_try
                ds = ds_try


                for eval_measure in eval_measures:
                        # if saved evaluations exist
                        try:
                                with open(get_out_name(sampler, eval_measure), 'rb') as pickle_file:
                                        cur_evals = pickle.load(pickle_file)
                                        pickle_file.close()
                        except:
                                cur_evals = {}
                        #import pdb; pdb.set_trace()
 
                        if 'SeqPost' in sampler:
                                d = 1
                                in_file_names = glob.glob(get_in_name(sampler, 1, d))
                                #print len(in_file_names)
                                counter = 0
                                for in_file_name in in_file_names:
                                        sample_num = in_file_name.split("=")[-1]
                                        counter += 1

                                        pkl_file = open(in_file_name)
                                        cur_sample = pickle.load(pkl_file)

                                        print "sampler={}, eval={}, ns={}, d={}, samples={}, samples seen so far={}".format(sampler, eval_measure, ns, d, len(in_file_names), counter)  
                                        for n in ns:
                                                if d not in cur_evals:
                                                        cur_evals[d] = {}
                                                if n not in cur_evals[d]:
                                                        cur_evals[d][n] = {}
                                                #if eval_measure == 'discrep' and n == 749:


                                        

                                                if sample_num not in cur_evals[d][n]:
                                                        # compute eval
                                                        #print("sample_num: {}".format(sample_num))
                                                        #print("n: {}".format(n))
                                                        cur_sample_n = cur_sample[0:n]
                                                        #print("len of cur_sample_n: {}".format(len(cur_sample_n)))
                                                        cur_evals[d][n][sample_num] = eval_measures[eval_measure](cur_sample_n)
                                                        #print(cur_evals[d][n][sample_num], eval_measure, sample_num)
                                                        #sys.exit()
                                                        
                        
                        else:
                                for d in ds:
                                        for n in ns:
                                        
                                                in_file_names = glob.glob(get_in_name(sampler, n,d))
                                                print "sampler={}, eval={}, n={}, d={}, samples={}".format(sampler, eval_measure, n, d, len(in_file_names))
                                                for in_file_name in in_file_names:
                                                

 
                                                        if d not in cur_evals:
                                                                cur_evals[d] = {}
                                                        if n not in cur_evals[d]:
                                                                cur_evals[d][n] = {}

                                                        
                                                        sample_num = in_file_name.split("=")[-1]
                                                        
                                                        if sample_num not in cur_evals[d][n]:
                                                                pkl_file = open(in_file_name)
                                                                #print in_file_name
                                                                cur_sample = pickle.load(pkl_file)
                                                                # compute eval
                                                                cur_evals[d][n][sample_num] = eval_measures[eval_measure](cur_sample)
                                                                #print n, d, cur_sample
                                                                #print(cur_evals[d][n][sample_num])
                                                                
                        with open(get_out_name(sampler, eval_measure), 'wb') as pickle_file:
                                pickle.dump(cur_evals, pickle_file)

                                                                                                
                        #print cur_evals.keys()


def compute_discrep_for_samples():
        samplers = get_samplers()
        eval_measures = get_eval_measures()
        ns = get_ns()
        ds = get_ds()        
        compute_individual_errors(samplers, eval_measures, ns, ds)
        #import cProfile
        #import re
        #cProfile.run('compute_individual_errors(samplers, eval_measures, ns, ds)', sort='cumtime')


if __name__ == "__main__":
        compute_discrep_for_samples()



