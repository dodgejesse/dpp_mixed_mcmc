import numpy as np
import itertools

# {ln(2k - 3 / ln(2)}
def one_val(j):
    if j == 1:
        return 1
    else:
        cur = np.log(2.0 * j - 3) / np.log(2)
        frac_part = cur - np.floor(cur)
        return frac_part

def get_sequence(n,d):
    xs = np.array([]).reshape(0,d)
    
    
    one_d_seq = np.array([])
    add_more_points = True
    i = 0
    import itertools

    while add_more_points:
        i = i + 1
        new_one_d_val = one_val(i)

        smaller_set = set(list(itertools.product(one_d_seq,repeat=d)))
        one_d_seq = np.append(one_d_seq, new_one_d_val)
        larger_set =  set(list(itertools.product(one_d_seq,repeat=d)))
        new_vals = larger_set - smaller_set

        # find random permutation of the new_vals
        new_vals = np.random.permutation(list(new_vals))
        #import pdb; pdb.set_trace()
        
        for new_val in new_vals:
            xs = np.concatenate((xs, [new_val]), axis=0)
            
            #xs = np.append(xs, [np.array(new_val)])
            if len(xs) == n:
                add_more_points = False
                break

    return xs


def main():
    print(get_sequence(10, 2))

if __name__ == "__main__":
    main()
