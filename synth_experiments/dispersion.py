import scipy as sp
import scipy.spatial as spatial
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


# notes: 
# current implementation (reflecting points across all faces of the unit cube) works up to about d=6, n=50.
# without reflecting, can build voronoi diagram up to about d=10, n=50.
# if we need to get up to d=10, we can try doing something like:
# https://stackoverflow.com/questions/36063533/clipping-a-voronoi-diagram-python


def main():
    cur_sample = get_sample(d=2, n=29, snum='63_1')
    
    start_time = time.time()
    vor = spatial.Voronoi(cur_sample)
    print('took {} seconds'.format(time.time() - start_time))
    #print_vor_attribs(vor)

    #bounded_lines = get_bounded_lines(vor)
    #unbounded_lines = get_unbounded_lines(vor)
    #print_lines(bounded_lines, unbounded_lines)
    #print_vor(vor)


    vor = bounded_voronoi(cur_sample)
    #print vor.vertices
    #print_vor(vor)
    #compute_dispersion(vor)
    #unit_tests_bounded_vor(vor)
    #print_bounded_ddim_vor(vor)
    print_bounded_vor(vor)
    
    
def get_sample(d='4', sampler='UniformSampler', n='88', snum='3_1'):
    in_file_name = './pickled_data/dim={}/sampler={}_n={}_d={}_samplenum={}'.format(d, sampler, n,d, snum)
    #print('reading data from {}'.format(in_file_name))

    pkl_file = open(in_file_name)
    cur_sample = pickle.load(pkl_file)
    #print('data read.')
    return cur_sample


def get_bounded_lines(vor):
    line_segments = []
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            line_segments.append([(x, y) for x, y in vor.vertices[simplex]])

    return line_segments


def get_unbounded_lines(vor):
    line_segments = []

    center = vor.points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * vor.points.ptp(axis=0).max()

            line_segments.append([(vor.vertices[i, 0], vor.vertices[i, 1]),
                                  (far_point[0], far_point[1])])
    return line_segments
    

def print_vor(vor):
    spatial.voronoi_plot_2d(vor)
    #plt.ylim((-.5,1))
    #plt.xlim((-.5,1))

    #plt.savefig('/home/ec2-user/scratch/plots/voronoi_debug.pdf')
    plt.show()


def print_lines(bounded_lines, unbounded_lines):
    print('')
    print('bounded_lines')
    for line in bounded_lines:
        print(line)

    print('')
    print('unbounded_lines')
    print(unbounded_lines)
    

def print_vor_attribs(vor):

    print('')
    print('the points')
    print(vor.points)

    print vor
    print(vor.min_bound)
    print(vor.max_bound)
    #sys.exit()

    print('')
    print('the vertices')
    print(vor.vertices)

    print('')
    print('ridge points')
    print(vor.ridge_points)
    
    print('')
    print('ridge vertices')
    print(vor.ridge_vertices)

    print('print zip')
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        print pointidx
        print simplex
        print('')
    

def get_epsilon(d):
    return sys.float_info.epsilon * d * 1000


# this function taken from 
# https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells#
# and adjusted to work with dimensions other than 2
def bounded_voronoi(points):
    eps = get_epsilon(len(points[0]))

    # Mirror points
    all_points = np.copy(points)
    for d in range(len(points[0])):
        new_points_down = np.copy(points)        
        new_points_down[:, d] = -new_points_down[:, d]

        new_points_up = np.copy(points)
        new_points_up[:, d] = 1 + (1 - new_points_up[:, d])
        
        all_points = np.append(all_points, new_points_down, axis=0)
        all_points = np.append(all_points, new_points_up, axis=0)

    # Compute Voronoi
    #sys.stdout.write("starting Voronoi diagram computation... ")
    #vor_start_time = time.time()
    qhull_options = 'Qbb Qc Qz Qx Q12' if len(points[0]) > 4 else 'Qbb Qc Qz Q12'
    vor = sp.spatial.Voronoi(all_points, qhull_options=qhull_options)
    #vor = sp.spatial.Voronoi(all_points)
    #sys.stdout.write("done! took {} seconds.\n".format(time.time() - vor_start_time))


    #points_start_time = time.time()
    #sys.stdout.write('filtering the {} points... '.format(len(vor.vertices)))
    vor.filtered_vertices = []
    for vertex in vor.vertices:
        vertex_in_cube = True
        for d in range(len(vertex)):
            if -eps > vertex[d] or vertex[d] > 1 + eps:
                vertex_in_cube = False
        if vertex_in_cube:
            vor.filtered_vertices.append(vertex)

    #sys.stdout.write('done! took {} seconds.\n'.format(time.time() - points_start_time))
    vor.filtered_vertices = np.array(vor.filtered_vertices)
    vor.filtered_points = points

    return vor


def compute_dispersion(vor):

    tree = spatial.KDTree(vor.filtered_vertices)

    #rounded = np.round(tree.data,3)
    #print(rounded)

    #rounded.sort()
    #print(rounded)
    
    #print('')
    #print(vor.filtered_points)
    
    #print(tree.query(vor.filtered_points))
    min_dists = tree.query(vor.filtered_points)


    #print('dispersion:')
    return max(min_dists[0])


def print_bounded_ddim_vor(vor):
    print vor.filtered_regions
    unique_vertex_indices = set()
    for region in vor.filtered_regions:
        for vertex in region:
            unique_vertex_indices.add(vertex)
    print unique_vertex_indices

    print vor.vertices[list(unique_vertex_indices)]
    print len(vor.vertices[list(unique_vertex_indices)])

    vor.vertices[list(unique_vertex_indices)]


    for region in vor.filtered_regions:
        print vor.vertices[region, :]
        for item in np.ndarray.tolist(vor.vertices[region, :]):

            unique_vertices.add((item[0], item[1]))
    print(len(unique_vertices))
    print(unique_vertices)


def print_bounded_vor(vor):
    fig = plt.figure()
    ax = fig.gca()
    # Plot initial points
    ax.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], 'b.')
    # Plot ridges points
    for region in vor.filtered_regions:
        vertices = vor.vertices[region, :]
        ax.plot(vertices[:, 0], vertices[:, 1], 'go')
    # Plot ridges
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    plt.show()





if __name__ == "__main__":
    main()
