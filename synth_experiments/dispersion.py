import scipy as sp
import scipy.spatial as spatial
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import time


# notes: 
# current implementation (reflecting points across all faces of the unit cube) works up to about d=6, n=50.
# without reflecting, can build voronoi diagram up to about d=10, n=50.
# if we need to get up to d=10, we can try doing something like:
# https://stackoverflow.com/questions/36063533/clipping-a-voronoi-diagram-python
# also, discrepancy can be computed in less than a second when d=2 up to n=500

def main():
    import pdb; pdb.set_trace()
    cur_sample = get_sample(d=2, n=29, snum='63_1')
    
    start_time = time.time()
    #vor = spatial.Voronoi(cur_sample,qhull_options='Qb0:0 QB0:1 Qb1:0 QBk:1')
    vor = spatial.Voronoi(cur_sample)


    bounded_lines = get_bounded_lines(vor)
    unbounded_lines = get_unbounded_lines(vor)
    print_lines(bounded_lines, unbounded_lines)
    print_vor(vor)    

    print('took {} seconds'.format(time.time() - start_time))

    vor = bounded_voronoi(cur_sample)
    print_bounded_vor(vor)
    
    
# loads a sample from disk, usually for debugging
def get_sample(d='4', sampler='UniformSampler', n='88', snum='3_1'):
    in_file_name = './pickled_data/dim={}/sampler={}_n={}_d={}_samplenum={}'.format(d, sampler, n,d, snum)

    pkl_file = open(in_file_name)
    cur_sample = pickle.load(pkl_file)
    return cur_sample

    
# epsilon used for bounding a voronoi diagram within the unit cube.
def get_epsilon(d):
    return sys.float_info.epsilon * d * 1000

# the distances between the points in 1-d
def one_dim_vor(unsorted_points):
    points = sorted(unsorted_points)
    #print points
    distances = []
    distances.append(points[0][0] / 2.0)
    for i in range(len(points)):
        cur_point = points[i][0]
        next_point = points[i+1][0] if i+1 < len(points) else 1.0
        distances.append((next_point - cur_point)/2.0)
    return distances


# this function taken from 
# https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells#
# and adjusted to work with dimensions other than 2
def bounded_voronoi(points):
    if len(points[0]) == 1:
        return one_dim_vor(points)
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
    qhull_options = 'Qbb Qc Qz Qx Q12' if len(points[0]) > 4 else 'Qbb Qc Qz Q12'
    vor = sp.spatial.Voronoi(all_points, qhull_options=qhull_options)
    
    vor.filtered_vertices = []
    for vertex in vor.vertices:
        vertex_in_cube = True
        for d in range(len(vertex)):
            if -eps > vertex[d] or vertex[d] > 1 + eps:
                vertex_in_cube = False
        if vertex_in_cube:
            vor.filtered_vertices.append(vertex)

    vor.filtered_vertices = np.array(vor.filtered_vertices)
    vor.filtered_points = points

    return vor

# the function which computes the dispersion of a set of points
def compute_dispersion(vor):
    # if points are 1d, this is a list of distances between points
    if type(vor) == type([]):
        return max(vor)
    # if points > 1d, this is a voronoi diagram
    # we find the largest distance between the vertices and the points
    else:
        tree = spatial.KDTree(vor.filtered_points)
        min_dists = tree.query(vor.filtered_vertices)

        return max(min_dists[0])

# a debugging function that can print relevant items in a d-dimensional voronoi diagram (potentially d != 2)
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

# a function which can filter the regions of a voronoi diagram to those only in the unit cube
# currently just used for debugging / printing
def filter_regions(vor):
    eps = get_epsilon(len(vor.points[0]))
    # Filter regions
    fil_reg_start_time = time.time()
    sys.stdout.write('filtering the {} regions... '.format(len(vor.regions)))
    regions = []
    for region in vor.regions:
        
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                cur_example_in_unit_cube = True
                for d in range(len(vor.vertices[index])):
                    if -eps > vor.vertices[index][d] or vor.vertices[index][d] > 1 + eps:
                        flag = False
                        break
        if region != [] and flag:
            regions.append(region)
    sys.stdout.write('done! took {} seconds.\n'.format(time.time() - fil_reg_start_time))
    vor.filtered_regions = regions


# prints a 2-d voronoi diagram within the unit cube.
def print_bounded_vor(vor):
    filter_regions(vor)
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

    # to print the line which is shortest
    tree = spatial.KDTree(vor.filtered_points)
    min_dists = tree.query(vor.filtered_vertices)

    

    print('')
    print('minimum distances:')
    print min_dists
    print('')
    print('tree.data:')
    print tree.data
    print('')
    print('vor.filtered_vertices')
    print(vor.filtered_vertices)
    

    max_min_dist = -1
    index = 0.1
    for i in range(len(min_dists[0])):
        if min_dists[0][i] > max_min_dist:
            max_min_dist = min_dists[0][i]
            index = i
    print('largest found distance, and index:')
    print(max_min_dist, index)
    first_point = vor.filtered_points[min_dists[1][index]]
    second_point = vor.filtered_vertices[index]
    x_vals = [first_point[0], second_point[0]]
    y_vals = [first_point[1], second_point[1]]

    ax.plot(x_vals, y_vals, 'r')
    
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    plt.savefig('plots/voronoi/vor_diag_bound.pdf', bbox_inches='tight')

# prints a non-bounded 2-d voronoi diagram
def print_vor(vor):
    spatial.voronoi_plot_2d(vor)
    plt.savefig('plots/voronoi/full_vor_diag_bound.pdf', bbox_inches='tight')

# prints the lines found in the voronoi diagram
def print_lines(bounded_lines, unbounded_lines):
    print('')
    print('bounded_lines')
    for line in bounded_lines:
        print(line)

    print('')
    print('unbounded_lines')
    for line in unbounded_lines:
        print(line)
    #print(unbounded_lines)
    
# finds the bounded lines in the voronoi diagram
def get_bounded_lines(vor):
    line_segments = []

    import pdb; pdb.set_trace()

    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            #cur_point = []
            #for i in range(len(vor.vertices[simplex])):
            #    cur_point.append(vor.vertices[simplex][i])
            #line_segments.append(cur_point)
            #line_segments.append([(x, y) for x, y in vor.vertices[simplex]])
            line_segments.append([tuple(coords) for coords in vor.vertices[simplex]])

    return line_segments

# finds the unbounded lines in the voronoi diagram.
# these have a starting point and a direction.
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



if __name__ == "__main__":
    main()
