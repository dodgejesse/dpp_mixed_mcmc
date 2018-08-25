import sys
import scipy.spatial as spatial
import numpy as np
import itertools
import time
import dispersion

import discrepancy



# code written to test dispersion.py, specifically the code which reflects points across all faces of the unit cube.



def main():

    ns = discrepancy.get_ns()

    fail_dists = []
    dispersions = {}
    d = 2
    sampler = 'SobolSampler'
    for n in ns:
        dispersions[n] = []
        for example_num in range(30):
            cur_sample = dispersion.get_sample(d=d, sampler=sampler, n=n, snum='{}_1'.format(example_num))
            
            vor = dispersion.bounded_voronoi(cur_sample)
            fail_dists += unit_tests_bounded_vor(vor)
            dispersions[n].append(dispersion.compute_dispersion(vor))

            #if n < d + 2:
            #    continue
            #start_time = time.time()
            #vor = spatial.Voronoi(cur_sample)
            #print('took {} seconds'.format(time.time() - start_time))

            
    print fail_dists
    
    for n in ns:
        avg = np.average(dispersions[n])
        print n, avg
    
def two_sets_of_vs_same(vor, eps):
    # check that vor.filtered_vertices and vor.filtered_regions have the same vertices
    same_start_time = time.time()
    sys.stdout.write('checking that two ways of computing vertices are in the unit cube give the same result... ')
    unique_vertex_indices = set()
    for region in vor.filtered_regions:
        for vertex in region:
            unique_vertex_indices.add(vertex)

    vs = vor.vertices[list(unique_vertex_indices)]

    assert len(vs) == len(vor.filtered_vertices)
    
    for vertex in vor.filtered_vertices:
        match_flag = False
        for vertex_2 in vs:
            if np.all(vertex_2 == vertex):
                match_flag = True
        assert match_flag
            
    sys.stdout.write('done! took {} seconds. looks correct.\n'.format(time.time() - same_start_time))

def has_all_corners_of_cube(vor, eps):
    d = len(vor.vertices[0])
    # check to make sure there's one vertex in each corner
    corners_start_time = time.time()
    sys.stdout.write('checking that the vertices include the corners of the unit cube... ')

    cube_corners = list(itertools.product('10', repeat=d))
    cube_corners = np.array([list(map(int, corner)) for corner in cube_corners])
    
    dists_on_fail = []
    for corner in cube_corners:
        match = False
        for vertex in vor.filtered_vertices:
            if np.all(abs(vertex - corner) < eps):
                # already found a match! can't have two!
                assert not match    

                match = True
        if not match:
            print('')
            print corner
            #print vor.filtered_vertices
            tree = spatial.KDTree(vor.vertices)
            dist, index = tree.query(corner)
            closest_point = tree.data[index]
            dists_on_fail.append(max(abs(closest_point - corner)))
            print closest_point
            
        #assert match
            
    print('done! took {} seconds. successfully found each corner of the unit cube as a vertex.\n'.format(time.time() - corners_start_time))
    return dists_on_fail


def unit_tests_bounded_vor(vor):
    d = len(vor.vertices[0])
    eps = dispersion.get_epsilon(d)

    #two_sets_of_vs_same(vor, eps)
    return has_all_corners_of_cube(vor, eps)




if __name__ == "__main__":
    main()
