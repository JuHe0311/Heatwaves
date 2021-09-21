import numpy as np

#### this module contains all connectors and selectors needed for deep graphs #####

# connectors calculate the distance between every pair of nodes based on their 3D location
# connectors
# distance between x coordinates of two nodes
def grid_2d_dx(x_s, x_t):
    dx = x_t - x_s
    return dx
  
# distance between y coordinates of two nodes
def grid_2d_dy(y_s, y_t):
    dy = y_t - y_s
    return dy

# selectors
def s_grid_2d_dx(dx, sources, targets):
    dxa = np.abs(dx)
    sources = sources[dxa <= 1]
    targets = targets[dxa <= 1]
    return sources, targets

def s_grid_2d_dy(dy, sources, targets):
    dya = np.abs(dy)
    sources = sources[dya <= 1]
    targets = targets[dya <= 1]
    return sources, targets


# create edges between nodes in one family and add to the matrix count_edges
# connector
def same_fams(F_s, F_t):
    dist = F_s - F_t
    return dist

# selector
def sel_fams(dist,sources, targets):
    sources = sources[dist == 0]
    targets = targets[dist == 0]
    return sources, targets  

# create superedges between the supernodes to find heatwave clusters with strong regional overlaps
# compute intersection of geographical locations
def cp_node_intersection(g_ids_s, g_ids_t):
    intsec = np.zeros(len(g_ids_s), dtype=object)
    intsec_card = np.zeros(len(g_ids_s), dtype=np.int)
    for i in range(len(g_ids_s)):
        intsec[i] = g_ids_s[i].intersection(g_ids_t[i])
        intsec_card[i] = len(intsec[i])
    return intsec_card

# compute a spatial overlap measure between clusters
def cp_intersection_strength(n_unique_g_ids_s, n_unique_g_ids_t, intsec_card):
    min_card = np.array(np.vstack((n_unique_g_ids_s, n_unique_g_ids_t)).min(axis=0), 
                        dtype=np.float64)
    intsec_strength = intsec_card / min_card
    return intsec_strength

# compute temporal distance between clusters
def time_dist(dtime_amin_s, dtime_amin_t):
    dt = dtime_amin_t - dtime_amin_s
    return dt

  
