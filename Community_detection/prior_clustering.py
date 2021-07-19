# input is the cpv pandas Dataframe

count_edges = np.zeros((cpv.index[-1],cpv.index[-1]))
count_edges.shape

# clustering is performed 100 times
# for every clustering edges between the nodes in the same cluster are created and the number of edges between two nodes are counted in the matrix count_edges
for i in range(100):
    # leave out random 1% of the data --> need to use cpv
    X = np.arange(len(cpv))
    ss = ShuffleSplit(n_splits=1, test_size=0.01)
    for train_index, test_index in ss.split(X):
        train = train_index
    ccpv = pd.DataFrame()
    for index in train:
        SR_Row = pd.Series(cpv.iloc[index])
        ccpv = ccpv.append(SR_Row)
    
    # ccpv is the dataset with the train set
    ccpg = dg.DeepGraph(ccpv)

    ccpg.create_edges(connectors=[cp_node_intersection, 
                                 cp_intersection_strength],
                     no_transfer_rs=['intsec_card'],
                     logfile='create_cpe',
                     step_size=1e7)
    
    # clustering step
    from scipy.cluster.hierarchy import linkage, fcluster

    # create condensed distance matrix
    dv = 1 - ccpg.e.intsec_strength.values
    del ccpg.e

    # create linkage matrix
    lm = linkage(dv, method='average', metric='euclidean')
    del dv

    # form flat clusters and append their labels to cpv
    ccpv['F'] = fcluster(lm, 1000, criterion='maxclust')
    del lm

    # relabel families by size
    f = ccpv['F'].value_counts().index.values
    fdic = {j: i for i, j in enumerate(f)}
    ccpv['F'] = ccpv['F'].apply(lambda x: fdic[x])
    
    # create edges between nodes in one family and add to the matrix count_edges
    ccpg.create_edges(connectors=same_fams, selectors=sel_fams)
    edges = ccpg.e
    edges.reset_index(inplace=True)
    edges.sort_values(by=['s','t'])

    # add edges to table that counts the number of edges between two nodes
    for i in range(len(edges)):
        s = edges.s.iloc[i]
        t = edges.t.iloc[i]
        count_edges[s-1][t-1] = count_edges[s-1][t-1] + 1
        count_edges[t-1][s-1] = count_edges[t-1][s-1] + 1

# reshape the count_edges matrix
# indices dictionary: keys are the original cp values of the respective heatwaves, values are the index numbers in
# the new array count_edges_2
indices = ccpv.index.values
indices.sort()
indices = { i : 0 for i in indices}
count_edges_2 = np.zeros((len(ccpv),len(ccpv)))
counter_i = 0
counter_j = 0
for i in range(len(count_edges)):
    if i+1 in indices:
        counter_j = 0
        for j in range(len(count_edges)):
            if j+1 in indices:
                count_edges_2[counter_i][counter_j] = count_edges[i][j]
                counter_j = counter_j + 1
        indices[i+1] = counter_i
        counter_i = counter_i + 1

# create dataframe from count_edges_2 and prepare it to create a graph from it
c_e_2 = pd.DataFrame(count_edges_2)
c_e_2 = c_e_2.stack().reset_index()
c_e_2.columns = ['source','target','number_of_edges']
c_e_2.sort_values(by=['number_of_edges'], inplace=True, ascending=False)

# replace source and target number with original cp numbers of the heatwaves
for i in range(len(c_e_2)):
    source = c_e_2.source.iloc[i]
    target = c_e_2.target.iloc[i]
    c_e_2.source.iloc[i] = list(indices.keys())[list(indices.values()).index(source)]
    c_e_2.target.iloc[i] = list(indices.keys())[list(indices.values()).index(target)]

