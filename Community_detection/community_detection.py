# input is a Dataframe from which a graph is created

# arguments: weights=True or False --> states if the graph should be weighted or not, method of community detection --> possible arguments:
# walktrap, multilevel

# create a weighted graph
method = # insert argument here
graph = ig.Graph.TupleList(c_e_2.values, 
                       weights=True, directed=False)
graph.vs["label"] = graph.vs["name"]

# next to do: delete all edges with weight zero!
graph.es.select(weight=0).delete()

if method == "walktrap":
  dendrogram_walktrap = graph.community_walktrap(weights=graph.es['weight'])
  clusters = dendrogram_walktrap.as_clustering()
  pal = ig.drawing.colors.ClusterColoringPalette(len(clusters))
  graph.vs['color']=pal.get_many(clusters.membership)
  ig.plot(graph)
  # save plot somehow
else: 
  dendrogram_multi = graph.community_multilevel(weights=graph.es['weight'])
  ig.plot(dendrogram_multi)
  # save plot somehow
