#! /bin/usr/python3.6
import igraph
# igraph also requires pycairo
from os.path import isfile
from util import timed
from functools import wraps
import pickle
from collections import defaultdict
import pandas as pd
from distance import Similarity
from sys import stdout


class Edge():
    """
    Simple abstraction for easier use by others. Its easier than creating weight vectors.
    """

    def __init__(self, start, end, weight):
        """
        :param int start: id of start node.
        :param int end: id of end node.
        :param float weight: weight of edge.
        """
        self.start = start
        self.end = end
        self.weight = weight
    
    def __str__(self):
       return f"{str(self.start)} ----{str(self.weight)}-----> {str(self.end)}\n"
    
    def __repr__(self):
        return self.__str__()



class Graph():

    # CONSTANTS DEFINITIONS
    CLUSTER = 'cluster'
    SIM = 'weight'

    def __init__(self, graph=None, similarity=None):
        if graph:
            self.__graph__ = graph
        else:
            self.__graph__ = igraph.Graph(directed=True)

        # data structor to store nodes in a given cluster for faster access.
        self.clusters = {}

        if not similarity is None:
            self.add_similarity(similarity)



    ###########################################################################################
    ##  Low level background methods.
    ###########################################################################################
    

    def __add_edges__(self, edges, weights, graph=None):
        """
        Add edges from standard notation (start, end) and weight companion list.
        :param list edges: list of (start, end) tuples.
        :param list weights: list of floats.
        """
        if graph is None:
            graph = self.__graph__

        # translate from name tuples to object tuples
        edges = [(self.__get_node_by_name__(e1), self.__get_node_by_name__(e2)) for e1, e2 in edges]

        # get base list based on if list exists or not. Probably a cleaner way to do this.
        num_edges = len(graph.es)
        if num_edges == 0:
            temp_weights = list()
        else:
            temp_weights = graph.es[Graph.SIM]
        temp_weights.extend(weights)

        graph.add_edges(edges)
        graph.es[Graph.SIM] = temp_weights
        return



    def __find_edge__(self, start_node, end_node, graph=None):
        """
        find edge by its end points
        :param int start: source node of edge.
        :param int end: end node of edge.
        """
        if graph is None:
            graph = self.__graph__

        start = self.__get_node_by_name__(start_node)
        end = self.__get_node_by_name__(end_node)

        edges = graph.es.find(_source=start.index, _target=end.index)
        return edges
    

    def __add_label__(self, node, label, value, graph=None):
        """
        Add label to node.
        :param int node: Id of node to add label to.
        :param str label: the label to add the value to.
        :param obj value: the value to add.
        :param igraph graph: Optional graph.
        """
        if graph is None:
            graph = self.__graph__
        n = self.__get_node_by_name__(node) 
        n[label] = value
        return n
    


    def __get_node_by_name__(self, name, graph=None):
        """
        retrieve a node by its name.
        :param obj name: identifier for this node.
        """
        if graph is None:
            graph = self.__graph__
        node = graph.vs.select(name=name) 
        if len(node) != 1:
            raise ValueError('The node name provided couldn\'t be found or is repeated: %s' % name)
        return node[0]


    ###########################################################################################
    ##  Interface methods
    ###########################################################################################

    
    def add_edges(self, edges):
        """
        Adds edges from an iterable
        :param list edges: List of Edge objects.
        """
        if not isinstance(edges, list):
            raise ValueError('Edges parameter should be a list of edge objects.')
        
        new_edges = [(edge.start, edge.end) for edge in edges]
        weights = [edge.weight for edge in edges]
        self.__add_edges__(new_edges, weights)
    

    def add_vertices(self, vertices):
        """
        Add vertices to graph by name.
        :param list vertices: list of vertice names (can be any basic datatype)
        """
        if not isinstance(vertices, list):
            raise ValueError('Vertices parameters should be a list of names.')

        for vertex in vertices:
            self.__graph__.add_vertex(name=vertex, cluster=None)


    @timed 
    def add_similarity(self, similarity):
        """
        Adds similarity matrix information to database. Adds all ~8900 photos and the weighted edges
        to their k most similar partners.
        :param Pandas.Dataframe similarity: photo-photo similarity matrix
        """
        assert(all(similarity.index == similarity.columns))
        # TODO photo too large for C int, breaks igraph lib.
        photos = list(similarity.index)
        self.add_vertices(photos)
        # get the nearest neighbors similarity wise to each image and add to database.
        for i, photo in enumerate(photos):
            print(str(i))
            photo = int(photo)
            row = similarity.loc[photo].sort_values(ascending=False)
            edges = [(photo, int(other)) for other in row.index]
            weights = [other for other in row]
            self.__add_edges__(edges, weights)
        return
    

    @timed
    def add_edge_dict(self, edge_dict):
        """
        initializes graph from similarity matrix.
        :param Pandas.Dataframe similarity:
        """
        nodes = list(edge_dict.keys())
        # form edges.
        edges = list()
        weights = list()
        for node in nodes:
            edges.extend([(node, other) for other in edge_dict[node]])
            weights.extend([edge_dict[node][other] for other in edge_dict[node]])
        self.add_vertices(nodes)
        self.__add_edges__(edges, weights)
        return

    

    def get_images(self):
        return self.__graph__.vs()['name']


    
    def node(self, name):
        """
        Get node by name. In this application, it is the image id.
        :param obj name: Name of node to retrieve.
        """
        return self.__get_node_by_name__(name)
        

    def edge(self, src, end):
        """
        Get edge interface object between two nodes. Used to find weight of edge if it exists.
        ;param obj src: Name of start node, or tail.
        :param obj end: Name of end node, or head.
        """
        e = self.__find_edge__(src, end)
        if e:
            return Edge(src, end, e[Graph.SIM])
        # if edge doesn't exist.
        return None


    def subgraph(self, out_degree):
        """
        NOTE: Untested!
        Returns a subgraph of the main graph with max out degree of k.
        When removing edges it keeps the largest k.
        :param int out_degree: number of out edges for each vertex.
        """
        subgraph = self.__graph__.copy()
        for vertex in subgraph.vs:
            edges = [subgraph.es[i] for i in subgraph.incident(vertex, mode=igraph.OUT)]
            while vertex.outdegree > out_degree:
                # find and remove edge with smallest weight.
                # NOTE: Removing edge is simple as getting the object and calling .delete()
                min_edge = min(edges, key=lambda a: a[Graph.SIM])
                min_edge.delete()
        return subgraph

    
    def neighbors(self, node, clusters=[]):
        """
        Returns the neighboring nodes to this one with the edge weight between them.
        :param int node: id of node to find neighbors of.
        :param list clusters: iterable of clusters to search for neighbors in.
        :return list edges: returns list of edges to neighboring nodes, indicating the similarity.
        """
        n = self.__get_node_by_name__(node)
        neighbors = n.neighbors(mode='OUT')

        # if a set of clusters were specified, then limit neighbors to only those in the clusters.
        if clusters:
            temp = []
            for cluster in clusters:
                vertices = self.clusters[cluster]
                intersection = [i for i in neighbors if i in vertices]
                temp.extend(intersection)
                [neighbors.remove(i) for i in intersection] # for efficiency
            neighbors = temp
        
        # turn into Edge interface object.
        return_val = [self.edge(node, neighbor['name']) for neighbor in neighbors]
        
        return return_val

    
    def add_to_cluster(self, node, cluster):
        """
        Adds cluster label to the node. NOTE: If node is already in a cluster, this will be
            overwritten in teh graph itself for display. The previous cluster will be restored
            when the node is removed from the new cluster.
        :param int node: node id to add label to.
        :param str cluster: cluster identifier to add to node.
        """
        # if this is an iterable of nodes, run code for each.
        if isinstance(node, list):
            for n in node:
                self.add_to_cluster(n, cluster)
            return
        
        # Add to local specification.
        if not cluster in self.clusters:
            self.clusters[cluster] = list()
        self.clusters[cluster].append(node)
        # add to graph.
        self.__add_label__(node, Graph.CLUSTER, cluster)
    

    def remove_from_cluster(self, node, cluster):
        """
        Removes node from cluster. This sets the cluster in the graph to another cluster if the node
            belongs to a second one, or None otherwise.
        :param int node: node id to add label to.
        :param str cluster: cluster identifier to remove from.
        """
        if not cluster in self.clusters:
            raise ValueError("Cluster specified doesn't exist: %s" % cluster)
        
        if isinstance(node, list):
            for n in node:
                self.remove_from_cluster(n, cluster)
            return

        self.clusters[cluster].remove(node)
        # see if this has any alternative labels in the clusters dictionary.
        c = None
        for cluster, l in self.clusters.items():
            if node in l:
                c = cluster
                break
        self.__add_label__(node, Graph.CLUSTER, c)
    


    def clear_clusters(self, graph=None):
        """
        Deletes cluster data from graph.
        :param Graph graph: Graph obj. If None, uses default.
        """
        if graph is None:
            graph = self.__graph__

        self.clusters = {}
        clusters = [None for v in graph.vs]
        self.__graph__.vs[Graph.CLUSTER] = clusters
    


    def get_adjacency(self, graph=None, weighted=False):
        """
        """
        if graph is None:
            graph = self.__graph__

        param_dict = {}
        if weighted:
            param_dict['attribute'] = Graph.SIM
            param_dict['default'] = 0
         
        a = graph.get_adjacency(**param_dict)
        labels = graph.vs['name']
        return pd.DataFrame(data=a.data, index = labels, columns=labels)

    

    def display(self, graph=None, clusters=[], filename='out.png', emphasis=[], emph_color=None, label=False):
        """
        Show representation of the graph. Saves to a png file so that the image can be viewed in
            image application with more capable zooming opportunities.
        :param igraph graph: graph to display. If none, uses main graph.
        :param list clusters: clusters (by name) to show on the display. If none are provided, shows
            all nodes as if in a single cluster.
        :param str filename: location to save the display to.
        :param list emphasis: List of nodes (by name) to emphasize. Nodes will be made larger and have
            unique coloring, if emph_color is set to a value.
        """
        if graph is None:
            graph = self.__graph__
        
        # set up colors by cluster label.
        cdict = defaultdict(lambda: 'grey')
        colors = ['blue', 'red', 'green', 'yellow', 'white', 'black', 'orange', 'purple', 'grey', 'pink']
        for cluster, color in zip(clusters, colors):
            cdict[cluster] = color

        # Set up visual_sytle
        visual_style = {}
        dim = 14000
        visual_style['bbox'] = (dim, dim)
        v_size = [50 if vertex['name'] in emphasis else 20 for vertex in graph.vs]
        visual_style['vertex_size'] = v_size
        # color emphasis if listed in emphasis, else do the clusters coloring
        v_colors = [emph_color if (not emph_color is None and vertex['name'] in emphasis) \
                    else cdict[vertex[Graph.CLUSTER]] for vertex in graph.vs]
        visual_style['vertex_color'] = v_colors
        visual_style['layout'] = graph.layout('drl') # layout optimized for large graphs.
        if label:
            visual_style['vertex_label'] = graph.vs['name'] # display the id on the node.
        #if len(graph.es) > 0:
        #    visual_style['edge_label'] = graph.es[Graph.SIM] # set edge weights to display

        igraph.plot(graph, filename, **visual_style)
    


    def display_text(self, graph=None, file=stdout):
        """
        """
        if isinstance(file, str):
            file = open(file, 'w+')
        
        if graph is None:
            graph = self.__graph__
        
        for node in graph.vs():
            neighbors = node.neighbors(mode='out')
            names = [int(a['name']) for a in neighbors]
            file.write(f"Node {int(node['name'])} : Neighbors = {names}\n")




    def display_clusters_text(self, clusters=None, keys=None, file=stdout):
        """
        Show clusters as text. Saves them to file if a file is specified, else goes to stdout.
        :param list clusters: clusters dictionary. If None, uses default.
        :param list keys: clusters to print from dictionary. If None, uses all.
        :param str filename: filename to print to. If no file, uses stdout.
        """
        if isinstance(file, str):
            file = open(file, 'w+')
        
        if clusters is None:
            clusters = self.clusters
        
        if keys is None:
            keys = clusters.keys()

        for cluster in keys:
            file.write('CLUSTER = %s\n' % cluster)
            if not cluster in clusters:
                continue
            for img in clusters[cluster]:
                file.write('\t%s\n' % img)

    

    def nodes_in_cluster(self, cluster, use_dict=True, graph=None):
        """
        Get all nodes in the cluster specified. Uses dictionary or graph.
        :param obj cluster: cluster identifier.
        """
        if use_dict:
            return cluster[cluster]
        else:
            if graph is None:
                graph = self.__graph__
            kwargs = {Graph.CLUSTER : cluster}
            return graph.vs.select(**kwargs)


    def drop(self):
        """
        Deletes all graph data. Use with Caution!
        """
        pass

    
    def save(self, location):
        """
        Saves graph to binary for easy loading later.
        :param path location: location to save to.
        """
        self.__graph__.write_pickle(fname=location)
        with open(location + '_dict', 'wb+') as f:
            pickle.dump(self.clusters, f, protocol=pickle.HIGHEST_PROTOCOL) 
    

    @staticmethod
    def load(location):
        """
        Load graph from binary.
        :param path location: location to read from. 
        """
        if not isfile(location):
            raise FileNotFoundError('The location specified does not exist: %s' % location)
        g = igraph.Graph.Read_Pickle(fname=location)
        # create clusters dictionary again
        graph = Graph(graph=g)
        with open(location + '_dict', 'rb') as f:
            graph.clusters = pickle.load(f)
        return graph


class GraphDriver():
    """
    Quick testing suite to validate the graph.
    """
    @staticmethod
    def test():
        g = Graph()
        g.add_vertices(['foo', 'bar', 'baz'])
        g.display()
        input('Press anything to continue.')
        g.add_vertices(['other', 'nodes'])
        g.display()
        input('Press anything to continue')
        g.add_edges([Edge('foo', 'bar', 1), Edge('bar', 'baz', 2.3), Edge('baz', 'foo', 5)])
        g.display()
        input('Press anything to continue.')
        g.add_edges([Edge('other', 'nodes', 3.14), Edge('nodes', 'bar', 6.0)])
        g.display()
        input('Press anything to continue')
        # test emphasis
        g.display(emphasis=['bar', 'nodes'])
        input('Press anything to continue.')
        g.display(emphasis=['bar', 'other'], emph_color='red')
        input('Press anything to continue.')
        n = g.node('bar')
        e = g.edge('bar', 'baz')
        print(f"Node = {n}")
        print(f"Edge = {e}")
        input('Press anything to continue.')
        # Test neighbors.
        temp = g.neighbors('bar')
        print(f"Neighbors to bar = {temp}")
        temp = g.neighbors('nodes')
        print(f'Nieghbors to nodes = {temp}')
        input('Press anything to continue.')
        # test Cluster specification.
        g.add_to_cluster(['foo', 'bar', 'baz'], 'C1')
        g.display(clusters=['C1'])
        input('Press anything to continue...')
        g.add_to_cluster(['other', 'nodes', 'bar'], 'C2')
        g.display(clusters=['C1', 'C2'])
        input('Press anything to Continue.')
        g.display(clusters=['C1', 'C2'], emphasis=['other'])
        input('Press anything to continue.')
        g.display(clusters=['C1', 'C2'], emphasis=['other'], emph_color='orange')
        input('Press anything to continue.')
        g.remove_from_cluster(['bar'], 'C1')
        g.display(clusters=['C1', 'C2'])
        input('Press anything to continue.')
        g.remove_from_cluster(['nodes', 'other'], 'C2')
        g.display(clusters=['C1', 'C2'])
        input('Press anything to continue.')
        # test load and save.
        g.save('foo')
        input('Press anything to continue...')
        del(g)
        g = Graph.load('foo')
        g.display(clusters=['C1', 'C2'])
        input('Press anything to continue')
    
    @staticmethod
    def test2():
        g = Graph()
        df = pd.DataFrame([[1,2,3,4], [7,2,3,1], [1,2,4,5]])
        sim = Similarity.cosine_similarity(df, df)
        g.add_similarity(sim)
        g.display()

    @staticmethod
    def test_adjacency():
        g = Graph()
        df = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]])
        sim = Similarity.cosine_similarity(df, df)
        print(sim)
        g.add_similarity(sim)
        s2 = g.get_adjacency(weighted=True)
        assert(sim.equals(s2))

#GraphDriver.test()
#GraphDriver.test2()
# GraphDriver.test_adjacency()

""""
Notes:
Remove self similarity arcs.


MULTIPLE LABELS CAN BE APPLIED TO ONE ID.

Methods for graph:
    display w/ a provided set of nodes larger and different color.

    Task 2 - cluster methods above should be sufficient
    Task 3 - Color and resize K dominant nodes.
    Task 4 - 
    Task 5 - Output all nodes for indexing by task 5.


For 5 what ar ethe vectors that will be provided? What are their dimensions? 

For Spectral Partiitioning:
    1. Get adjacency matrix from igraph.
    2. Pass to scipy to make into laplacian.
    3. Use numply.linalg.eig to get eigenvectors
    4. Use second eigenvector to form two clusters.
"""

"""
for img in images:
    label = labels[img]
    graph.add_to_cluster(img, label)
graph.display(clusters=labels.keys())
"""