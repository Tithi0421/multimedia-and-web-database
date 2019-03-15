#! /bin/usr/python3.6

from distance import Distance, Similarity, Scoring
from heapq import heappush, merge, nsmallest
from util import timed
from numpy.linalg import norm
from numpy import dot
import numpy as np
from multiprocessing import Pool
from math import ceil
from decompose import Decompose
from sklearn.metrics.pairwise import cosine_similarity
from numpy import union1d
from database import Database
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


# A class is implemented for this tuple to ensure that
#   the heapq can compare when appending.
class DistanceMeasure():
    def __init__(self, an_id, distance):
        self.id = an_id
        self.dist = distance

    def __lt__(self, other):
        return self.dist < other.dist
    
    def __str__(self):
        return "( id = " + str(self.id) + ", distance = " + str(self.dist) + ")"
    
    def __repr__(self):
        return self.__str__()


class Neighbor():    

    @staticmethod
    def knn_worker(vector, table):
        """
        Worker for each process of the distance calculation. Runs the actual distance measure \
        and aggregates the results into heap for fast merge sort with other processes.
        """

        dist_table = Distance.l_p_distance(3, vector, table)
        distances = list()
        for index in dist_table.index:
            heappush(distances, DistanceMeasure(index, dist_table[index]))
        
        return distances



    @staticmethod
    def knn(k, vector, table, processes=1):
        """
        Given a vector (pandas Series) and a table (pandas Dataframe) finds the distance \
        from vector to each row of the table. Returns the indexes of the 'k' rows in table \
        with the shortest distance to vector. Will parallellize the nearest neighbor \
        calculation across p processes if processes is set to a value other than 1.
        """

        if processes > 1:
            rows = table.shape[0]
            size = ceil(rows / processes)

            args = []
            for i in range(processes):
                subtable = table.iloc[i * size : (i + 1) * size]
                args.append((vector, subtable))
            
            p = Pool(processes)
            print("Starting threads!")
            out = p.starmap(Neighbor.knn_worker, args)
            dists = merge(*out)
        
        else:
            dists = Neighbor.knn_worker(vector, table)
        
        nearest = nsmallest(k, dists)
        return nearest



    @staticmethod
    @timed
    def knn_textual(k, an_id, atype, database, processes=1):
        """
        KNN method for textual vectors. Performs the setup of getting the vector and table \
            from the atype (user, photo, location), an_id (vector id) and calls KNN.
        
        The KNN cuts the vector and table to only the columns present in the vector for \
            efficiency and because the professor seems to suggest this is acceptable.
        """
        vector = database.get_txt_vector(atype, an_id)
        vec_indexes = vector.nonzero()[0]
        vector = vector[vec_indexes]
        table = database.get_txt_desc_table(atype).drop(vector.name)
        table = table.iloc[:, vec_indexes]

        return Neighbor.knn(k, vector, table, processes)


    # KNN Specific method for visual vectors. Retrieves the appropriate table 
    #   and vector and then finds the nearest k.
    #
    # If locationid is None, the knn is done for all locations.
    # If model is None, the knn is done for all visual models.
    @staticmethod
    @timed
    def knn_visual(k, photoid, database, locationid=None, model=None, processes=1):
        """
        KNN Specific method for visual vectors. Retrieves the visual description table based \
            on the locationid and model. If locationid is None, the table is for all locations. \
            If model is None, the table is for all visual models. If both are none, the table is \
            for all locations and visual models. Calls KNN on vector and table derived.
        
        The KNN cuts the vector and table to only the columns present in the vector for \
            efficiency and because the professor seems to suggest this is acceptable.
        """
        table = database.get_vis_table(locationid, model)
        vector = table.loc[photoid]
        vec_indexes = vector.nonzero()[0]
        vector = vector[vec_indexes]
        table = table.iloc[:, vec_indexes]

        return Neighbor.knn(k, vector, table, processes)
    
    @staticmethod
    @timed
    def knn_visual_LSH(k, this_image, database, those_images, processes=1):
        """
        KNN Specific method for visual vectors. Retrieves the visual description table based \
            on the imageIds passed. If locationid is None, the table is for all locations. \
            If model is None, the table is for all visual models. If both are none, the table is \
            for all locations and visual models. Calls KNN on vector and table derived.

        The KNN cuts the vector and table to only the columns present in the vector for \
            efficiency and because the professor seems to suggest this is acceptable.
        """
        # edit table to get only desired imageIds(received from LSH bucketing) to be included
        num_comparisons = len(those_images)

        whole_table = pd.DataFrame(database.get_vis_table())
        whole_table = whole_table.T
        whole_table = whole_table.to_dict('list')

        table = {}
        for image in those_images:
            table[image] = whole_table[image]

        table = pd.DataFrame.from_dict(table, orient='index')
        # print(table)

        vector = table.loc[this_image]
        print(vector)
        vec_indexes = vector.nonzero()[0]
        vector = vector[vec_indexes]

        table = table.iloc[:, vec_indexes]

        return Neighbor.knn(k, vector, table, processes), num_comparisons
