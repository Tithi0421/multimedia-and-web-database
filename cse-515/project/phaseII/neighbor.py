from distance import Distance, Similarity
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
    def knn_dot(k, vector, table):

        def take_first(elem):
            return elem[0]

        result = []
        vector = np.array(vector)
        indexes = table.index
        table = np.array(table)
        i = 0
        for other_vector in table:
            other_vector_id = indexes[i]
            i = i + 1
            other_vector = np.array(other_vector)
            similarity = abs(dot(vector, other_vector))
            similarity = similarity/(norm(vector)*norm(other_vector))
            result.append([similarity, other_vector_id])

        result.sort(key=take_first)
        result.reverse()
        nearest = result[0:k]

        return nearest

    @staticmethod
    def knn_vd(n, this_matrix, vis_model, k, method, this_matrix_id, database):

        def take_first(elem):
            return elem[0]

        this_matrix = np.array(this_matrix)
        locationid = database.get_location_ids()
        result = []
        for i in locationid:
            that_matrix = Decompose.decompose_loc_vis2(vis_model, k, method, i, database)
            that_matrix = np.array(that_matrix)
            similarity = cosine_similarity(this_matrix, that_matrix)
            similarity = np.array(similarity).sum()
            similarity = similarity / (this_matrix.shape[0] * that_matrix.shape[0])
            result.append([similarity, i])
        result.sort(key=take_first)
        result.reverse()
        nearest = result[0:n]
        for i in result:
            if i[1] == this_matrix_id:
                print("score of current location:" + str(i[0]))
        return nearest

    @staticmethod
    def knn_loc(n, this_vector, vis_model, k, method, database):

        def take_first(elem):
            return elem[0]

        this_vector = np.array(this_vector)
        locationid = database.get_location_ids()
        result = []
        for i in locationid:
            that_matrix = Decompose.decompose_loc_vis2(vis_model, k, method, i, database)
            that_matrix = np.array(that_matrix)
            loc_similarity = []
            for that_vector in that_matrix:
                similarity = abs(dot(this_vector, that_vector))
                similarity = similarity / (norm(this_vector) * norm(that_vector))
                loc_similarity.append(similarity)
            loc_similarity = np.array(loc_similarity).sum()
            loc_similarity = loc_similarity / that_matrix.shape[0]
            result.append([loc_similarity, i])
        result.sort(key=take_first)
        result.reverse()
        nearest = result[0:n]
        return nearest
