##
# Neighbor Calculator for running phase 1.
# from vectorize import Vectorizer
from distance import Distance, Similarity
from heapq import heappush, merge, nsmallest
from util import timed
from numpy import union1d
from multiprocessing import Pool
from math import ceil

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


    ########################################################################
    ## THESE ARE ALL DIFFERENT KNN WORKER METHODS THAT WERE ATTEMPTED TO COMPARE
    ## TIME EFFICIENCY.
    

    # RUNTIME KNN USERS - 283s.
    @staticmethod
    def knn_worker(vector, table):
        
        dist_table = Distance.l_p_distance(3, vector, table)
        distances = list()
        for index in dist_table.index:
            heappush(distances, DistanceMeasure(index, dist_table[index]))
        
        return distances


    # Main KNN method - takes a vector and a table and finds the nearest K vectors
    #   in the table to the provided vector.
    @staticmethod
    def knn(k, vector, table, processes=1):

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




    # KNN Specific method for textual - gets the appropriate table and vector
    #   and then calls generic KNN.
    @staticmethod
    @timed
    def knn_textual(k, an_id, model, atype, database, processes=1):
        vector = database.get_txt_vector(atype, an_id, model)
        vec_indexes = vector.nonzero()[0]
        vector = vector[vec_indexes]
        table = database.get_txt_desc_table(atype, model).drop(vector.name)
        table = table.iloc[:, vec_indexes]

        return Neighbor.knn(k, vector, table, processes)


    ###############################################################################################

"""
    @staticmethod
    def knn_visual(k, locationid, model, database):
        vector = database.get_vis_vector(locationid, model, )


    @staticmethod
    def knn_visual_all(k, locationid, models, database, this_vector=None):
        nearest = {}
        others = database.get_location_ids()
        others.remove(locationid)

        # Calculate this_vector
        if not this_vector:
            this_vector = Vectorizer.visual_vector_multimodel(locationid, database, models)

        # Make a vector where each item is an average for that model
        for other in others:
            other_vector = Vectorizer.visual_vector_multimodel(locationid, database, models)
            # get distance between vectors
            distance = Distance.l_p_distance(3, this_vector, other_vector)
            
            if len(nearest) < k:
                largest_key, largest_best = None, inf
            else:
                largest_key, largest_best = max(nearest.items(), key=itemgetter(1))
    
            if distance < largest_best:
                if largest_key:
                    nearest.pop(largest_key)
                nearest[other] = distance
        
        return nearest
                

    
    @staticmethod
    def visual_sim_contribution(this_vector, ids, database, model, k=3):
        other_vectors = [Vectorizer.visual_vector(locid, database, model) for locid in ids]
        return Neighbor.similarity_contribution(this_vector, other_vectors, k, positional=True)
    

    @staticmethod
    def visual_sim_multimodal(this_vector, ids, database, models, k=3):
        other_vectors = [Vectorizer.visual_vector_multimodel(locid, database, models) for locid in ids]
        return Neighbor.similarity_contribution(this_vector, other_vectors, k, positional=True)
        """
    