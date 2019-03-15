##
# Neighbor Calculator for running phase 1.
from vectorize import Vectorizer
from distance import Distance, Similarity
from math import inf
from operator import itemgetter
from collections import defaultdict
from heapq import nlargest

class Neighbor():
    
    ##
    # NOTE: optional this_vector parameter prevents recomputing this vector if already computed elsewhere.
    @staticmethod
    def knn_textual(k, an_id, model, item_type, database, this_vector = None):
        nearest = {}
        if not this_vector:
            this_vector = Vectorizer.text_vector(an_id, database, model, item_type)


        # Get the id's for all items of the same type.
        if item_type == 'photo':
            an_id = int(an_id)
            others = database.get_photo_ids()
        elif item_type == 'user':
            others = database.get_user_ids()
        elif item_type == 'poi':
            an_id = int(an_id)
            others = database.get_location_ids()
        else:
            raise ValueError('[ERROR] The provided type was invalid.\ntype = ' + str(item_type))
        
        # remove this id from the list
        others.remove(an_id)

        # for each other one, get their vector and calculate distance.
        for other in others:
            
            other_vector = Vectorizer.text_vector(other, database, model, item_type)
            if len(other_vector) == 0: # Find any elements with no textual descriptors 
                continue

            distance = Distance.l_p_distance(3, this_vector, other_vector)


            if len(nearest) < k:
                largest_key, largest_best = None, inf
            else:
                largest_key, largest_best = max(nearest.items(), key=itemgetter(1))

            if distance < largest_best:
                # remove the key with the largest distance if it exists
                if largest_key:
                    nearest.pop(largest_key)
                
                nearest[other] = distance
            
            if all([value == 0 for value in nearest.values()]):
                break
        
        # Return your K nearest
        return nearest


    @staticmethod
    def similarity_by_id(this_vector, ids, database, model, itemtype, k=3):
        other_vectors =[Vectorizer.text_vector(an_id, database, model, itemtype) for an_id in ids]
        return Neighbor.similarity_contribution(this_vector, other_vectors, k)


    @staticmethod
    def similarity_contribution(this_vector, other_vectors, k=3, positional=False):

        avg_contrib = defaultdict(int)
        for other_vector in other_vectors:
            contrib = Similarity.similarity_contribution(this_vector, other_vector, k, positional)
            for key in contrib.keys():
                if positional:
                    avg_contrib[(list(this_vector.keys())[key], list(other_vector.keys())[key])] += contrib[key]
                else:
                    avg_contrib[key] += contrib[key]
        return nlargest(k, avg_contrib.keys(), key=(lambda k: avg_contrib[k]))


    ###############################################################################################


    @staticmethod
    def knn_visual(k, locationid, model, database, this_vector=None):
        nearest = {}
        if not this_vector:
            this_vector = Vectorizer.visual_vector(locationid, database, model)
        
        others = database.get_location_ids()
        others.remove(locationid)
        # Get distance to each other vecctor and add to nearest if it is less than the
        #   distance to an existing vector.
        for other in others:
            other_vector = Vectorizer.visual_vector(other, database, model)
            distance = Distance.l_p_distance(3, this_vector, other_vector, positional=True)

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