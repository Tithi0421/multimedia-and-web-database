##
#
from operator import itemgetter
from collections import defaultdict
from math import sqrt
from itertools import zip_longest


def get_feature_set(vector1, vector2):
    return list(set(vector1) | set(vector2))

def dot_product(vector1, vector2):
    feature_set = get_feature_set(vector1, vector2)
    product = 0
    for feature in feature_set:
        product += vector1[feature] * vector2[feature]
    return product

##
# This distance class uses the dictionary keys (the features) to compare two vectors. This
#   should be the default use case.
class Distance():
    
    
    @staticmethod
    def mahalonobis(vector1, vector2):
        # Create Covariance Matrix
        #   calculate covariance between each feature.
        # sqrt((v1-v2)*covariance*(v1-v2)^T)
        pass

    @staticmethod
    def quadratic(vector1, vector2):
        # Calculate similarity matrix
        #   caluclate similarity between each feature.
        # sqrt((v1-v2)*covariance*(v1-v2)^T)
        feature_set = get_feature_set(vector1, vector2)
        diff = defaultdict(int)
        for feature in feature_set:
            diff[feature] = vector1[feature] - vector2[feature]
        # TODO Have to add the A matrix in here.
        dp = dot_product(diff, diff)
        return sqrt(dp)


    @staticmethod
    def l_p_distance(p, vector1, vector2, positional=False):

        distance = 0

        if not positional:
            feature_set = get_feature_set(vector1, vector2)
            for item in feature_set:
                distance += (abs(vector1[item] - vector2[item]))**p
        else:
            for value1, value2 in zip_longest(vector1.values(), vector2.values()):
                # Exchange none items with 0s
                if value1 == None:
                    value1 = 0
                if value2 == None:
                    value2 = 0
                distance += abs(value1 - value2)**p
        
        return distance ** (1/p)        
    
    @staticmethod
    def l_2_distance(vector1, vector2, positional=False):
        return Distance.l_p_distance(2, vector1, vector2,positional)


##
# Default Similarity class which uses dictionary keys (feature ids) to compare two vectors. Should
#   be used as the default similarity measure.
class Similarity():
    
    @staticmethod
    def intersection():
        pass
    
    @staticmethod
    def cosine_similarity():
        pass
    
    @staticmethod
    def dot_similarity(vector1, vector2, positional=False):
        similarity = 0

        if not positional:
            feature_set = get_feature_set(vector1, vector2)
            for feature in feature_set:
                similarity += vector1[feature] * vector2[feature]
        else:
            for value1, value2 in zip_longest(vector1.values(), vector2.values()):
                # If either value is none, this should be converted to 0. Since 0 will have no contribution
                #   to similarity (anything * 0 == 0) we can just continue.
                if value1 == None or value2 == None:
                    continue
                similarity += value1 * value2
        
        return similarity
    
    @staticmethod
    def similarity_contribution(vector1, vector2, k, positional=False):
        """
            :vector1 - first vector for similarity
            :vector2 - second vector for similarity
            :k  - number of items to return (top k contribution)
        """
        
        contribution = {}

        if not positional:
            feature_set = get_feature_set(vector1, vector2)
            
            for feature in feature_set:
                sim_contrib = vector1[feature] * vector2[feature]
                key_min = min(contribution.keys(), key=(lambda k: contribution[k]), default=None)
                if len(contribution) < k:
                    # just add.
                    contribution[feature] = sim_contrib
                elif sim_contrib > contribution[key_min]:
                    # pop and add.
                    contribution.pop(key_min)
                    contribution[feature] = sim_contrib
        else:
            for i, (value1, value2) in enumerate(zip_longest(vector1.values(), vector2.values())):
                # If either value is none, this should be converted to 0. Since 0 will have no contribution
                #   to similarity (anything * 0 == 0) we can just continue.
                if value1 == None or value2 == None:
                    continue
                sim_contrib = value1 * value2
                key_min = min(contribution.keys(), key=(lambda k: contribution[k]), default=None)
                if len(contribution) < k:
                    # just add.
                    contribution[i] = sim_contrib
                elif sim_contrib > contribution[key_min]:
                    # pop and add.
                    contribution.pop(key_min)
                    contribution[i] = sim_contrib


        return contribution