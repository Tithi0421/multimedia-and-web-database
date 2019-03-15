from scipy.spatial.distance import cdist

##
# This distance class uses the dictionary keys (the features) to compare two vectors. This
#   should be the default use case.
class Distance():
    
    
    @staticmethod
    def mahalonobis(vector1, table):
        # Create Covariance Matrix
        #   calculate covariance between each feature.
        # sqrt((v1-v2)*covariance*(v1-v2)^T)
        pass

    @staticmethod
    def quadratic(vector, table):
        # Calculate similarity matrix
        #   caluclate similarity between each feature.
        # sqrt((v1-v2)*covariance*(v1-v2)^T)
        pass


    ############################################################################
    ## THESE ARE ALL DISTANCE MEASURES DONE USING DIFFERENT STRATEGIES TO COMPARE
    ## TIME DIFFERENCES.


    @staticmethod
    def l_p_distance(p, vector, table, positional=False):

        distances = table.sub(vector, axis='columns').abs().pow(p).sum(1).pow(1/p)
        return distances

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
    def dot_similarity(vector, table):
        return table.dot(vector.T)

"""
    @staticmethod
    def similarity_contribution(vector1, vector2, k, positional=False):
        ""
            :vector1 - first vector for similarity
            :vector2 - second vector for similarity
            :k  - number of items to return (top k contribution)
        ""
        
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

        """