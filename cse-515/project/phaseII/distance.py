import numpy as np
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

class Scoring():

    @staticmethod
    def score_matrix(sim_matrix):
        similarity = 0
        denominator = sum(sim_matrix.shape) / 2
        for i in range(min(sim_matrix.shape)):
            x, y = np.where(sim_matrix == np.max(sim_matrix))[0][0], np.where(sim_matrix == np.max(sim_matrix))[1][0]
            similarity += sim_matrix[x, y]
            sim_matrix = np.delete(sim_matrix, (x), axis=0)
            sim_matrix = np.delete(sim_matrix, (y), axis=1)
        return similarity / denominator

    # Explanation of the above algorithm to find a similarity score for a similarity matrix:
    # Sum of max. vals is equal to 0.
    # Select maximum value from doc-doc matrix and add it to Sum of max. vals.
    # Remove row and column with maximum value from the matrix.
    # Repeat steps 2-3 until rows or columns are ended.
    # Denominate Sum of max. vals by average number of key words in two texts.
    # Final estimation would be equal to 1, if both documents have identical length, and every word from Doc 1 has equivalent in Doc 2.
