#! /bin/usr/python3.6

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from util import timed

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
    
    @staticmethod
    def E_distance(vector1, vector2):
        vector1 = np.array(vector1, dtype=np.float)
        vector2 = np.array(vector2, dtype=np.float)
        distances = np.power(np.sum(np.power((vector1 - vector2), 3)), (1./3))
        return distances

    @staticmethod
    def E2_distance(vector1, vector2):
        vector1 = np.array(vector1, dtype=np.float)
        vector2 = np.array(vector2, dtype=np.float)
        distances = np.sqrt(np.sum(np.power((vector1 - vector2), 2)))
        return distances

##
# Default Similarity class which uses dictionary keys (feature ids) to compare two vectors. Should
#   be used as the default similarity measure.
class Similarity():
    
    @staticmethod
    def intersection():
        pass
    
    @staticmethod
    def dot_similarity():
        pass
    
    @staticmethod
    @timed
    def cosine_similarity(table1, table2):
        """
        :param Pandas.Dataframe table1:
        :param Pandas.Dataframe table2:
        """
        index1 = table1.index
        index2 = table2.index

        similarity = cosine_similarity(table1, table2)

        return pd.DataFrame(data=similarity, index=index1, columns=index2)
    
    # finds similarity between two vectors (numpy arrays)
    @staticmethod
    def cos_similarity(vector1, vector2):

        vector1 = np.array(vector1, dtype=np.float)
        vector2 = np.array(vector2, dtype=np.float)
        similarity = np.dot(vector1, vector2) / (np.dot(vector1, vector1) * np.dot(vector2, vector2))

        return similarity


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
