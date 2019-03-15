#! /bin/usr/python3.6

from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD as SVD
from pandas import DataFrame
from util import timed
from functools import wraps
import numpy as np
from sklearn.preprocessing import MinMaxScaler



def reindex(f):
    """
    Wrapper to grab indexes from initial table and add them to the reduced table.
    """
    @wraps(f)
    def wrapper(table, k):
        indexes = table.index
        cols = table.columns
        out, principle_comp = f(table, k)
        return DataFrame(out, index=indexes, columns=range(k)), DataFrame(principle_comp, index=range(k), columns=cols)
    return wrapper

class Decompose():

    @staticmethod
    @reindex
    def svd(table, k):
        """
        Decompose table into matrices U . S . V = table.
        :param DataFrame table: table to decompose.
        :param int k: number of components to get from decomposition.
        :return DataFrame reduced matrix.
        """
        matrix = SVD(n_components=k)
        out = matrix.fit_transform(table)
        return out, matrix.components_

    @staticmethod
    @reindex
    def pca(table, k):
        """
        Decompose table into matrices V . L . V^{T} = table and returns the obj matrix.
        :param DataFrame table: table to decompose.
        :param int k: number of components to get from decomposition.
        :return DataFrame reduced matrix.
        """
        matrix = PCA(n_components=k)
        out = matrix.fit_transform(table)
        return out, matrix.components_


    @staticmethod
    @reindex
    def lda(table, k):
        """
        Decompose table into matrices U . S . V^{T} = table and returns the obj matrix.
        :param DataFrame table: table to decompose.
        :param int k: number of components to get from decomposition.
        :return DataFrame reduced matrix.
        """
        matrix = LDA(n_components=k)
        out = matrix.fit_transform(table)
        return out, matrix.components_

    
    @staticmethod
    def switchboard(table, k, method):
        """
        Selects the method to run based on the method name. Useful for every
        :param DataFrame table: the table to reduce.
        :param int k: The number of latent semantics to use.
        :param str method: The method of decomposition to use (pca, lda, svd)
        :return DataFrame with reduced objects.
        """
        if method == 'pca':
            f = Decompose.pca
        elif method == 'svd':
            f = Decompose.svd
        elif method == 'lda':
            f = Decompose.lda
        else:
            raise ValueError("Invalid decomposition method specified: " + method)

        return f(table, k)


    @staticmethod
    def decompose_text(term_space, k, method, database):
        """
        Used by task1 and task2.
        :param str term_space: The type of term space to grab. (user, photo, location)
        :param int k: The number of latent semantics to use.
        :param str method: The method of decomposition to use (pca, lda, svd)
        :return DataFrame with reduced objects and latent semantics.
        """
        table = database.get_txt_desc_table(term_space)
        reduced, principle_comp = Decompose.switchboard(table, k, method)
        return reduced, principle_comp
    

    @staticmethod
    def decompose_loc(k, method, locationid, database):
        table = database.get_vis_table(locationid=locationid)
        if method == 'lda':
            scalar = MinMaxScaler()
            indexes = table.index
            cols = table.columns
            table = scalar.fit_transform(table)
            table = DataFrame(table, columns=cols, index=indexes)
        reduced, ps = Decompose.switchboard(table, k, method)
        return reduced, ps
