#! /bin/usr/python3.6 from loader import Loader
from distance import Similarity
from graph import Graph
from os.path import isdir, isfile, join, realpath
from os import mkdir
import argparse
from util import timed, show_images, save_images, safe_mkdir, images_to_web
import numpy as np
import numpy.linalg as la
import scipy.cluster.vq as vq
from scipy.sparse import csc_matrix
from collections import namedtuple, defaultdict
from itertools import product
from task5 import LSH
from loader import Loader
from task6 import KNN, PPR


class Interface():

    def __init__(self, runall=False):
        self.__database__ = None
        self.__graph__ = None
        self.__valid_types__ = ['photo', 'user', 'poi']
        self.__vis_models__ = ['CM', 'CM3x3', 'CN', 'CN3x3',
                               'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
        self.__io__(runall)

    def __io__(self, runall=False):
        print("Welcome to the CSE 515 data software. Please enter a command.\
              \nEnter \"help\" for a list of commands.")

        # for precomputing everything.
        if runall:
            # self.create_graphs()
            self.run_two_to_three()
            # self.run_four_to_six()
            return

        def __filepath__(parser, arg):
            if not isdir(arg):
                parser.error("The directory %s doesn't exist." % arg)
            else:
                return True

        parser = argparse.ArgumentParser()
        parser.add_argument('-task', type=int, choices=range(0, 7), required=True, metavar='#')
        parser.add_argument('--k', type=int, metavar='#')
        parser.add_argument('--alg', type=str, metavar="algorithm_to_use")
        parser.add_argument('--imgs', type=int, nargs='+', metavar='img1 img2 ...')
        parser.add_argument('--imageId', type=int, metavar='imageId')
        parser.add_argument('--load', type=str, metavar='filepath')
        parser.add_argument('--graph', type=str, metavar='filename')
        parser.add_argument('--layers', type=int, metavar='L')
        parser.add_argument('--hashes', type=int, metavar='k')
        parser.add_argument('--bins', type=int, metavar='b')
        parser.add_argument('--file', type=str, metavar='filepath')
        # parser.add_argument('--cluster', type=int, metavar='c')
        parser.add_argument('--vectors', type=str)  # Assuming this is a file locaiton
        while True:
            user_input = input("\nEnter a Command:$ ")
            user_input = user_input.split(' ')
            try:
                args = parser.parse_args(user_input)
            except argparse.ArgumentError as e:
                print("The command line input could not be parsed.")
                print(e)
                continue
            except BaseException as e:
                print('The command line arguments could not be parsed.')
                print(e)
                continue

            # load the database from the folder.
            if args.load:
                try:
                    self.load(args)
                except Exception as e:
                    print('Something went wrong during database load.')
                    print(e)

            # load the graph from the file.
            if args.graph:
                try:
                    self.graph(args)
                except Exception as e:
                    print('Something went wrong loading the graph.')
                    print(e)
            
            if args.task == 0:
                continue

            # Get method with this name - makes it easy to create new interface methods
            #   without having to edit this method.
            try:
                method = getattr(self, 'task' + str(args.task))
            except AttributeError:
                print('The command specified was not a valid command.\n')

            try:
                method(args)
            except Exception as e:
                print('Something went wrong while executing ' + str(method.__name__))
                print(str(e))

    def help(self, args):
        """
        Command:\thelp
        Description:\tPrints the interface information about the program.
        """

        print("The following are valid commands to the program.")
        for item in dir(self):
            if not item.startswith('__'):
                method = getattr(self, item)
                print(method.__doc__)


    def load(self, args):
        """
        Command:\t--load <filepath>
        Description:\tLoads the database at that file path. If the file does not exist, will prompt to create a new database using the folder of a users choice.
        Arguments:
        \t<filepath> - A valid file path in the system.
        """

        folder = realpath(args.load)

        if not isdir(folder):
            print("[ERROR] The provided path was not a folder, and therefore not a valid data directory.")
            return

        self.__database__ = Loader.make_database(folder)
        print("Database loaded successfully.")


    def graph(self, args):
        """
        Command:\t--graph <filepath>
        Description:\tLoads the graph from the binary (pickle file) specified.
        Arguments:
        \t<filepath> a valid file path in the system.
        """
        f = args.graph

        if not isfile(realpath(f)):
            f = f'./precomputed/graph{f}/graph{f}'
            # print("[ERROR] The provided path was not a valid file.")
            # return

        self.__graph__ = Graph.load(realpath(f))
        print('Graph loaded successfully.')


    @timed
    def task1(self, args, path='.'):
        """
        -task 1 --k #.
        """
        if self.__graph__ == None:
            if args.k == None:
                raise ValueError('Parameter K must be defined for task 1.')
            k = int(args.k)
            self.__graph__ = Loader.make_graphs(self.__database__, k, path=path)
        # visualize graph.
        self.__graph__.display_text(file=join(path, f'graph.txt'))
        self.__graph__.display(filename=join(path, f'task1_.png'))
    


    @timed
    def task2(self, args, path='.'):
        """
        -task 2 --k #
        """
        if args.k == None:
            raise ValueError('K must be defined for task 2.')
        c = int(args.k)
        # alg = args.alg

        # YOUR CODE HERE.
        clusters = {}
        clusters1 = {}
        images = self.__graph__.get_images()
        list_of_clusters = []
        list_of_clusters1 = []
        lengOfA = 0
        lengOfB = 0
        lengOfClusters = {}
        A = self.__graph__.get_adjacency()
        D = np.diag(np.ravel(np.sum(A, axis=1)))
        L = D - A
        l, U = la.eigh(L)
        f = U[:, 1]
        labels = np.ravel(np.sign(f))
        # Clustering function
        for image in images:

            if (labels[images.index(image)] == -1):
                cluster = 'A'
                clusters[image] = cluster
                lengOfA += 1

                if not cluster in list_of_clusters:
                    list_of_clusters.append(cluster)
            else:
                cluster = 'B'
                clusters[image] = cluster
                lengOfB += 1
                if not cluster in list_of_clusters:
                    list_of_clusters.append(cluster)

        # display
        # for image in images:
            # self.__graph__.add_to_cluster(image, clusters[image])
        # self.__graph__.display_clusters_text(keys=list_of_clusters, file=join(path, 'task2.txt'))
        # self.__graph__.display(clusters=list_of_clusters, filename=join(path, 'task2.png'))
        images_to_web(clusters, self.__database__, join(path, 'task2_spectral.html'))
        print("Clusters in A:", lengOfA)
        print("Clusters in B:", lengOfB)

        #Algorithm 2
        l1, u1 = la.eigh(A)
        u1.sort(axis=1)
        f1 = u1[:, -c:]
        means, labels1 = vq.kmeans2(f1, c)
        for j in range(c):
            indices = [i for i, x in enumerate(labels1) if x == j]
            lengOfClusters[j] = len(indices)
            for everyIndice in indices:
                cluster = j
                clusters1[images[everyIndice]] = cluster
            if not j in list_of_clusters1:
                list_of_clusters1.append(j)

        # for image in images:
            # self.__graph__.add_to_cluster(image, clusters1[image])
        # self.__graph__.display_clusters_text(keys=list_of_clusters1, file=join(path, 'task2_kspectral.txt'))
        # self.__graph__.display(clusters=list_of_clusters1, filename=join(path, 'task2_kspectral.png'))
        images_to_web(clusters1, self.__database__, join(path, 'task2_kspectral.html'))
        print("Second algorithm:\n", lengOfClusters)

    @timed
    def task3(self, args, path='.'):
        """
        -task 3 --k # 
        """
        if args.k == None:
            raise ValueError('K must be defined for task 3.')
        k = int(args.k)

        # YOUR CODE HERE.
        G = self.__graph__.get_adjacency()
        images = self.__graph__.get_images()
        n = G.shape[0]
        s = 0.86
        maxerr = 0.01

        # transform G into markov matrix A
        A = csc_matrix(G, dtype=np.float)
        rsums = np.array(A.sum(1))[:, 0]
        ri, ci = A.nonzero()
        A.data /= rsums[ri]

        # bool array of sink states
        sink = rsums == 0

        # Compute pagerank r until we converge
        ro, r = np.zeros(n), np.ones(n)
        # account for sink states
        Di = sink / float(n)
        # account for teleportation to state i
        Ei = np.ones(n) / float(n)
        # while np.sum(np.abs(r - ro)) > maxerr:
        for _ in range(150):

            if np.sum(np.abs(r - ro)) <= maxerr:
                break
            ro = r.copy()
            # calculate each pagerank at a time
            for i in range(0, n):
                # in-links of state i
                Ai = np.array(A[:, i].todense())[:, 0]


                r[i] = ro.dot(Ai * s + Di * s + Ei * (1 - s))

        weights = r / float(sum(r))
        orderedWeights = np.argsort(weights)
        ReorderedWeights = np.flipud(orderedWeights)
        # m = max(weights)
        # ind = np.argmax(weights)
        listOfImages = list()
        for xx in range(k):
            listOfImages.append(images[ReorderedWeights[xx]])
        # weightDict = {}
        # for xx in range(len(weights)):
        #     weightDict[xx] = weights[xx]
        print(listOfImages)
        show_images(listOfImages, self.__database__)
        save_images(listOfImages, self.__database__, join(path, 'out'))

    @timed
    def task4(self, args, path='.'):
        """
        -task 4 --k # --imgs id1 id2 id3
        """
        if args.k == None or args.imgs == None:
            raise ValueError('K and Imgs must be defined for task 4.')
        k = int(args.k)
        imgs = list(args.imgs)
        # 6 2976167 83 38391649 299 135049429
        # YOUR CODE HERE.
        G = self.__graph__.get_adjacency()
        images = self.__graph__.get_images()
        indexes = list()
        for x in imgs:
            indexes.append(images.index(x))
        n = G.shape[0]
        s = 0.6
        maxerr = 0.1

        # transform G into markov matrix A
        A = csc_matrix(G, dtype=np.float)
        rsums = np.array(A.sum(1))[:, 0]
        ri, ci = A.nonzero()
        A.data /= rsums[ri]

        # bool array of sink states
        sink = rsums == 0

        Ei = np.zeros(n)
        for ii in indexes:
            Ei[ii] = 1 / len(imgs)
        # Compute pagerank r until we converge
        ro, r = np.zeros(n), np.ones(n)
        # while np.sum(np.abs(r - ro)) > maxerr:
        for _ in range(100):

            if np.sum(np.abs(r - ro)) <= maxerr:
                break

            ro = r.copy()
            # calculate each pagerank at a time
            for i in range(0, n):
                # in-links of state i
                Ai = np.array(A[:, i].todense())[:, 0]
                # account for sink states
                Di = sink / float(n)
                # account for teleportation to state i

                r[i] = ro.dot(Ai * s + Di*s + Ei * (1 - s))

        weights = r / float(sum(r))
        orderedWeights = np.argsort(weights)
        ReorderedWeights = np.flipud(orderedWeights)
        # m = max(weights)
        # ind = np.argmax(weights)
        listOfImages = list()
        for xx in range(k):
            listOfImages.append(images[ReorderedWeights[xx]])
        print(listOfImages)
        show_images(listOfImages, self.__database__)
        save_images(listOfImages, self.__database__, join(path, 'out'))


    @timed
    def task5(self, args, path='.'):
        """
        Use as:
        -task 5 --layers # --hashes # --k # --imageId #
        """
        if args.layers == None or args.hashes == None or \
                args.k == None or args.imageId == None:
            raise ValueError('Layers, Hashes, Vectors, K, and IMG must all be defined for task 5.')

        layers = int(args.layers)
        hashes = int(args.hashes)
        t = int(args.k)
        imageId = args.imageId
        if args.vectors:
            vectors = str(args.vectors)

        # YOUR CODE HERE
        lsh = LSH()
        nearest = lsh.main(layers, hashes, imageId, vectors=(), t=t, database=self.__database__)
        show_images(nearest, self.__database__)
        save_images(nearest, self.__database__, join(path, 'out'))


    @timed
    def task6(self, args, path='.'):
        """
        -task 6 --alg (knn/ppr) --file input/file/path (--k # if knn)
        """
        if args.alg == None and args.file == None:
            raise ValueError('Alg must be defined for task 6.')
        
        if not isfile(realpath(args.file)):
            raise ValueError('File specified was not a valid file.')

        imageIDs = list()
        labels = list()
        with open(args.file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.isspace():
                    continue
                imgid, label = line.split()
                imageIDs.append(int(imgid))
                labels.append(label)
        clusters = set(labels)

        alg = str(args.alg)
        print(alg)

        # YOUR CODE HERE

        if alg == "knn":
            if args.k != None:
                k = int(args.k)
            else:
                k = 3

            knn = KNN()
            result = knn.knn_algorithm(imageIDs, labels, k, self.__database__)
            # print("result: " + str(result))

        elif alg == "ppr":

            G = self.__graph__.get_adjacency()
            images = self.__graph__.get_images()
            indexes = list()

            for x in imageIDs:
                indexes.append(images.index(x))

            ppr = PPR()
            result = ppr.ppr_algorithm(imageIDs, labels, indexes, G, images)
            # print("result: " + str(result))

        else:
            raise ValueError('An invalid algorithm was passed to task 6.')

        
        # for image in result:
            #self.__graph__.add_to_cluster(image, result[image])
        #self.__graph__.display_clusters_text(keys=clusters, file=join(path, f'task6{alg}.txt'))
        #self.__graph__.display(clusters=clusters, filename=join(path, f'task6{alg}.png'))
        images_to_web(result, self.__database__, join(path, f'task6{alg}.html'))
            

    def quit(self, *args):
        """
        Command:\tquit
        Description:\tExits the program and performs necessary cleanup.
        """
        exit(1)

    
    def run_two_to_three(self):
        """
        Utility to precompute all inputs desired by the professor.
        """
        graphs = [3, 10] # 10 
        task2 = [2,4,10] # 10
        task3 = [] # [1, 5,10]

        basepath = realpath('./precomputed')
        safe_mkdir(basepath)

        # Create named tuple to simulate args.
        fields = ['k', 'alg', 'imgs', 'imageId', 'load', 'graph', 'layers', 'hashes', 'file', 'bins']
        Args = namedtuple('Arguments', fields)
        Args.__new__.__defaults__ = (None,) * len(fields)


        # call load once.
        self.load(Args(load='dataset'))

        # graphpath = '/home/crosleyzack/school/fall18/cse515/repo/project/phaseIII/graph/graph'
        for graph in graphs:
            # load graph
            self.graph(Args(graph=f'{graph}'))

            # Create folder for this graph size.
            working_dir = join(basepath, f'graph{graph}')
            safe_mkdir(working_dir)

            print(f'Starting task 2 graph {graph}')

            
            # task 2.
            task_dir = join(working_dir, 'task2')
            safe_mkdir(task_dir)

            for k in task2:
                print(f'\tTask 2, K = {k}')
                subdir = join(task_dir, f'k{k}')
                safe_mkdir(subdir)
                self.task2(Args(k=k), path=subdir)

            print(f'Starting task 3 graph {graph}')
            
            # task 3.
            task_dir = join(working_dir, 'task3')
            safe_mkdir(task_dir)

            for k in task3:
                print(f'\tTask 3, K = {k}')
                subdir = join(task_dir, f'k{k}')
                safe_mkdir(subdir)
                self.task3(Args(k=k), path=subdir)
    

    def run_four_to_six(self):
        """
        Utility to precompute all inputs desired by the professor.
        """
        graphs = [3,10]
        task4 = [] # 5,10]
        task5_k = [5,10]
        task5_l = (1,5,10)
        task6_knn = [1,3,10]
        task6_files = ['task6sample1', 'task6sample2']

        basepath = realpath('./precomputed')
        safe_mkdir(basepath)

        # Create named tuple to simulate args.
        fields = ['k', 'alg', 'imgs', 'imageId', 'load', 'graph', 'layers', 'hashes', 'file', 'bins']
        Args = namedtuple('Arguments', fields)
        Args.__new__.__defaults__ = (None,) * len(fields)


        # call load once.
        self.load(Args(load='dataset'))

        # graphpath = '/home/crosleyzack/school/fall18/cse515/repo/project/phaseIII/graph/graph'
        for graph in graphs:
            # load graph
            self.graph(Args(graph=f'{graph}'))

            # Create folder for this graph size.
            working_dir = join(basepath, f'graph{graph}')
            safe_mkdir(working_dir)

            print(f'Starting task 4 graph {graph}')

            # task 4.
            task_dir = join(working_dir, 'task4')
            safe_mkdir(task_dir)
            task4_images = [[2976144, 3172496917, 2614355710], [27483765, 2492987710, 487287905]]
            # used list from submission sample.

            for k, images in product(task4, task4_images):
                subdir = join(task_dir, f'k{k}img{images[0]}') # include first image in dir name.
                safe_mkdir(subdir)
                self.task4(Args(k=k, imgs=images), path=subdir)

            """
            # task 5.
            task_dir = join(working_dir, 'task5')
            safe_mkdir(task_dir)
            hashes = 5 # no set value. Just generated one at random.

            for k in task5_k:
                subdir = join(task_dir, f'k{k}l{l}')
                safe_mkdir(subdir)
                self.task5(Args(k=k, layers=l, hashes=hashes))

            print(f'Starting task 6 graph {graph}')
            """

            # task 6.
            task_dir = join(working_dir, 'task6')
            safe_mkdir(task_dir)

            for k, f in product(task6_knn, task6_files):
                subdir = join(task_dir, f'knn_k{k}_file{f}')
                safe_mkdir(subdir)
                self.task6(Args(alg='knn', file=f, k=k), path=subdir)
            
            for f in task6_files:
                subdir = join(task_dir, 'ppr')
                safe_mkdir(subdir)
                self.task6(Args(alg='ppr', file=f), path=subdir)
          

            
    def create_graphs(self):
        fields = ['k', 'alg', 'imgs', 'imageId', 'load', 'graph', 'layers', 'hashes', 'file', 'bins']
        Args = namedtuple('Arguments', fields)
        Args.__new__.__defaults__ = (None,) * len(fields)

        basepath = realpath('./precomputed')
        safe_mkdir(basepath)

        graphpath = '/home/crosleyzack/school/fall18/cse515/repo/project/phaseIII/graph/graph'
        for graph in range(3,11):
            self.graph(Args(graph=f'{graph}'))

            # Create folder for this graph size.
            working_dir = join(basepath, f'graph{graph}')
            safe_mkdir(working_dir)
            self.task1(Args(k=graph), path=working_dir)





if __name__ == '__main__':
    Interface(runall=False)
