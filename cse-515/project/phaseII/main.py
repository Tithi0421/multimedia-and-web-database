from neighbor import Neighbor
from decompose import Decompose
from loader import Loader
from database import Database
from os.path import isfile, abspath, isdir
from distance import Scoring
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD as SVD
import numpy as np
from pandas import DataFrame as df
import xml.etree.ElementTree as ET
from operator import itemgetter
from util import timed
from collections import defaultdict
import collections
from tensorly.decomposition import parafac
import tensorly as tl
from sklearn.cluster import KMeans
import _pickle as pk



# CHANGES SINCE SUBMISSION
#   ADDED MinMaxScalar to all LDA algs.
#   Changed tensor algorithm to account for change in location count.
#   Changed load algorithm to handle unexpected newlines present in the demo data.


class Interface():

    def __init__(self, demo_load=False):
        self.__database__ = None
        self.__valid_types__ = ['photo', 'user', 'poi']
        self.__vis_models__ = ['CM', 'CM3x3', 'CN', 'CN3x3',
                'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
        if not demo_load:
            self.__io__()
        else:
            self.run_1_2()


    def __latent_semantic_string__(self, table, save=True):
        """
        Pretty prints out the latent semantics.
        :param DataFrame table: The latent semantic vectors. Columns are individual vectors, rows are features.
        """
        table = table.T

        if save:
            table.to_csv('latent_semantics.csv') # For easy reference to teacher
        for i, col_id in enumerate(table):
            column = table[col_id]
            column = column.sort_values(ascending=False)
            to_return += '-' * 50
            to_return += "\nLATENT SEMANTIC " + str(i)
            to_return += '-' * 50
            to_return += column.__repr__()
        return to_return



    def __io__(self):
        print("Welcome to the CSE 515 data software. Please enter a command.\
              \nEnter \"help\" for a list of commands.")
        while True:
            user_input = input("\nEnter a Command:$ ")
            user_input = user_input.split(' ')
            command = user_input[0]
            command.replace('_', '')
            # Get method with this name - makes it easy to create new interface methods
            #   without having to edit this method.
            try:
                method = getattr(self, command)
            except AttributeError:
                print('The command specified was not a valid command.\n')

            try:
                method(*user_input[1:])
            except Exception as e:
                print('Something went wrong while executing ' + str(method.__name__))
                print(str(e))



    def help(self, *args):
        """
        Command:\   thelp
        Description:\tPrints the interface information about the program.
        """

        print("The following are valid commands to the program.")
        for item in dir(self):
            if not item.startswith('__'):
                method = getattr(self, item)
                print(method.__doc__)



    # takes a single argument - file to load.
    def load(self, *args):
        """
        Command:\tload <filepath>
        Description:\tLoads the database at that file path. If the file does not exist, will prompt to create a new database using the folder of a users choice.
        Arguments:
        \t<filepath> - A valid file path in the system.
        """

        if len(args) < 1:
            print("[ERROR] Not enough args given to load parameter.")
            print("\targs = " + str(args))
            return
        if len(args) >= 2:
            print("[ERROR] Too many args given to load parameter.")
            print("\targs = " + str(args))
            return

        folder = abspath(args[0])

        if not isdir(folder):
            print("[ERROR] The provided path was not a folder, and therefore not a valid data directory.")
            return

        self.__database__ = Loader.make_database(folder)
        print("Database loaded successfully.")



    # For calculating the latent semantics of the data (Task I)
    def task1(self, *args):
        """
        Command:\ttask1 <term space> <k> <method>
        Description:\tCalculates the top k latent semantics in the term space given a \
        methodology. Latent semantics will be presented in term-weight pairs.
        Arguments:
        \tTerm Space - What text description type to pull. (user, photo, poi)
        \tK - Number of latent semantics to return.
        \tMethod - The decomposition method to use. (PCA, SVD, LDA)
        """

        if len(args) < 3:
            print("[ERROR] Not enough args were provided. Expected 3 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        if len(args) > 3:
            print("[ERROR] Too many arguments were provided. Expected 3 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        if not self.__database__:
            print("[ERROR] The Database must be loaded before this can be run.")
            print("\tCommand: load <filepath>")
            return

        try:
            term_space = args[0]
            k = int(args[1])
            method = args[2]
        except:
            print("[ERROR] One or more arguments could not be parsed: " + str(args))

        _, latent_semantics = Decompose.decompose_text(term_space, k, method.lower(), self.__database__)
        string = self.__latent_semantic_string__(latent_semantics)
        return string



    # TODO - this should be implemented so it can follow Task 1, not be restarted
    #   from scratch.
    def task2(self, *args):
        """
        Command:\ttask2 <term space> <k> <method> <j> <id>
        Description:\tIdentifies the top k latent semantics uses an id in the term \
        space to identify the most related j ids.
        Arguments:
        \tTerm Space - What text description type to pull. (user, photo, poi).
        \tK - Number of latent semantics to return.
        \tMethod - The decomposition method to use. (PCA, SVD, LDA).
        \tJ - Number of nearest terms to find.
        \tID - The id to find nearest neighbors to using the latent semantics.
        """
        if len(args) < 5:
            print("[ERROR] Not enough args were provided. Expected 5 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        if len(args) > 5:
            print("[ERROR] Too many arguments were provided. Expected 5 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        if not self.__database__:
            print("[ERROR] The Database must be loaded before this can be run.")
            print("\tCommand: load <filepath>")
            return

        try:
            term_space = args[0]
            k = int(args[1])
            method = args[2]
            j = int(args[3])
            anid = args[4]
            if term_space == 'photo' or term_space == 'poi':
                anid = int(anid)
        except:
            print("[ERROR] One or more arguments could not be parsed: " + str(args))

        reduced_table, latent_semantics = Decompose.decompose_text(term_space, k, method.lower(), self.__database__)
        ls_str = self.__latent_semantic_string__(latent_semantics)
        print(ls_str)
        vector = reduced_table.loc[anid]
        neighbors = Neighbor.knn(j, vector, reduced_table)
        neighbors_str = ""
        neighbors_str += "NEIGHBORS to " + str(anid)
        for i, neighbor in enumerate(neighbors):
            neighbors_str += f"{i}: ID = {neighbor.id}, DIST = {neighbor.dist}"
        print(neighbors_str)
        return ls_str + '\n' + neighbors_str



    # TODO - This should also be reimplemented so the query for j nearest can be done
    #   after the initial query, rather than from scratch.
    def task3(self, *args):
        """
        Command:\ttask3 <vis model> <k> <method> <j> <id>
        Description:\tIdentifies k latent semantics using model and methodology. Given an \
        id, finds the top j most related items to that id.
        Arguments:
        \tVisual Space - What visual description model to pull. ('CM', 'CM3x3', 'CN', \
        'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3')
        \tK - Number of latent semantics to return.
        \tMethod - The decomposition method to use. (PCA, SVD, LDA).
        \tJ - Number of nearest terms to find.
        \tID - The id of the image to find nearest neighbors to using the latent semantics.
        """
        if len(args) < 5:
            print("[ERROR] Not enough args were provided. Expected 5 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        if len(args) > 5:
            print("[ERROR] Too many arguments were provided. Expected 5 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        if not self.__database__:
            print("[ERROR] The Database must be loaded before this can be run.")
            print("\tCommand: load <filepath>")
            return

        try:
            vis_model = args[0]
            k = int(args[1])
            method = args[2]
            j = int(args[3])
            anid = int(args[4])
        except:
            print("[ERROR] One or more arguments could not be parsed: " + str(args))

        # Get latent semantics.
        matrix = Decompose.decompose_vis(vis_model, k, method, self.__database__)

        # Get nearest images from latent semantics.
        vector = matrix.loc[anid]
        vector1 = []
        for i in range(k):
            vector1.append(vector[i])
        nearest_img = "5 nearest images to " + str(args[3]) + " are:\n"
        nearest = Neighbor.knn_dot(5, vector1, matrix)
        nearest_img += "IMAGE ID\t\tSCORE\n"
        for values in nearest:
            nearest_img += str(values[1]) + "\t\t" + str(values[0]) + '\n'
        print(nearest_img)

        # Get nearest locations from latent semantics.
        nearest_loc = "5 nearest locations to " + str(args[3]) + " are:"
        nearest = Neighbor.knn_loc(5, vector1, vis_model, k, method, self.__database__)
        nearest_loc += "LOCATION ID\t\tSCORE"
        for values in nearest:
            nearest_loc += str(values[1]) + "\t\t\t" + str(values[0]) + "\n"
        print(nearest_loc)
        return nearest_img + '\n\n' + nearest_loc


    def task4(self, *args):
        """
        Command:\ttask4 <locationid> <vis model> <k> <method>
        Description:\tIdentifies the top k latent semantics for the given location \
        and lists the five most related locations.
        Arguments:
        \tLocation Id - A valid location id. (1 - 30)
        \tVisual Space - What visual description model to pull. ('CM', 'CM3x3', 'CN', \
        'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3').
        \tK - Number of latent semantics to identify.
        \tMethod - The decomposition method to use. (PCA, SVD, LDA).
        """
        if len(args) < 4:
            print("[ERROR] Not enough args were provided. Expected 4 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        if len(args) > 4:
            print("[ERROR] Too many arguments were provided. Expected 4 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        if not self.__database__:
            print("[ERROR] The Database must be loaded before this can be run.")
            print("\tCommand: load <filepath>")
            return

        try:
            locationid = int(args[0])
            vis_model = args[1]
            k = int(args[2])
            method = args[3]
        except:
            print("[ERROR] One or more arguments could not be parsed: " + str(args))

        # Get latent semantics.
        matrix = Decompose.decompose_loc_vis(vis_model, k, method, locationid, self.__database__)

        # Get nearest 5 locations from latent semantics. Neighbor.knn may be useful for you.

        nearest = Neighbor.knn_vd(5, matrix, vis_model, k, method, locationid, self.__database__)
        nearest_loc = "5 nearest locations to " + str(args[0]) + " are:\n"
        nearest_loc += "LOCATION ID\t\tSCORE\n"
        for values in nearest:
            nearest_loc += str(values[1]) + "\t\t\t" + str(values[0]) + '\n'
        print(nearest_loc)
        return nearest_loc


    def task5(self, *args):
        """
        Command:\ttask5 <locationid> <k> <method>
        Description:\tIdentifies the top k latent semantics for the location id \
        across all models and identifies the top 5 related locations to the \
        latent semantics.
        Arguments:
        \tLocation Id - A valid location id. (1 - 30)
        \tK - Number of latent semantics to identify.
        \tMethod - The decomposition method to use. (PCA, SVD, LDA).
        """
        if len(args) < 3:
            print("[ERROR] Not enough args were provided. Expected 3 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        if len(args) > 3:
            print("[ERROR] Too many arguments were provided. Expected 3 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        if not self.__database__:
            print("[ERROR] The Database must be loaded before this can be run.")
            print("\tCommand: load <filepath>")
            return

        try:
            locationid = int(args[0])
            k = int(args[1])
            method = args[2]
        except:
            print("[ERROR] One or more arguments could not be parsed: " + str(args))

        # WRITE CODE IN THESE FUNCTIONS #####################################################
        reduced_table, latent_semantics = Decompose.decompose_loc(k, method, locationid, self.__database__) # Get latent semantics.
        # reduced_table.to_csv('Image-semantic_task5.csv')
        self.__latent_semantic_string__(latent_semantics)
        reducedLocations = dict()  # A dictionary that has locationid as the key and the reduced tables as the values.
        similarityScore = dict()
        for i in range(1,36):
            if i == locationid:
                reducedLocations[i] = reduced_table
            else:
                rtable, _ = Decompose.decompose_loc(k, method, i, self.__database__)
                reducedLocations[i] = rtable
            sim = cosine_similarity(reduced_table,reducedLocations[i])
            # The dot_similarity (matrix, matrix) returns a 2D array that has ids of one location as rows, and another location's id as columns.
            # The values are the similarity between each id (Just like a similarity matrix).

            similarityScore[i] = Scoring.score_matrix(sim)
        orderedSimilarityList = sorted(similarityScore, key=similarityScore.__getitem__) # returns a list containing the Locations ids in the increasing order of the similarity
        orderedSimilarityList.reverse()  # Reverse the list to get it in the decreasing order
        print (similarityScore)
        print("The most related 5 locations are:")
        for i in range(5):
            print("Id: ", orderedSimilarityList[i], "Score: ", similarityScore[orderedSimilarityList[i]])


    def task6(self, *args):
        """
        Command:\ttask6 <k>
        Description:\tCreates a location - location similarity matrix, performs SVD, \
        and reports the top k latent semantics.
        Arguments:
        \tK - Number of latent semantics to identify.
        """
        if len(args) < 1:
            print("[ERROR] Not enough args were provided. Expected 1 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        if len(args) > 1:
            print("[ERROR] Too many arguments were provided. Expected 1 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        if not self.__database__:
            print("[ERROR] The Database must be loaded before this can be run.")
            print("\tCommand: load <filepath>")
            return

        try:
            k = int(args[0])
        except:
            print("[ERROR] One or more arguments could not be parsed: " + str(args))

        # First getting all the location tables having all the visual discriptors.

        # IMP NOTE: Remember to change all the static values after the test data arrives.
        all_location_tables = dict()
        for id in range(1,36):
            all_location_tables[id] = Database.get_vis_table(self.__database__,locationid=id)
        # Start finding the similarity between each pair of locations and store the results into a dictionary.
        if isfile('./all_similarities.npy'):
            all_similarities = np.load('all_similarities.npy').item()
        else:
            all_similarities = dict()
            for i in range(1,36):
                for j in range(i,36):
                    cos_sim = cosine_similarity(all_location_tables[i], all_location_tables[j])
                    all_similarities[(i, j)] = Scoring.score_matrix(cos_sim)
                    all_similarities[(j, i)] = all_similarities[(i, j)]
            np.save('all_similarities.npy', all_similarities)

        similarity_matrix = df(index=range(1,36),columns=range(1,36))
        for i in range(1,36):
            sim_list = list()
            for j in range(1,36):
                sim_list.append(all_similarities[(i, j)])
            similarity_matrix.loc[i] = sim_list
        similarity_matrix.to_csv('Task6_SimilarityMatrix.csv')
        print("Location-Location Similarity matrix created...")

        reduced = SVD(n_components=k)
        # reduced_table = reduced.fit_transform(similarity_matrix)
        reduced.fit(similarity_matrix)
        VTranspose = df(data=reduced.components_, index=range(1,k+1),columns=range(1,36))
        VTranspose.to_csv('task6transposetable.csv')

        filename = 'devset_topics.xml'
        tree = ET.parse(filename)
        root = tree.getroot()
        location_name = dict()
        location_dict = dict()
        i = 0

        while i < len(root):
            v1 = int(root[i][0].text)
            v2 = root[i][1].text
            location_name[v1] = v2
            i += 1

        print("Top k latent semantics in form of their location-weight pair are:")
        for _, row in VTranspose.iterrows():
            for j in range(1, 36):
                loc = location_name[j]
                location_dict[loc] = row[j]
            sorted_location_dict = sorted(location_dict.items(), key=itemgetter(1), reverse=True)

            # print("latent semantic " + str(index) + " : " + str(sorted_location_dict))
            print(f"LATENT SEMANTIC {i}")
            for key, value in sorted_location_dict:
                print(f"{key} : {value}")


    def task7(self, *args):
        """
        Command:\ttask7 <k>
        Description:\tCreates a user-image-location tensor, performs rank k CP \
        decomposition of this tensor, and creates k non-overalapping groups of \
        users, images, and locations based on the discovered latent semantics.
        Arguments:
        \tK - Number of non-overlapping groups to create.
        """
        if len(args) < 1:
            print("[ERROR] Not enough args were provided. Expected 1 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        if len(args) > 1:
            print("[ERROR] Too many arguments were provided. Expected 1 but got " + str(len(args)))
            print("\targs = " + str(args))
            return
        #if not self.__database__:
        #    print("[ERROR] The Database must be loaded before this can be run.")
        #    print("\tCommand: load <filepath>")
        #    return

        try:
            k = int(args[0])
        except:
            print("[ERROR] One or more arguments could not be parsed: " + str(args))

        # First getting the terms of all the user ids, image ids and location ids into dictionaries.
        # A very naive code implementation.
        # Add these files to the PhaseII folder, or mention the path to the files.
        userFile = 'devset_textTermsPerUser.txt'
        imageFile = 'devset_textTermsPerImage.txt'
        locationFile = 'devset_textTermsPerPOI.wFolderNames.txt'

        userDictionary = dict()  # contains userid as the key and terms as its terms as the value.
        imageDictionary = dict()  # contains imageid as the key and terms as its terms as the value.
        locationDictionary = dict()  # contains locationid as the key and terms as its terms as the value.

        uflag = 1
        utermlist = list()
        ucurrent_id = ""
        with open(userFile, encoding='latin-1') as f:
            for line in f:
                for word in line.split():
                    if uflag == 1:
                        if word.startswith('"'):
                            utermlist.append(word)
                            userDictionary[ucurrent_id] = utermlist
                            uflag = 2
                            continue

                        ucurrent_id = word
                        utermlist = []
                        continue
                    if uflag == 2:
                        uflag = 3
                        continue
                    if uflag == 3:
                        uflag = 4
                        continue
                    if uflag == 4:
                        uflag = 1
                        continue

        iflag = 1
        itermlist = list()
        icurrent_id = ""
        with open(imageFile, encoding='latin-1') as f:
            for line in f:
                for word in line.split():
                    if iflag == 1:
                        if not word.startswith('"'):
                            icurrent_id = word
                            itermlist = []
                            continue
                        itermlist.append(word)
                        imageDictionary[icurrent_id] = itermlist
                        iflag = 2
                        continue
                    if iflag == 2:
                        iflag = 3
                        continue
                    if iflag == 3:
                        iflag = 4
                        continue
                    if iflag == 4:
                        iflag = 1
                        continue
        # imageDictionary[icurrent_id] = itermlist
        # print(userDictionary)
        # print(len(imageDictionary.keys()))
        # print(len(userDictionary.keys()))

        locNames = dict()
        locidToName = dict()
        tree = ET.parse('./devset_topics.xml')
        root = tree.getroot()
        for elem1 in root:
            locNames[elem1[1].text] = int(elem1[0].text)  # Key: Location Name, Value: Location Ids
            locidToName[int(elem1[0].text)] = elem1[1].text  # Key: Location Ids, Value: Location Name

        lflag = 1
        ltermlist = list()
        lcurrent_id = ""
        with open(locationFile, encoding='utf8') as f:
            for line in f:
                for word in line.split():
                    if lflag == 1:
                        if word.startswith('"'):
                            ltermlist.append(word)
                            locationDictionary[lcurrent_id] = ltermlist
                            lflag = 2
                            continue
                        lcurrent_id = locNames[word]
                        ltermlist = []
                        lflag = 5
                        continue
                    elif lflag == 2:
                        lflag = 3
                        continue
                    elif lflag == 3:
                        lflag = 4
                        continue
                    elif lflag == 4:
                        lflag = 1
                        continue
                    elif lflag == 5:
                        if word.startswith('"'):
                            ltermlist.append(word)
                            locationDictionary[lcurrent_id] = ltermlist
                            lflag = 2
                            continue
                        continue

        locationDictionary[lcurrent_id] = ltermlist

        # At this point the dictionaries have a list of terms for each ids

        # Constructing the Tensor.
        useridToNumber = dict()  # takes a number from 1 to 530 as key, gives out userid. Used for mapping.
        imageidToNumber = dict()  # takes a number from 1 to 8912 as key, gives out an imageid. Used for mapping.
        ordered_userDictionary = collections.OrderedDict(sorted(userDictionary.items()))
        ordered_imageDictionary = collections.OrderedDict(sorted(imageDictionary.items()))

        #  Cleaning the data to remove the error data.
        i = 1
        for key, value in ordered_userDictionary.items():
            if '@' in key:
                useridToNumber[i] = key
                i += 1
        i = 1
        for key, value in ordered_imageDictionary.items():
            if float(key) > 5000:
                imageidToNumber[i] = key
                i += 1
        len1 = len(useridToNumber)
        len2 = len(imageidToNumber)
        len3 = len(locationDictionary)

        if isfile('./tensor.pkl'):
            tensor1 = pk.load(open("./tensor.pkl","rb"))
        else:
            tensor1 = np.zeros(shape=(len1,len2,len3), dtype=np.int16)
            for i in range(1,len(useridToNumber.keys())+1):
                for j in range(1,len(imageidToNumber.keys())+1):
                    for m in range(1,len(locationDictionary.keys())+1):
                        print(i,j,m)
                        tensor1[i-1][j-1][m-1] = len(list(set(userDictionary[useridToNumber[i]]) & set(imageDictionary[imageidToNumber[j]]) & set(locationDictionary[m])))

            pk.dump(tensor1,open("tensor.pkl", "wb"))

        tensor2 = tl.tensor(tensor1)
        cp_rank = k
        factors = parafac(tensor2,rank=cp_rank)

        factor_matrix1 = np.matrix(factors[0])
        factor_matrix2 = np.matrix(factors[1])
        factor_matrix3 = np.matrix(factors[2])

        user_matrix = df(factor_matrix1,index=[useridToNumber[i] for i in range(1, len(useridToNumber)+1)],columns = range(1,k+1))
        user_matrix.to_csv('task7U.csv')
        image_matrix = df(factor_matrix2,index=[imageidToNumber[i] for i in range(1, len(imageidToNumber)+1)], columns=range(1,k+1))
        image_matrix.to_csv('task7I.csv')
        location_matrix = df(factor_matrix3, index=locNames.keys(), columns=range(1, k + 1))
        location_matrix.to_csv('task7L.csv')

        # data1 = pd.read_csv('task7U.csv', row.names=1)
        data1 = df.from_csv('task7U.csv',header=0,index_col=0)
        data1.head()

        kmeans1 = KMeans(n_clusters=k)
        kmeans1.fit(data1)
        X = np.array(data1)
        lables1 = kmeans1.labels_
        userDefDict = defaultdict(list)
        for i in range(1, len(useridToNumber)+1):
            templist = userDefDict[lables1[i-1]]
            templist.append(useridToNumber[i])
            userDefDict[lables1[i-1]] = templist
        # listzip1 = [(useridToNumber[i],"group " + str(lables1[i-1])) for i in range(1,len(useridToNumber)+1)]
        fout = "task7UserGroups.txt"
        fo = open(fout,"w")
        for key, v in userDefDict.items():
            fo.write(str(key) + ':' + str(v) + '\n\n')
        fo.close()

        data2 = df.from_csv('task7I.csv', header=0, index_col=0)
        data2.head()

        kmeans2 = KMeans(n_clusters=k,max_iter=80)
        kmeans2.fit(data2)
        Y = np.array(data2)
        lables2 = kmeans2.labels_
        # listzip2 = [(imageidToNumber[i], "group " + str(lables2[i - 1])) for i in range(1, len(imageidToNumber) + 1)]
        imageDefDict = defaultdict(list)
        for i in range(1, len(imageidToNumber) + 1):
            templist = imageDefDict[lables2[i - 1]]
            templist.append(imageidToNumber[i])
            imageDefDict[lables2[i - 1]] = templist
        # listzip1 = [(useridToNumber[i],"group " + str(lables1[i-1])) for i in range(1,len(useridToNumber)+1)]
        fout = "task7ImageGroups.txt"
        fo = open(fout, "w")
        for key, v in imageDefDict.items():
            fo.write(str(key) + ':' + str(v) + '\n\n')
        fo.close()


        data3 = df.from_csv('task7L.csv', header=0, index_col=0)
        data3.head()

        kmeans3 = KMeans(n_clusters=k)
        kmeans3.fit(data3)
        Z = np.array(data3)
        lables3 = kmeans3.labels_
        # listzip3 = [(locidToName[i], "group " + str(lables3[i - 1])) for i in range(1, len(locidToName) + 1)]
        locationDefDict = defaultdict(list)
        locidToName = defaultdict()
        for i in range(1, len(locidToName) + 1):
            templist = locationDefDict[lables3[i - 1]]
            templist.append(locidToName[i])
            locationDefDict[lables3[i - 1]] = templist
        # listzip1 = [(useridToNumber[i],"group " + str(lables1[i-1])) for i in range(1,len(useridToNumber)+1)]
        fout = "task7LocationGroups.txt"
        fo = open(fout, "w")
        for key, v in locationDefDict.items():
            fo.write(str(key) + ':' + str(v) + '\n\n')
        fo.close()


    def quit(self, *args):
        """
        Command:\tquit
        Description:\tExits the program and performs necessary cleanup.
        """
        exit(1)


    def run_1_2(self):
        subdir = 'demo/'
        self.load('/home/crosleyzack/school/fall18/cse515/repo/project/phaseII/phase2_testdata/devset')
        for space, current_id in zip(['photo', 'user', 'poi'], [9636162061, '15394372@N00', 10]):
            for method in ['pca', 'lda', 'svd']:
                k = 8
                string = f"{space}_{k}_{method}"
                print(f"Loading {string}")
                lsm = self.task1(space, k, method)
                nearest = self.task2(space,k, method, k, current_id)
                task1_file_name = subdir + f"t1_{string}.txt"
                task2_file_name = subdir + f"t2_{string}_{current_id}.txt"
                with open(task1_file_name, 'w+') as f:
                    f.write(lsm)
                with open(task2_file_name, 'w+') as f:
                    f.write(nearest)

                for visual_model in ['CM']:#, 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']:
                    out_3 = self.task3(visual_model, k, method, 5, current_id)
                    file_3 = subdir + f"t3_{visual_model}_{k}_{method}_{current_id}.txt"
                    with open(file_3, 'w+') as f:
                        f.write(out_3)
                    # <locationid> <vis model> <k> <method>
                    for location in range(1,36):
                        task4_filename = subdir + f"t4_{location}_{visual_model}_{k}_{method}.txt"
                        if not isfile(task4_filename):
                            out_4 = self.task4(location, visual_model, k, method)
                            with open(task4_filename, 'w+') as f:
                                f.write(out_4)


if __name__ == '__main__':
    Interface()
