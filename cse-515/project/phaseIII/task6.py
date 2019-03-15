import operator
from distance import Similarity
from distance import Distance
import pandas as pd
from util import timed, show_images
from scipy.sparse import csc_matrix
import numpy as np


class KNN:

    def get_neighbors(self, labelled_set, labels, imageInstance, k):
        distances = []
        j = 0
        for x in range(len(labelled_set)):
            # dist = Similarity.cos_similarity(imageInstance, labelled_set[x])
            dist = Distance.E2_distance(imageInstance, labelled_set[x])
            distances.append((labelled_set[x], labels[j], dist))
            j += 1
        distances.sort(key=operator.itemgetter(2), reverse=False)
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][1])
        # print("neighbors:" + str(neighbors))
        return neighbors

    def get_response(self, neighbors):
        class_votes = {}
        for x in range(len(neighbors)):
            response = neighbors[x]
            if response in class_votes:
                class_votes[response] += 1
            else:
                class_votes[response] = 1
        sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]

    def get_labelled_set(self, imageIDs, image_dict):
        labelled_set = []
        for imageID in imageIDs:
            labelled_set.append(image_dict[int(imageID)])
        return labelled_set

    @timed
    def knn_algorithm(self, imageIds, labels, k, database):
        image_dict = pd.DataFrame(database.get_vis_table())
        image_dict = image_dict.T
        image_dict = image_dict.to_dict('list')

        labelled = {}
        for j, imageId in enumerate(imageIds):
            labelled[imageId] = labels[j]
            # labelled.append({imageId: labels[j]})
        # print(labelled)

        labelled_set = self.get_labelled_set(imageIds, image_dict)
        print("Working")
        for v, image in enumerate(image_dict):
            image = int(image)
            if image not in labelled:
                neighbors = self.get_neighbors(labelled_set, labels, image_dict[image], k)
                result = self.get_response(neighbors)
                # print(str(v) + " labelled as :" + str(result))
                labels.append(result)
                imageIds.append(image)
                labelled[image] = result
                # labelled.append({image: result})
                # labelled_set.append(image_dict[image])

        return labelled

    def main(self):
        k = 3
        imageIDs = ['3298433827', '299114458', '948633075', '4815295122', '5898734700', '4027646409', '1806444675',
                    '4501766904', '6669397377', '3630226176', '3630226176', '3779303606', '4017014699']
        labels = ['fort', 'sculpture', 'sculpture', 'sculpture', 'sculpture', 'fort', 'fort', 'fort', 'sculpture',
                  'sculpture', 'sculpture', 'sculpture', 'sculpture']
        '''
        j = 0
        for i in args:
            if j % 2 == 0:
                imageIDs.append([i])
            else:
                labels.append([i])
            j = j + 1
        '''
        result = self.knn_algorithm(imageIDs, labels, k, database=())
        print("result: " + str(result))
        
        
class PPR:
    
    @timed
    def ppr_algorithm(self, imageIDs, labels, indexes, G, images):

        labelled = {}
        for j, imageId in enumerate(imageIDs):
            labelled[imageId] = labels[j]

        num_labels = len(set(labels))
        set_labels = frozenset(labels)
        ind_labels = {}
        for itr, l in enumerate(set_labels):
            ind_labels[l] = itr
        # print("ind_label" + str(ind_labels))

        for x in imageIDs:
            indexes.append(images.index(x))
        n = G.shape[0]
        s = 0.86
        maxerr = 0.1

        # transform G into markov matrix A
        A = csc_matrix(G, dtype=np.float)
        rsums = np.array(A.sum(1))[:, 0]
        ri, ci = A.nonzero()
        A.data /= rsums[ri]


        temp = 1 / num_labels
        r_labels = np.array([temp] * num_labels * n)
        r_labels = r_labels.reshape(n, num_labels)

        # account for seed teleportation
        a = 0
        Ei = np.zeros(n)
        for ii in indexes:
            if a > (len(imageIDs) - 1):
                break
            Ei[ii] = (1 / len(imageIDs))
            this_image = imageIDs[a]
            current_label = labelled[this_image]
            ind = int(ind_labels[current_label])
            r_labels[ii] = np.zeros(num_labels)
            r_labels[ii][ind] = 1
            a += 1

        # Compute pagerank r until we converge
        ro, r = np.zeros(n), np.ones(n)

        # while np.sum(np.abs(r - ro)) > maxerr:
        for out_itr in range(10):

            if np.sum(np.abs(r - ro)) <= maxerr:
                break

            print(f'Working: {out_itr}')

            ro = r.copy()
            # calculate each pagerank at a time
            for i in range(0, n):
                # in-links of state i
                # print("Working: " + str(out_itr) + " : " + str(i))
                Ai = np.array(A[:, i].todense())[:, 0]
                max_ind = int(np.argmax(Ai))
                r_labels[i, :] = np.sum([r_labels[i, :], r_labels[max_ind, :]], axis=0)
                r[i] = ro.dot(Ai * s + Ei * (1 - s))

        itr = 0
        labelled = {}
        for image in images:
            label_id = np.argmax(r_labels[itr])
            label = list(ind_labels.keys())[list(ind_labels.values()).index(label_id)]
            labelled[image] = label
            itr += 1

        return labelled


if __name__ == '__main__':
    knn = KNN()
    knn.main()
