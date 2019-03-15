import numpy as np
import pandas as pd
from distance import Distance
from database import Database
from neighbor import Neighbor


class LSH():

    def get_random_vectors(self, L, k):
        random_vectors = {}
        for i in range(L):
            for j in range(k):
                random_vectors[i, j] = np.random.randint(0, 9, size=(2, 945)) * 0.1
        return random_vectors

    def get_distances(self, vectors, image_dict):
        results = {}
        result = []
        for i in vectors:
            print("Working on hash" + str(i))
            dist = {}
            projection = []
            for image in image_dict:
                a = np.array(vectors[i], dtype=np.float)
                b = a[1]
                a = a[0]
                p = np.array(image_dict[image], dtype=np.float)
                ap = p - a
                ab = b - a
                # distance = np.array(np.around((ab * ap) / Distance.E2_distance(a, b), decimals=4))
                projection.append(np.array(np.around((a + np.dot(ab, ap) / np.dot(ab, ab) * ab), decimals=3)))
                # projection.append(np.array(np.around()))
            min_proj = np.amin(projection, axis=0)
            image_keys = list(image_dict.keys())
            j = 0
            # print(min_proj)
            for x in projection:
                distance = Distance.E2_distance(min_proj, x)
                ind = image_keys[j]
                dist[ind] = distance
                j = j + 1
            result.append(dist)
            results['h' + str(i)] = result
            result = []
        # print(results)
        return results


    def get_buckets(self, results, num_buckets=4):
        buckets = {}
        for result in results:
            hash = results[result]
            for images in hash:
                # print(images)
                distances = list(images.values())
                # print(distances)
                distances = np.around(np.array(distances, dtype=np.float), decimals=3)
                dist = (np.amax(distances) - np.amin(distances)) / num_buckets
                bucket_range = range(int(np.amin(distances)), int(np.amax(distances)), int(dist))
                # print(bucket_range)
                bucket = {}
                for i in range(num_buckets):
                    bucket[str(i)] = []
                for image in images:
                    # print(images[image])
                    x = images[image]
                    j = -1
                    for b in bucket_range:
                        # print(b)
                        if x < b:
                            bucket[str(j)].append(image)
                            break
                        j = j + 1
                buckets[result] = bucket
                break
        # print(buckets)
        return buckets

    def get_index_structure(self, L, k, image_dict):

        random_vectors = self.get_random_vectors(L, k)

        # projection code
        results = self.get_distances(random_vectors, image_dict)

        # bucketing code
        num_buckets = 5
        buckets = self.get_buckets(results, num_buckets)

        return buckets, random_vectors

    def main(self, L=2, k=3, imageId=5175916261, vectors=[], t=5, database=()):
        # imageId = imageId[0]
        # imageId = int(imageId[1:-1])
        image_dict = pd.DataFrame(database.get_vis_table())
        image_dict = image_dict.T
        image_dict = image_dict.to_dict('list')

        index_structure, vectors = self.get_index_structure(L, k, image_dict)
        # print(index_structure['h(0, 0)'])

        # get all the imageIds which are in the same bucket as the given imageId
        imageId_set = set(image_dict.keys())
        # print(imageId_set)
        num_total_images = 0
        for i in range(L):
            for j in range(k):
                temp_imageId = set()
                for bucket in index_structure['h(' + str(i) + ', ' + str(j) + ')']:
                    if imageId in index_structure['h(' + str(i) + ', ' + str(j) + ')'][str(bucket)]:
                        temp_imageId = temp_imageId.intersection(set(index_structure['h(' + str(i) + ', ' + str(j) + ')'][str(bucket)]))
                    num_total_images = num_total_images + len(temp_imageId)
                imageId_set = imageId_set.union(temp_imageId)
        # print(imageId_set)

        # calculate similarity and get t nearest images

        nearest, num_comparisons = Neighbor.knn_visual_LSH(t, imageId, database, list(imageId_set))
        print("Number of non-unique images considered\t: " + str(num_total_images))
        print("Number of unique images compared\t: " + str(num_comparisons))
        for image in nearest:
            print(image)

        return [int(n.id) for n in nearest]


if __name__ == '__main__':
    lsh = LSH()
    lsh.main()
