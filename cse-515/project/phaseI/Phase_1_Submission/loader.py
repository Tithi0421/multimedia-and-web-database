# CSE 515
# Project Phase 1

from lxml import etree
import sys
from os import listdir
from os.path import isfile
from dateutil.parser import parse
from cv2 import cv2
import csv
import threading
from collections import Counter
import math
import time # for testing efficiency.
from database import Database

###############################################################
###                 LOGGER                                  ###
###############################################################

class Logger():
    def __init__(self):
        self.logs = {}
    
    def add_log(self, file, msg):
        if not file in self.logs.keys():
            self.logs[file] = []
        self.logs[file].append(msg)
    
    def to_file(self):
        with open('error_log', 'w+') as f:
            for key, value in self.logs.items():
                f.write("Errors in " + key)
                for msg in value:
                    f.write("\t" + msg)



################################################################
####                    GENERIC LOADER                      ####
################################################################

class GenericReader():

    def __init__(self):
        # self.loaded = Queue(maxsize=0)
        pass

    ##
    # Method to load a file of this type.
    # NOTE: this should append to self.loaded.
    def load_file(self, file, database):
        raise NotImplementedError
    

    ##
    #
    def load_files(self, files, database):
        for file in files:
            self.load_file(file, database)


    ##
    # Method to load a folder of files.
    def __load_folder_worker__(self, folder, database, num_threads=1):

        items = list()
        if isfile(folder):
            return items

        files = self.__get_files__(folder)
        chunk_size = math.ceil(len(files) / num_threads)

        threads = list()
        for i in range(0, len(files), chunk_size):
            subset = files[i:i + chunk_size]
            this_thread = threading.Thread(target=self.load_files, args=(subset,))
            this_thread.start()
            print("Beginning Thread " + str(this_thread.getName()))
            threads.append(this_thread)

        return threads
    

    ##
    # 
    def load_folder(self, folder, database, num_threads):

        # If we are set to spawn no threads, simply run everything through this thread.
        if num_threads is 0:
            self.load_files(self.__get_files__(folder), database)

        else:
            threads = self.__load_folder_worker__(folder, database, num_threads)
            for thread in threads:
                thread.join()
    

    ##
    # Method to load a folder of folder.
    def load_directory(self, directory, database, threads_per_folder):
        for folder in listdir(directory):
            self.load_folder(directory + '/' + folder, database, threads_per_folder)

    ##
    # 
    def __get_files__(self, folder):
        return [folder + '/' + file for file in listdir(folder) if isfile(folder + '/' + file)]


################################################################
####              TEXT DESCRIPTION LOADER                   ####
################################################################

# NOTE -  The textual descriptions can be found in the folder Dataset/desctext
#   Each of them are structured as followed.
#   ID  "Text Term That Appeared" TF IDF TF-IDF
#   The ID will be the id of an image, user, or other depending on the
#       file.

class DescriptionReader(GenericReader):

    ##
    # 
    def load_file(self, file, database):

        if not isfile(file):
            raise FileNotFoundError('Could not parse description file ' + str(file) + ' as it doesn\'t exist')
        
        with open(file) as f:
            for line in f:
                # Tokenize
                tokens = line.split(' ')
                tokens.remove('\n')
                # Find out how many tokens make up the ID
                for i, token in enumerate(tokens):
                    if '"' in token:
                        # This is our first term. Return our index to
                        #   get the id.
                        index = i
                        break
                id = ' '.join(tokens[0: index])
                for i in range(index,len(tokens), 4):
                    term, tf, idf, tfidf = tokens[i : i+4]
                    self.add_to_db(id, term, tf, idf, tfidf, database)

    ##
    #
    def add_to_db(self, id, term, tf, idf, tfidf, database):
        raise NotImplementedError


class PhotoDescriptionReader(DescriptionReader):
    def add_to_db(self, id, term, tf, idf, tfidf, database):
        # as test w/ smaller db
        database.add_photo(id, None, None, None, None, None)
        database.add_photo_desc(id, term, tf, idf, tfidf)

class UserDescriptionReader(DescriptionReader):
    def add_to_db(self, id, term, tf, idf, tfidf, database):
        database.add_user(id, None, None, None, None, None, None, None, None)
        database.add_user_desc(id, term, tf, idf, tfidf)

class LocationDescriptionReader(DescriptionReader):
    def add_to_db(self, id, term, tf, idf, tfidf, database):
        # Get the location id by the name
        location = database.get_loc_id(id)
        if not location:
            raise NameError('There was no location in the database with name ' + str(id) + ', type: ' + str(type(id)))
        database.add_loc_desc(location[0][0], term, tf, idf, tfidf)



################################################################
####                    LOCATION DATA                       ####
################################################################
##
# NOTE: There are 593 user files, each with a ton of images. This loader
#   takes a while to run.
class LocationReader(GenericReader):

    ##
    #
    def load_file(self, file, database):
        self.load_files(file, 'poiNameCorrespondences.txt', database)

    ##
    # Loads the location file using location and name_correlation files.
    def load_files(self, name_file, corr_file, database):
        name_corr = self.load_name_corr(corr_file, database)
        self.load_locations(name_file, name_corr, database)

    ##
    # Create dictionary of title > Location
    def load_locations(self, file, name_correlation, database):

        if not isfile(file):
            raise FileNotFoundError('The location data could not be loaded as the provided file was invalid: ' + str(file))
        if not type(name_correlation) is dict:
            raise TypeError('Name correlation was not of the appropriate dictionary type: ' + str(type(name_correlation)))

        tree = etree.parse(file)

        root = tree.getroot()

        for location in root:
            # Load all data from branch.
            locationid = location.find('number').text
            title = location.find('title').text
            name = name_correlation[title]
            latitude = location.find('latitude').text
            longitude = location.find('longitude').text
            wiki = location.find('wiki').text
            database.add_location(locationid, title, name, latitude, longitude, wiki)

    ##
    # Load title > name correlations
    def load_name_corr(self, file, database):

        if not isfile(file):
            raise FileNotFoundError('The name correlation dictionary could not be created as the provided parameter was not a vaild file: ' + str(file))
        
        name_correlation = {}
        with open(file) as f:
            for line in f:
                name, title = line.strip('\n').split('\t')
                name_correlation[title] = name
        
        return name_correlation

################################################################
####                     USER LOADER                        ####
################################################################

class UserReader(GenericReader):


    ##
    # load user from file.
    def load_file(self, file, database):
        
        if not isfile(file):
            raise FileNotFoundError('File from which to load user could not be found: ' + str(file))
        
        tree = etree.parse(file)
        
        # get the two main sections of the xml file
        root = tree.getroot()
        userid = root.get('user')
        credibility = root.find('credibilityDescriptors')
        photos = root.find('photos')

        # load in separate function
        self.__load_credibility__(credibility, file, userid, database)
        self.__load_photos__(photos, file, database)

    ## 
    # Load information from the credibility section of the user
    #   xml.
    def __load_credibility__(self, credibility_root, file, userid, database):
        
        visualScore = self.__get_credibility_value__(credibility_root, 'visualScore')
        faceProportion = self.__get_credibility_value__(credibility_root, 'faceProportion')
        tagSpecificity = self.__get_credibility_value__(credibility_root, 'tagSpecificity')
        locationSimilarity = self.__get_credibility_value__(credibility_root, 'locationSimilarity')
        photoCount = self.__get_credibility_value__(credibility_root, 'locationSimilarity')
        uniqueTags = self.__get_credibility_value__(credibility_root, 'locationSimilarity')
        uploadFrequency = self.__get_credibility_value__(credibility_root, 'locationSimilarity')
        bulkProportion = self.__get_credibility_value__(credibility_root, 'bulkProportion')

        database.add_user(userid, visualScore, faceProportion, tagSpecificity, locationSimilarity,
                            photoCount, uniqueTags, uploadFrequency, bulkProportion)


    def __get_credibility_value__(self, credibility_root, field):

        try:
            returnval = credibility_root.find(field).text
            returnval = float(returnval)
        except Exception as e:
            returnval = None
            #raise IOError("An exception occured loading value " + str(field) + " as float/real.")
        return returnval

    ##
    # Load the photos from the user.
    def __load_photos__(self, photos_root, file, database):

        print("Loading " + str(len(photos_root.getchildren())) + " photos...")
        for photo in photos_root:
            id=photo.get('id')
            userid=photo.get('userid')
            date_taken=photo.get('date_taken') 
            views=int(photo.get('views'))
            title=photo.get('title')
            url_b=photo.get('url_b')
            tags=photo.get('tags').split(' ')
            database.add_photo(id, userid, date_taken,views, title, url_b)
            for tag in tags:
                database.add_tag(id, tag)


################################################################
####                     IMAGE LOADER                       ####
################################################################
# TODO - Update this to read into database.
# def get_photo_visual_desc(self, photoid, model):
# def add_visual_descriptor(self, photoid, locationid, value, model):
class VisualDescriptionReader(GenericReader):

    ##
    # Load the image into a numpy array.
    def load_model(self, folder, locationid, title, model, database):
        file = self.get_file_name(title, model)
        with open(folder + '/' + file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                photoid = row[0]
                database.add_photo_location(photoid, locationid)
                visual_descriptors = row[1:]
                for visual_descriptor in visual_descriptors:
                    database.add_visual_descriptor(photoid, visual_descriptor, model)


    def get_file_name(self, location_title, model):
        return location_title + " " + model + ".csv"


    def load_csvs(self, folder, models, database):
        # get all location titles
        titles = [title[0] for title in database.get_location_titles()]
        for title in titles:
            location = database.get_location_by_title(title)
            locationid = location[0]
            for model in models:
                self.load_model(folder, locationid, title, model, database)


################################################################
####                    Load All Data                       ####
################################################################
class Loader():
        
    def load_timed(self, folder, num_threads=0):
        start_time = time.time()
        self.load_database(folder, num_threads)
        return start_time - time.time()

    ##
    #
    def load_database(self, folder, database, num_threads=0):
        # UserReader().load_folder(folder + '/desccred', database, num_threads)
        # database.commit()
        # print("Users and Photos Loaded...")
        LocationReader().load_files(folder + '/devset_topics.xml',
                folder + '/poiNameCorrespondences.txt', database)
        database.commit()
        print("Locations Loaded...")
        LocationDescriptionReader().load_file(folder + '/desctxt/devset_textTermsPerPOI.txt', database)
        database.commit()
        print("Location Descriptions Loaded...")
        PhotoDescriptionReader().load_file(folder + '/desctxt/devset_textTermsPerImage.txt', database)
        database.commit()
        print("Photo Descriptions Loaded...")
        UserDescriptionReader().load_file(folder + '/desctxt/devset_textTermsPerUser.txt', database)
        database.commit()
        print("User Descriptions Loaded...")
        models = ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
        VisualDescriptionReader().load_csvs(folder + '/descvis/img/', models, database)
        database.commit()
        print("Visual Descriptors Loaded...")
        return database
        

    ##
    # Construct 
    def make_database(self, filepath, num_threads=0):
        database = Database(filepath)
        database.make_tables()
        return database

    ##
    # 
    def test_database(self, database):
        # Location Validation
        locations = database.get_locations()
        print("Found " + str(len(locations)) + " locations.")
        for i, location in enumerate(locations):
            location = location[0]
            loc_desc = database.get_loc_desc(location)
            print(str(i) + ": Location " + str(location) + "\t" + str(loc_desc))
        # User Validation
        users = database.get_user_list()
        print("Found " + str(len(users)) + " users...")
        for i, user in enumerate(users):
            user = user[0]
            user_desc = database.get_user_desc(user)
            print(str(i) + ": User " + str(user) + "\t" + str(user_desc))
            user_photos = database.get_user_photos(user)
            print("Found " + str(len(user_photos)) + " for User " + user)
            for j, photo in enumerate(user_photos):
                photo = photo[0]
                photo_desc = database.get_photo_desc(photo)
                if photo_desc:
                    print("\t" + str(j) + ": Photo " + str(photo) + "\t" + str(photo_desc))


    ##
    # Get the most efficient number of threads to use for this computer.
    # TODO Should this method be removed?
    def simple_climb(self, folder, start = 7):

        best_time = self.load_timed(folder, start)
        print("Initial Time Set: " + str(best_time) + " for " + str(start) + " threads.")
        valid_ops = [lambda a: a +1, lambda a: a-1]

        this_op = valid_ops[0]
        current = start
        best_threads = start
        tried_threads = Counter()
        # While the most common state is less than three.
        while tried_threads.most_common(1)[1] <= 3:
            current = this_op(current)
            new_time = self.load_timed(folder, current)

            if new_time < best_time:
                best_time = new_time
                best_threads = start
                print("New Best Time Set: " + str(best_time) + " for " + str(best_threads) + " threads.")
            elif new_time > best_time:
                this_op = [op for op in valid_ops if not op is this_op][0]
            
            tried_threads.update(current)

        return best_threads, best_time