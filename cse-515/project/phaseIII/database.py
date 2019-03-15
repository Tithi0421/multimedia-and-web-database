#! /bin/usr/python3.6

import pandas as pd
from os import listdir, mkdir
from os.path import basename, join, isfile, splitext, split, isdir, abspath
from csv import reader
from util import timed
from collections import defaultdict
import pickle

class Database():

    ##
    # Store the various pandas dataframes for local use.
    def __init__(self, source=None):
        self.source = source # indicates the dataset file location. 
        self.vis_descriptors = {}
        self.vis = None
        # self.txt_descriptors = {}
        self.locations = None
        self.vis_models = ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
        self.loc_map = {}
        

    ##################################################################
    ###                         ADDING DATA                        ###
    ##################################################################
    

    ##
    # Stores a location pandas dataframe.
    #
    # Data is a dictionary to be loaded into a frame with organization:
    #   <'locationid', title', 'name', 'latitude', 'longitude', 'wiki'>
    #   in each row, with 30 rows for the 30 locations. Index is the location
    #   id, an integer from 1 to 30.
    #
    # NOTE This table isn't needed for any specific part of the assignment,
    #   but is used for associating location names, location titles, and 
    #   location ids.
    def add_locations(self, locations):
        self.locations = pd.DataFrame(data=locations, columns=['id', 'title', 'name', 'latitude', 'longitude', 'wiki'])
        self.locations = self.locations.set_index('id')
        print('Locations Loaded...')


    ##
    # Stores textual descriptors data.
    #
    # txt_descriptors dictionary of dictionaries mapping from:
    #   id -> [(term tf idf tfidf), ...]
    # each dictionary corresponding to locations, photos, users.
    #
    # NOTE id is not the key, as id's will repeat. (each id has multiple terms)
    def add_txt_descriptors(self, txt_descriptors):

        for desc_type, table in txt_descriptors.items():

            if desc_type == 'poi':
                old_table = dict(table)
                table = {}
                # we want to change the location names to the location ids
                #rows = [ [self.get_location_by_name(row[0])['id']] + row[1:] for row in rows]
                for poi_name in old_table.keys():
                    table[self.get_location_by_name(poi_name).name] = old_table[poi_name]
                # Manually delete old_table from memory - memory is maxed out otherwise
                del(old_table)

            b = pd.DataFrame.from_dict(data=table, orient='index', dtype='float')
            b.sort_index(inplace=True)
            self.txt_descriptors[desc_type] = b.to_sparse().fillna(0)
            del(b) # - attempt at memory efficiency.
        
        print("User Descriptions Loaded...")
    

    ##
    # Stores visual descriptors data.
    #
    # locationid is the id of the location that is being loaded into the database.
    # model in ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
    # csv file to load.
    # 
    # Each keypair maps to a dataframe with structure:
    #   <'photoid', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'>
    def add_visual_descriptors(self, files):
        def get_info_from_file(file):
            name = split(file)[1]
            return name.split(' ')

        for file in files:
            filename = Database.get_file_name(file)
            loc_title, model = get_info_from_file(filename)
            location = self.get_location_by_title(loc_title)
            locationid = int(location.name)
            # get number of columns in CSV file.
            with open(file, 'r') as f:
                readr = reader(f)
                ncol=len(next(readr))
            # load into dataframe.
            col_names = [model + '_' + str(i) for i in range(ncol-1)]
            table = pd.read_csv(file, index_col=0, header=None, names=col_names, dtype='float')
            self.vis_descriptors[locationid, model] = table.sort_index().to_sparse().fillna(0)
        
        # NOTE: This is added in phase III for efficiency purposes. Limits functionality for speed.
        self.simplify_db()

        print("Visual Descriptors Loaded...")


    
    def load_vis(self, subdir='saved'):
        """
        """
        path = abspath(join(subdir, 'visdata.pickle'))
        if isfile(path):
            self.vis_descriptors = None
            self.vis = pd.read_pickle(path)
            with open(abspath(join(subdir, 'loc.pickle')), 'rb') as f:
                self.loc_map = pickle.load(f)
            print('Visual Descriptors Loaded...')
            return True
        return False


    def simplify_db(self, subdir='saved'):
        """
        Used in phase III to put data into a reduced state that is faster, since we don't need all the
        visual data separated by location.
        """
        arbitrary_model = 'CM'
        for location in self.locations.index:
            for photo in self.vis_descriptors[location, arbitrary_model].index:
                self.loc_map[int(photo)] = location
                #self.loc_map[location] = list([int(a) for a in self.vis_descriptors[location,arbitrary_model].index])

        # Set as combined table.        
        self.vis = self.get_vis_table()
        file_loc = abspath(join(subdir, 'visdata.pickle'))
        self.vis.to_pickle(file_loc)
        with open(abspath(join(subdir, 'loc.pickle')), 'wb+') as f:
            pickle.dump(self.loc_map, f)
        del(self.vis_descriptors)
        self.vis_descriptors = None


    ##################################################################
    ###                      Retrieving Data                       ###
    ##################################################################



    # Other ###################################################

    def get_img_loc(self, image):
        if self.loc_map is None:
            raise ValueError('Loc Map must be loaded to get image locations.')
        # iterate over locations to find one with image.
        if image in self.loc_map:
            return self.loc_map[image]
        """
        for loc_id in self.get_location_ids():
            table = self.get_vis_table(loc_id, model)
            if int(image) in table.index:
                return loc_id
        """
        # Not found.
        return None
    

    def get_img_locs(self, images):
        """
        Gets locations for each image in a set. More efficient than calling above multiple times for
            multiple images.
        :param list images: list of int image ids.
        :return dict: maps image ids to their loc_id.
        """
        if self.loc_map is None:
            raise ValueError('Loc Map must be loaded to get image locations!')
        # iterate and create the dictionary.
        returnval = defaultdict(lambda: None)
        for image in images:
            if not image in self.loc_map:
                continue
            returnval[image] = self.loc_map[image]
        """
        # arbitrarily select a model.
        model = self.vis_models[0]
        for loc_id in self.get_location_ids():
            table = self.get_vis_table(loc_id, model)
            intersection = [i for i in images if i in table.index]
            for item in intersection:
                returnval[item] = loc_id
            if len(returnval) == len(images):
                break
        """
        return returnval



    # Locations ###############################################
        
    def get_table_value(self, table, field, value):
        return table.loc[table[field] == value]
            
    def get_location_by_title(self, title):
        location = self.get_table_value(self.locations, 'title', title)
        return location.iloc[0]
    
    def get_location_by_name(self, name):
        """
        Returns the series corresponding to a location.
        """
        location = self.get_table_value(self.locations, 'name', name)
        return location.iloc[0]
        
    def get_location(self, an_id):
        # location = self.get_table_value(self.locations, 'id', an_id)
        return self.locations.loc[an_id]

    def get_location_ids(self):
        return self.locations.index
    
    def get_location_titles(self):
        returnval = {}
        locs = self.get_location_ids()
        titles = self.locations['title']
        for loc, title in zip(locs, titles):
            returnval[loc] = title
        return returnval

    # txt descriptors #####################################
    def get_txt_desc_table(self, atype):
        return self.txt_descriptors[atype]


    def get_txt_vector(self, atype, an_id):
         """
         returns a series with term index and integer indicating presence of term.
         """
         table = self.get_txt_desc_table(atype)
         return table.loc[an_id]


    # vis descriptors #####################################


    # def get_vis_table(self):
    #    return self.vis

    def get_vis_table(self, locationid=None, model=None):
        # for Phase III only - speeds things up.
        if not self.vis is None:
            return self.vis

        # If given both a location and a model, get the table.
        if locationid and model:
            return self.vis_descriptors[locationid, model]

        # If only given a model, combine all location tables for that
        #   model by rows. (The photoids should all be unique with the)
        #   same nine data columns.
        elif model:
            table = None
            for loc in self.get_location_ids():
                loc_table = self.get_vis_table(loc, model)
                if table is None:
                    table = loc_table
                else:
                    table = table.combine_first(loc_table)
                del(loc_table) # Delete table for memory storage.
            return table.sort_index().to_sparse().fillna(0)

        # If only given the location, combine all tables will all models
        #   for that location. Each table should have the same photos but
        #   different data, so we append the columns from each table by index.
        elif locationid:
            table = None
            for vis_model in self.vis_models:
                model_table = self.get_vis_table(locationid, vis_model)
                if table is None:
                    table = model_table
                else:
                    table = table.merge(right=model_table, how='outer', left_index = True, right_index = True)
                del(model_table)
            return table.sort_index().to_sparse().fillna(0)

        # If given neither, combine all tables into a single table. Each location
        #   its own series of rows and each model its own series of 9 columns.
        # NOTE - the resulting table will have 90 columns and thousands of rows.
        else:
            table = None
            for loc in self.get_location_ids():
                sub_table = self.get_vis_table(locationid=loc)
                #  combine in table as appropriate.
                if table is None:
                    table = sub_table
                else:
                    table = table.combine_first(sub_table)
                del(sub_table)
            return table.to_sparse().fillna(0)


    # Get vector corresponding to the photo id  based on the locationid and model.
    #   If model is none, the tables will be combined to get a vector for 
    #   every model.
    #   If locationid is none, every location will be checked.
    #
    #   NOTE - checking with locationid and/or model as None will cause 
    #       a table join and then discarding the table. It would be more
    #       efficient to get the table and then get the vector yourself in
    #       any condition which the table is needed as well.
    def get_vis_vector(self, locationid, model, photoid):
        table = self.get_vis_table(locationid, model)
        return table.loc[photoid]


    ##################################################################
    ###                         Utilities                          ###
    ##################################################################

    @staticmethod
    def get_file_name(file):
        name = basename(file)
        return splitext(name)[0]

    def __sanitize__(self, input):

        input = str(input)
        input = input.replace("'", "").replace('"', '')
        return input

    def print_vis_desc(self):
        #for key, table in self.vis_descriptors.items():
        #    print("VIS DESCRIPTOR TABLE: " + str(key))
        #    self.print_table(table)
        print("VISUAL DESCRIPTOR TABLE")
        self.print_table(self.vis)

    def print_txt_desc(self):
        for key, table in self.txt_descriptors.items():
            print("TXT DESCRIPTOR TABLE: " + str(key))
            self.print_table(table)

    def print_locations(self):
        self.print_table(self.locations)
    
    def print_table(self, table):
        print(repr(table))