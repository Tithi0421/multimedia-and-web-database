import pandas as pd
import os
from os import listdir
from os.path import basename, join, isfile, splitext
from csv import reader

class Database():

    ##
    # Store the various pandas dataframes for local use.
    def __init__(self):
        
        self.vis_descriptors = {}
        self.txt_descriptors = {}
        self.locations = None
        self.vis_models = ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
        


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
                    table[self.get_location_by_name(poi_name)['id']] = old_table[poi_name]
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
            name = os.path.split(file)[1]
            return name.split(' ')

        for file in files:
            filename = Database.get_file_name(file)
            loc_title, model = get_info_from_file(filename)
            location = self.get_location_by_title(loc_title)
            locationid = location['id']
            # get number of columns in CSV file.
            with open(file, 'r') as f:
                readr = reader(f)
                ncol=len(next(readr))
            # load into dataframe.
            col_names = [model + '_' + str(i) for i in range(ncol-1)]
            table = pd.read_csv(file, index_col=0, header=None, names=col_names, dtype='float')
            self.vis_descriptors[locationid, model] = table.sort_index().to_sparse().fillna(0)

        print("Visual Descriptors Loaded...")


    ##################################################################
    ###                      Retrieving Data                       ###
    ##################################################################


    # Locations ###############################################
        
    def get_table_value(self, table, field, value):
        return table.loc[table[field] == value]
            
    def get_location_by_title(self, title):
        location = self.get_table_value(self.locations, 'title', title)
        return location.iloc[0]
    
    def get_location_by_name(self, name):
        location = self.get_table_value(self.locations, 'name', name)
        return location.iloc[0]
        
    def get_location(self, an_id):
        location = self.get_table_value(self.locations, 'id', an_id)
        return location.iloc[0]

    def get_location_ids(self):
        return list(self.locations.loc[:,'id'])



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


    def get_vis_table(self, locationid=None, model=None):
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
                table.combine_first(sub_table)
                del(sub_table)
            return table.sort_index().to_sparse().fillna(0)


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
    ###                             IO                             ###
    ##################################################################

    def save_database(self, folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

        self.locations.to_csv(folder + '/locations.csv', index=False)

        txt_dir = folder + '/txt/'
        if not os.path.isdir(txt_dir):
            os.mkdir(txt_dir)
        for (atype, model), table in self.txt_descriptors.items():
            filename = atype + ' ' + model
            table.to_csv(txt_dir + '/' + filename + '.csv', chunksize=50000, index=False)

        vis_dir = folder + '/vis/'
        if not os.path.isdir(vis_dir):
            os.mkdir(vis_dir)
        for (locationid, model), table in self.vis_descriptors.items():
            filename = str(locationid) + ' ' + str(model)
            table.to_csv(vis_dir + '/' + filename + '.csv', chunksize=50000, index=False)


    def save_database_feather(self, folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

        self.locations.to_feather(folder + '/locations.feather')

        txt_dir = folder + '/txt/'
        if not os.path.isdir(txt_dir):
            os.mkdir(txt_dir)
        for (atype, model), table in self.txt_descriptors.items():
            filename = atype + ' ' + model
            table.to_feather(txt_dir + '/' + filename + '.feather')

        vis_dir = folder + '/vis/'
        if not os.path.isdir(vis_dir):
            os.mkdir(vis_dir)
        for (locationid, model), table in self.vis_descriptors.items():
            filename = str(locationid) + ' ' + str(model)
            table.to_feather(vis_dir + '/' + filename + '.feather')



    def save_database_hdf(self, folder):

        if not os.path.isdir(folder):
            os.mkdir(folder)

        self.locations.to_hdf(folder + '/locations.hdf', key='loc', mode='w', format='fixed')

        txt_keys = []
        for (atype, model), table in self.txt_descriptors.items():
            filename = atype + ' ' + model
            txt_keys.append(filename)
            table.to_hdf(folder + '/txt.hdf', key=filename, mode='a', format='fixed')

        with open(folder + '/txt_keys.txt', 'w+') as f:
            for key in txt_keys:
                f.write(key)

        vis_keys = []
        for (locationid, model), table in self.vis_descriptors.items():
            filename = str(locationid) + ' ' + str(model)
            vis_keys.append(filename)
            table.to_hdf(folder + '/vis.hdf', key=filename, mode='a', format='fixed')

        with open(folder + '/vis_keys.txt', 'w+') as f:
            for key in vis_keys:
                f.write(key)
    

    @staticmethod
    def load_database(folder):

        db = Database()

        # Each local variable is stored in its own folder with subfolders as appropriate.
        db.locations = pd.read_csv(folder + '/locations.csv')

        txt_dir = folder + '/txt/'
        for file in [file for file in listdir(txt_dir) if isfile(join(txt_dir, file))]:
            filename, _ = splitext(file)
            atype, model = filename.split(' ')
            db.txt_descriptors[atype, model] = pd.read_csv(txt_dir + file, engine='c', dtype='float')

        vis_dir = folder + '/vis/'
        for file in [file for file in listdir(txt_dir) if isfile(join(vis_dir, file))]:
            filename, _ = splitext(file)
            locationid, model = filename.split(' ')
            db.vis_descriptors[locationid, model] = pd.read_csv(vis_dir + file, engine='c', dtype='float')

        return db
    

    @staticmethod
    def load_database_feather(folder):

        db = Database()

        # Each local variable is stored in its own folder with subfolders as appropriate.
        db.locations = pd.read_feather(folder + '/locations.feather')

        txt_dir = folder + '/txt/'
        for file in [file for file in listdir(txt_dir) if isfile(join(txt_dir, file))]:
            filename = Database.get_file_name(file)
            atype, model = filename.split(' ')
            db.txt_descriptors[atype, model] = pd.read_feather(txt_dir + file, nthreads=8)

        vis_dir = folder + '/vis/'
        for file in [file for file in listdir(txt_dir) if isfile(join(vis_dir, file))]:
            filename = Database.get_file_name(file)
            locationid, model = filename.split(' ')
            db.vis_descriptors[locationid, model] = pd.read_feather(vis_dir + file, nthreads=8)

        return db


    @staticmethod
    def load_database_hdf(folder):

        db = Database()

        # Each local variable is stored in its own folder with subfolders as appropriate.
        db.locations = pd.read_hdf(folder + '/locations.hdf')

        lines = [line.rstrip('\n') for line in open(folder + '/txt_keys.txt')]
        for txt_key in lines:
            atype, model = txt_key.split(' ')
            db.txt_descriptors[atype, model] = pd.read_hdf(folder + '/txt.hdf', key=atype + ' ' + model, mode='r')

        lines = [line.rstrip('\n') for line in open(folder + '/vis_keys.txt')]
        for vis_key in lines:
            locationid, model = vis_key.split(' ')
            db.vis_descriptors[locationid, model] = pd.read_hdf(folder + '/vis.hdf', key=locationid + ' ' + model, mode='r')

        return db



    ##################################################################
    ###                         Utilities                          ###
    ##################################################################

    @staticmethod
    def get_file_name(file):
        name = os.path.basename(file)
        return os.path.splitext(name)[0]

    def __sanitize__(self, input):

        input = str(input)
        input = input.replace("'", "").replace('"', '')
        return input

    def print_vis_desc(self):
        for key, table in self.vis_descriptors.items():
            print("VIS DESCRIPTOR TABLE: " + str(key))
            self.print_table(table)

    def print_txt_desc(self):
        for key, table in self.txt_descriptors.items():
            print("TXT DESCRIPTOR TABLE: " + str(key))
            self.print_table(table)

    def print_locations(self):
        self.print_table(self.locations)
    
    def print_table(self, table):
        print(repr(table))