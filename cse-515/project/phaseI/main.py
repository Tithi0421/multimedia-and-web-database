# The interface for executing phase I.
#
# TODO
#   1. Figure how to get the terms w/ highest similarity contribution?
#           Term that contributes the least distance / highest similarity.
#   2. Ask if 'Nearest' is calculated using a distance measure of our choice?
#           Yes, but justify your selection.
#   3. Where does data come from for 4, 5?
#           NOTE: P5 is aggregating all 10 models of P4.

from os.path import isfile, abspath, isdir
from loader import Loader
from database import Database
from vectorize import Vectorizer
from neighbor import Neighbor


class Interface():

    def __init__(self):
        self.database = None
        self.valid_types = ['photo', 'user', 'poi']
        self.valid_txt_models = ['tf', 'df', 'tf-idf']
        self.valid_vis_models = ['CM', 'CM3x3', 'CN', 'CN3x3',
                'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3', 'ALL']
        self.loader = Loader()
        self.io()
    
    def io(self):
        
        print("Welcome to the CSE 515 data software. Please enter a command.\
              \nEnter \"help\" for a list of commands.")
        while True:
            user_input = input("\nEnter a Command:$ ")
            user_input = user_input.split(' ')

            if user_input[0] == 'help':
                self.help()
            elif user_input[0] == 'load':
                self.load(*user_input[1:])
            elif user_input[0] == 'get':
                # self.get(*user_input[1:])
                print("Deprecated")
            elif user_input[0] == 'quit':
                self.quit()
            elif user_input[0] == 'nearest-text':
                self.nearest_text(*user_input[1:])
            elif user_input[0] == 'nearest-visual':
                self.nearest_visual(*user_input[1:])
            elif user_input[0] == 'test':
                pass
            else:
                raise ValueError('The command specified was not a valid command.')

    ##
    # takes no arguments. Doesn't fail on any argument input.
    def help(self):
        print("The following are valid commands to the program.\
                \n\tload <filepath>\
                \n\tLoads the database at that file path. If the file does not exist, will prompt to create a new database using the folder of a users choice.\
                \n\t\t<filepath> - A valid file path in the system. \
                \n\tget <item type> <item>\
                \n\tGets the data from the database related to the item specified.\
                \n\t\t<item type> - The type of item we are looking for neighbors to. Valid options include 'photo', 'user', and 'poi'.\
                \n\t\t<item> - The id of the item in question. This is either a user id, photo id, and point of interest id.\
                \n\tnearest-text <item type> <item> <model> k\
                \n\tcalculates the nearest k items to the specified item using textual descriptors. If the item type, item, or model aren't valid it will return an error message.\
                \n\t\t<item type> - The type of item we are looking for neighbors to. Valid options include 'photo', 'user', and 'poi'.\
                \n\t\t<item> - The id of the item in question. This is either a user id, photo id, and point of interest id.\
                \n\t\t<model> - The distance method to use. Valid options include 'tf', 'df', and 'tf-idf'.\
                \n\tnearest-visual <item> <model> k\
                \n\tCalculates the nearest k items to the specified item using visual descriptors. If the item type, item, or model aren't valid it will return an error message.\
                \n\t\t<item> - The id of the item in question. This is a point of interest (location) id.\
                \n\t\t<model> - The distance method to use. Valid options include 'CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', and 'LBP3x3'.\
                \n\tquit\
                \n\t\texits the program and performs necessary cleanup.\
                \n\n")

    ##
    # takes a single argument - file to load.
    def load(self, *args):

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
        
        self.database = self.loader.make_database(folder)

        """if not isdir(folder):
            print("[WARNING] The folder provided does not exist.")
            user_input = input("Would you like to create and load it? (y/n):$ ")
            if user_input.lower()[0] == 'n':
                return
            elif user_input.lower()[0] == 'y':
                # Load database information from specified data folder.
                user_input = input("Please specify the directory of the dataset file to load:$ ")
                self.database = self.loader.make_database(user_input)
                print("Database loaded successfully!")
                # Allow user to save files into faster loaded format for future use.
                user_input = input('Would you like to save this database to the hard drive in an easy to load format:$ ')
                if user_input.lower()[0] == 'y':
                    user_input = input('Please specify the directory to save the files into: $')
                    self.database.save_database(user_input)
                    print("Successfully saved database to " + str(user_input))
            else:
                print("[ERROR] Please enter a valid input.")
        else:
            self.database = self.loader.load_database(folder)"""

    ##
    # retrieve an item from the database.
    # NOTE: By default this uses textual descriptors.
    #
    #   usage:
    #   arg[0] - type (user, photo, poi),
    #   arg[1] - id.
    """def get(self, *args):

        if not self.database:
            print("[ERROR] The database must be loaded for this action.")
            return

        if not len(args) == 2:
            print("Get expected 2 arguments but got " + str(len(args)) + ".")
            print("\targs = " + str(args))
            return
        
        item_type = args[0]
        if item_type == 'photo':
            item = self.database.get_photo(args[1])
            cols = self.database.get_photo_cols()
        elif item_type == 'user':
            item = self.database.get_user(args[1])
            cols = self.database.get_user_cols()
        elif item_type == 'poi':
            item = self.database.get_location(args[1])
            cols = self.database.get_location_cols()
        else:
            print("[ERROR] Type specified was invalid.")
        
        if item is None:
            print("No value could be found in the database matching those parameters.")
            return

        for i, tple in enumerate(item):
            print(str(i) + ").")
            for name, value in zip(cols, tple):
                print("\t" + str(name) + '\t' + str(value))
    """


    ##
    # 
    def quit(self, *args):
        #self.database.commit()
        #self.database.close()
        exit()

    ##
    # Takes a value for k, <item type>, <item>, <model>
    def nearest_text(self, *args):
        """
            arg[0] = type (user, photo, poi)
            arg[1] = id
            arg[2] = model (tf, df, tf-idf)
            arg[3] = k
        """
        
        if not self.database:
            print("[ERROR] The database must be loaded for this action.")
            return

        if not len(args) is 4:
            print("Nearest Text expected 4 arguments but got " + str(len(args)) + ".")
            print("\targs = " + str(args))
            return

        # Get the first argument
        try:
            k = int(args[3])
        except:
            print("[ERROR] K Value provided is invalid.")
            print("\tk = " + str(args[0]))
            return
        
        # Get the type of item we are considering.
        itype = args[0]
        if not itype in self.valid_types:
            print("[ERROR] Item Type value provided was invalid.")
            print("\tItem Type = " + str(args[1]))
            return
        
        # Get the model to use. We do this before the item as it is easier
        #   to differentiate valid from invalid
        model = args[2]
        model = model.lower()
        if not model in self.valid_txt_models:
            print("[ERROR] Model Type value provided was invalid.")
            print("\tModel Type = " + str(args[2]))
            return

        an_id = args[1]
        
        nearest = Neighbor.knn_textual(k, an_id, model, itype, self.database)
        contribs = Neighbor.similarity_by_id(an_id, nearest, 
                                            self.database, model, itype)

        print(str(k) + " Nearest Neighbors:")
        for i, (an_id, distance) in enumerate(nearest.items()):
            print('\t' + str(i) + ". " + str(an_id) + "; Distance = " + str(distance))
        print('Top 3 Features:')
        for i, item in enumerate(contribs):
            print('\t' + str(i) + '. ' + str(item))


    def nearest_visual(self, *args):
        """
            arg[0] = locationid
            arg[1] = model (CM, CM3x3, CN, CN3x3, CSD, GLRLM, GLRLM3x3, HOG, LBP, LBP3x3, ALL)
            arg[2] = k
        """
        
        if not self.database:
            print("[ERROR] The database must be loaded for this action.")
            return
        
        if not len(args) == 3:
            print("[ERROR] Expected three arguments but got " + str(len(args)) + ".")
            print("\targ = " + str(args))
        
        # Get the first argument
        try:
            k = int(args[2])
        except:
            print("[ERROR] K Value provided is invalid.")
            print("\tk = " + str(args[2]))
            return

        # Get the model to use. We do this before the item as it is easier
        #   to differentiate valid from invalid
        model = args[1]
        if not model in self.valid_vis_models:
            print("[ERROR] Model Type value provided was invalid.")
            print("\tModel Type = " + str(args[1]))
            return

        try:
            locationid = int(args[0])
        except:
            print("[ERROR] The ID specified was not valid")
            print("\tID = " + str(locationid) + "; Model = " + model)
            return
        
        if model == 'ALL':
            this_vector = Vectorizer.visual_vector_multimodel(locationid, self.database, self.valid_vis_models)
            nearest = Neighbor.knn_visual_all(k, locationid, self.valid_vis_models, self.database, this_vector = this_vector)
        else:
            # get vector representing item associated w/ id
            this_vector = Vectorizer.visual_vector(locationid, self.database, model)
            nearest = Neighbor.knn_visual(k, locationid, model, self.database, this_vector=this_vector)
        
        contribs = Neighbor.visual_sim_contribution(this_vector, nearest.keys(), self.database,
                                                    model, k=3)

        print(str(k) + " Nearest Neighbors:")
        for i, (an_id, distance) in enumerate(nearest.items()):
            print('\t' + str(i) + ". " + str(an_id) + "; Distance = " + str(distance))
        print('Top 3 Image Pairs:')
        for i, item in enumerate(contribs):
            print('\t' + str(i) + '. ' + str(item))
        


if __name__ == '__main__':
    Interface()