from collections import defaultdict
from database import Database

class Vectorizer():

    ################################################################
    ###                      Textual Vectors                     ###
    ################################################################

    @staticmethod
    def text_vector(an_id, database, model, item_type):
        if item_type == 'photo':
            return Vectorizer.create_photo_term_vector(an_id, database, model)
        elif item_type == 'user':
            return Vectorizer.create_user_term_vector(an_id, database, model)
        elif item_type == 'poi':
            return Vectorizer.create_location_term_vector(an_id, database, model)
        else:
            raise ValueError('[ERROR] The provided type was invalid.\ntype = ' + str(item_type))

    @staticmethod
    def create_vector_from_desc_list(desc, model):

        vector = defaultdict(int)
        for _, term, tf, df, tfidf in desc:
            if model == 'tf':
                vector[term] = tf
            elif model == 'df':
                vector[term] = df
            elif model == 'tf-idf':
                vector[term] = tfidf
            else:
                raise ValueError('[ERROR] The value provided for model was not valid.\nmodel = ' + str(model))
        return vector


    @staticmethod
    def create_user_term_vector(an_id, database, model):
        desc = database.get_user_desc(an_id)
        return Vectorizer.create_vector_from_desc_list(desc, model)
    
    @staticmethod
    def create_location_term_vector(an_id, database, model):
        desc = database.get_loc_desc(an_id)
        return Vectorizer.create_vector_from_desc_list(desc, model)
    
    @staticmethod
    def create_photo_term_vector(an_id, database, model):
        desc = database.get_photo_desc(an_id)
        return Vectorizer.create_vector_from_desc_list(desc, model)

    ################################################################
    ###                       Visual Vectors                     ###
    ################################################################

    @staticmethod
    def visual_vector(locationid, database, model):
        # get photos at location.
        photoids = database.get_photos_at_location(locationid)
        # get visual vector for each photo. Use this to make a 
        #   vector describing the location.
        loc_vector = defaultdict(int)
        for i, photoid in enumerate(photoids):
            vector = database.get_photo_visual_desc(photoid, model)
            average = sum(vector) / len(vector)
            loc_vector[photoid] = average
        return loc_vector

    @staticmethod
    def visual_vector_multimodel(locationid, database, models):
        vector = {}
        # Calculate this_vector
        for model in models:
            model_vector = Vectorizer.visual_vector(locationid, database, model)
            # get average for this factor
            vector[model] = sum(model_vector.values()) / len(model_vector)
        return vector