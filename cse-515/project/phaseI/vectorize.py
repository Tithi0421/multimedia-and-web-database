from collections import defaultdict
from database import Database

# UNNECESSARY!!!!

class Vectorizer():

    ################################################################
    ###                      Textual Vectors                     ###
    ################################################################

    @staticmethod
    def text_vector(db, atype, an_id, model):
        """
        usage:
            db - our database
            atype - the type of the id (user, photo, poi)
            an_id - self explanatory
            model - value of vector to use. (tf, idf, tfidf)
        """
        vector = db.get_txt_vector(atype, an_id, model)
        return vector


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