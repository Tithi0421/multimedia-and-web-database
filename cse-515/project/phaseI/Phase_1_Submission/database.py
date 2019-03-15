import sqlite3

class SQLDatabase():


    def __init__(self, database):
        self.conn = sqlite3.connect(database)
        self.tables = ['users', 'user_photos', 'user_descriptions', 'photos',
                        'tags', 'photo_descriptions', 'visual_descriptors',
                        'photo_locations', 'locations', 'loc_descriptions']
    
    ##
    # NOTE - only run on new databases, else you will get errors
    def make_tables(self, visual_models = ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']):
        # USER TABLES'
        self.conn.execute("CREATE TABLE IF NOT EXISTS users ( \
                            userid string, \
                            visualScore real, \
                            faceProportion real, \
                            tagSpecificity real, \
                            locationSimilarity real, \
                            photoCount real, \
                            uniqueTags real, \
                            uploadFrequency real, \
                            bulkProportion real\
                        )")
        self.conn.execute("CREATE TABLE IF NOT EXISTS user_photos (\
                            userid string,\
                            photoid int\
                        )")
        self.conn.execute("CREATE TABLE IF NOT EXISTS user_descriptions (\
                            userid string,\
                            term string,\
                            tf real,\
                            df real,\
                            tfidf real\
                        )")
        # PHOTO TABLES
        self.conn.execute("CREATE TABLE IF NOT EXISTS 'photos' (\
                            photoid int,\
                            userid string,\
                            date_taken string,\
                            views int,\
                            title string, \
                            url_b string\
                        )")
        self.conn.execute("CREATE TABLE IF NOT EXISTS tags (\
                            photoid int,\
                            tag string\
                        )")
        self.conn.execute("CREATE TABLE IF NOT EXISTS photo_descriptions (\
                            photoid int,\
                            term string,\
                            tf real,\
                            df real,\
                            tfidf real\
                        )")
            
        # Made one table per model - attempted speedup
        for model in visual_models:
            self.conn.execute("CREATE TABLE IF NOT EXISTS visual_descriptors_" + model + " (\
                                photoid int,\
                                visual_desc real \
                            )")
        
        self.conn.execute("CREATE TABLE IF NOT EXISTS photo_locations (\
                            photoid int,\
                            locationid int\
                        )")
        # LOCATION TABLES
        self.conn.execute("CREATE TABLE IF NOT EXISTS locations (\
                            locationid int,\
                            title string,\
                            name string,\
                            latitude real,\
                            longitude real,\
                            wiki string\
                        )")
        self.conn.execute("CREATE TABLE IF NOT EXISTS loc_descriptions (\
                            locationid int,\
                            term string,\
                            tf real,\
                            df real,\
                            tfidf real\
                        )")

    ##################################################################
    ###                          ADDING DATA                       ###
    ##################################################################

    ##
    # 
    def add_request(self, table, **kwargs):

        request = "INSERT INTO " + table + " ("

        for i, parameter in enumerate(kwargs.keys()):
            if not i is 0:
                request += ", "
            request += str(parameter)
        
        request += ") VALUES ("
        
        for i, value in enumerate(kwargs.values()):
            if not i is 0:
                request += ', '

            request += "\'" + self.__sanitize__(value) + "\'"
        
        request += ");"

        try:
            self.conn.execute(request)
        except Exception as e:
            print("Query failed!\n" +
                  "Request: " + str(request) + 
                  "Exception: " + str(e))

    ##
    #
    def add_tag(self, photoid, tag):

        self.add_request('tags', photoid=photoid, tag=tag)

    ##
    # 
    def add_photo_desc(self, id, term, tf, df, tfidf):

        self.add_request('photo_descriptions', photoid=id, term=term, tf=tf, df=df, tfidf=tfidf)

    ##
    #
    def add_user_desc(self, id, term, tf, df, tfidf):

        self.add_request('user_descriptions', userid=id, term=term, tf=tf, df=df, tfidf=tfidf)

    ##
    #
    def add_loc_desc(self, id, term, tf, df, tfidf):
        self.add_request('loc_descriptions', locationid=id, term=term, tf=tf, df=df, tfidf=tfidf)
    
    ##
    #
    def add_user(self, userid, visualScore, faceProportion, tagSpecificity, locationSimilarity,
                photoCount, uniqueTags, uploadFrequency, bulkProportion):

        self.add_request('users', userid=userid, visualScore=visualScore, faceProportion=faceProportion, tagSpecificity=tagSpecificity,
                        locationSimilarity=locationSimilarity, photoCount=photoCount, uniqueTags=uniqueTags,
                        uploadFrequency=uploadFrequency, bulkProportion=bulkProportion)
    
    ##
    #
    def add_location(self, locationid, title, name, latitude, longitude, wiki):

        self.add_request('locations', locationid=locationid, title=title, name=name, latitude=latitude,
                        longitude=longitude, wiki=wiki)
    
    ##
    # 
    def add_photo(self, photoid, userid, date_taken, views, title, url_b):

        self.add_request('photos', photoid=photoid, userid=userid, date_taken=date_taken, views=views,
                        title=title, url_b=url_b)
        self.add_request('user_photos', userid=userid, photoid=photoid)
    
    ##
    # 
    def add_visual_descriptor(self, photoid, value, model):

        self.add_request('visual_descriptors_' + model, photoid=photoid, visual_desc=value)
        
    ##
    #
    def add_photo_location(self, photoid, locationid):

        self.add_request('photo_locations', photoid=photoid, locationid=locationid)


    ##################################################################
    ###                      Retrieving Data                       ###
    ##################################################################

    def get_request(self, table, cols=['*'], **kwargs):

        request = "SELECT "
        for i, col in enumerate(cols):
            if not i is 0:
                request += ", "
            request += str(col)

        request += " FROM " + table
        
        if kwargs:
            request += " WHERE "

            for i, (key, value) in enumerate(kwargs.items()):
                if not i is 0:
                    request += " AND "
                request += str(key) + "=" + "\'" + self.__sanitize__(value) + "\'"
        
        request += ";"

        req = self.conn.execute(request)
        return req.fetchall()

    # Locations ###############################################

    def get_location_titles(self):
        
        return self.get_request('locations', cols=['title'])

    def get_location_ids(self):
        
        idlist =  self.get_request('locations', cols=['locationid'])
        return [anid[0] for anid in idlist]

    def get_location(self, locationid):

        return self.get_request('locations', locationid=locationid)

    def get_location_by_name(self, name):

        return self.get_request('locations', name=name)
    
    def get_location_by_title(self, title):

        val = self.get_request('locations', title=title)
        return val[0]

    def get_loc_id(self, name):

        return self.get_request('locations', cols=['locationid'], name=name)

    def get_loc_terms(self, locationid):

        return self.get_request('loc_descriptions', cols=['terms'], locationid=locationid)
    
    def get_loc_desc(self, locationid):

        return self.get_request('loc_descriptions', locationid=locationid)

    # Users #####################################

    def get_user(self, userid):

        return self.get_request('users', userid=userid)

    def get_user_ids(self):

        idlist = self.get_request('users', cols=['userid'])
        return [an_id[0] for an_id in idlist]

    def get_user_photos(self, userid):

        return self.get_request('user_photos', cols=['photoid'], userid=userid)

    def get_user_terms(self, userid):

        return self.get_request('user_descriptions', cols=['terms'], userid=userid)

    def get_user_desc(self, userid):

        return self.get_request('user_descriptions', userid=userid)

    # Photos #####################################

    def get_photo(self, photoid):

        return self.get_request('photos', photoid=photoid)
    
    def get_photo_location(self, photoid):

        return self.get_request('photo_locations', cols=['locationid'], photoid=photoid)
    
    def get_photos_at_location(self, locationid):

        idlist = self.get_request('photo_locations', cols=['photoid'], locationid=locationid)
        return [an_id[0] for an_id in idlist]

    def get_photo_ids(self):

        idlist = self.get_request('photos', cols=['photoid'])
        return [an_id[0] for an_id in idlist]

    def get_photo_terms(self, photoid):
        
        return self.get_request('photo_descriptions', cols=['terms'], photoid=photoid)

    def get_photo_desc(self, photoid):

        return self.get_request('photo_descriptions', photoid=photoid)
    
    def get_photo_visual_desc(self, photoid, model):

        desc = self.get_request('visual_descriptors_' + model, cols=['visual_desc'], photoid=photoid)
        return [d[0] for d in desc]


    #################################################################
    ###                   Structural Gets                        ####
    #################################################################

    def get_cols(self, table):
        cols = self.conn.execute("SELECT * FROM " + table + ";")
        return [col[0] for col in cols.description]

    def get_photo_cols(self):
        return self.get_cols('photos')
    
    def get_user_cols(self):
        return self.get_cols('users')
    
    def get_location_cols(self):
        return self.get_cols('locations')


    ##################################################################
    ###                         Utilities                          ###
    ##################################################################

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()
    
    def empty(self):
        self.conn.execute("DROP DATABASE main")
    
    def __sanitize__(self, input):

        input = str(input)
        input = input.replace("'", "").replace('"', '')
        return input

