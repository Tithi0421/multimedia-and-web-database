#! /bin/usr/python3.6
from neo4j import GraphDatabase
import graphistry

class Neo4jDb():


    def __init__(self, uri='bolt://localhost:7687', username='neo4j', password='_neo4j'):
        """
        Initializes connection to database.
        :param str URI: location of graph database on local host.
        :param str username: the name of the user to login to local database with.
        :param str password: password of the user to login with.
        """
        neo4j_creds = {'uri': uri, 'auth': (username, password)}
        graphistry_creds = {'server': uri, 'api': 2, 'key': 'apikey'}
        self.__db__ = GraphDatabase.driver(**neo4j_creds)
        graphistry.register(bolt=neo4j_creds, **graphistry_creds)
        with self.__db__.session() as session:
            session.write_transaction(self.set_constraints)


    def __del__(self):
        """
        Reponsible deconstructor that closes database. May happen automatically, but just to be safe.
        """
        self.__db__.close()


    def set_constraints(self, tx):
        """
        Sets database properties so photo ids must be unique.
        :param Neo4j.Transaction tx: Item to execute command.
        """
        return tx.run("CREATE CONSTRAINT ON (n:PHOTO) ASSERT n.id IS UNIQUE;")

    
    def add_edge(self, tx, start_node, end_node, edge_weight):
        """
        Creates an edge from one photo to another with a defined edge weight.
        :param Neo4j.Transaction tx: Item to execute the command.
        :param int start_node: Id of photo for out edge.
        :param float edge_weight: Weight of edge. In this case, similarity value.
        :param int end_node: Id of photo for in edge.
        """
        command = "MERGE (a:PHOTO {id:%s})\n" % start_node
        command += "MERGE (b:PHOTO {id:%s})\n" % end_node
        command += "MERGE (a)-[:SIMILARITY {weight:%s}]->(b)\n" % edge_weight
        return tx.run(command)


    def add_similarity(self, similarity, k):
        """
        Adds similarity matrix information to database. Adds all ~8900 photos and the weighted edges
        to their k most similar partners.
        :param Pandas.Dataframe similarity: photo-photo similarity matrix
        :param int k: number of out edges for each photo.
        TODO This will currently always return itself as most similar with value 1. Is this correct?
        """
        assert(all(similarity.index == similarity.columns))
        photos = similarity.index
        # get the nearest neighbors similarity wise to each image and add to database.
        for photo in photos:
            photo = int(photo)
            row = similarity.loc[photo]
            row = row.sort_values(ascending=False) 
            nearest = row.index[:k]
            with self.__db__.session() as session:
                # TODO this is where we run into an error. Fix it! (Whitespace expected)
                out = [session.write_transaction(self.add_edge, photo, other, row[other]) for other in nearest]
                print(str(out))


    def add_cluster(self, photo_id, cluster_id):
        """
        Adds cluster identifier to the graph node specified.
        :param int photo_id: Id of photo to add cluster label to.
        :param string cluster_id: cluster identifier to add to photo.
        """
        with self.__db__.session() as session:
            command  = 'MATCH (n:PHOTO {id:%s})\n' % photo_id
            command += 'SET n.cluster = %s\n' % cluster_id
            return session.run(command)
    

    def get_cluster(self, cluster_id):
        """
        Retrieve photos in the cluster.
        :param string cluster-id: cluster identifier to retrieve nodes from.
        """
        with self.__db__.session() as session:
            command  = 'MATCH (n:PHOTO {cluster:%s})\n' % cluster_id
            command += 'RETURN n.id'
            return session.run(command)
    

    def get_neighbors(self, photo_id):
        """
        Retrieves all the neighbors of this node with their edge weighting.
        :param int photo_id: The photo to find the neighbors of.
        """
        with self.__db__.session() as session:
            command  = 'MATCH (:PHOTO {id:%s})-[e:SIMILARITY]->(b:PHOTO)\n' % photo_id
            command += 'RETURN b.id, e.weight\n'
            return session.run(command)


    def drop(self):
        """
        Exactly what it says - drops the database. Use with caution!
        """
        with self.__db__.session() as session:
            session.run("MATCH (a:PHOTO) DETACH DELETE a")
    
    

    def visualize(self, query):
        """
        Create display of the graph after performing the query attached."
        """
        graphistry.register()