"""
Task 1: Implement a program which, given a user ID, a model
(TF, DF, TF-IDF), and value “k”, returns the most similar
k users based on textual descriptors. For each match, also
list the overall matching score as well as the 3 terms
that have the highest similarity contribution.

Task 2: Implement a program which, given an image ID, a
model (TF, DF,TF-IDF), and value “k”, returns the most
similar k images based on textual descriptors. For each
match, also list the overall matching score as well as the
3 terms that have the highest similarity contribution.

Task 3: Implement a program which, given a location ID, a
model (TF, DF, TF-IDF), and value "k”, returns the most
similar k locations based on textual descriptors. For each
match, also list the overall matching score as well as the
3 terms that have the highest similarity contribution.
Note: In this phase, the location IDs will always be
specified as the “number” field (i.e., 1 to 30) in the 
devsettopics.xml file

Task 4: Implement a program which, given a location ID, a
model (CM, CM3x3, CN, CN3x3,CSD,GLRLM, GLRLM3x3,HOG,LBP,
LBP3x3), and value “k”, returns the most similar k 
locations based on the corresponding visual descriptors
of the images as specified in the “img” folder. For each
match, also list the overall matching score as well as the
3 image pairs that have the highest similarity contribution.

Task 5: Implement a program which, given a location ID and
value “k”, returns the most similar k locations based on
the corresponding visual descriptors of the images as
specified in the “img” folder. For each match, also list
the overall matching score and the individual contributions
of the 10 visual models.
"""