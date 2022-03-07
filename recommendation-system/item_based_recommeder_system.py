import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.decomposition import NMF #Data colector
# Surprise 
import surprise
from surprise.reader import Reader
from surprise import Dataset
from surprise.model_selection import GridSearchCV
# CrossValidation
from surprise.model_selection import cross_validate
from surprise import SVD,NMF

np.random.seed(42) # replicating results
# Importing Online Data
# The work considers only tidy data in ratings.csv 
# and movies.csv. Specifically, ratings_df 
# records userId, movieId, and rating consecutively. 
# On the other hand, movies_df stores values in movieId and genres.

# movieId is, therefore, the mutual variable.

# Note that Surprise enables one to upload data, e.g. csv files,
#  for predictions through its own methods. On the other hand, 
#  as it is discussed below, Surprise also allows the user 
#  to use pandas' DataFrames. The author works with pd.DataFrame 
#  objects for convenience.

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

r = urlopen("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
zipfile=ZipFile(BytesIO(r.read()))
# print the contentof zip file
zipfile.namelist()
# Tidy of ratings (movieId)
ratings_df=pd.read_csv(zipfile.open('ml-latest-small/ratings.csv'))
# print("Columns of ratongs_df: {0}".format(ratings_df.columns))
movies_df=pd.read_csv(zipfile.open('ml-latest-small/movies.csv'))
# print("Columns of movies_df: {0}".format(movies_df.columns))
# print(ratings_df.head())
# print(ratings_df.info())
# print(ratings_df.describe())
print(movies_df.head())

#Note that movies_df contains only movieId and genres variables which store even multiple genres 
# separated by the vertical bar in one cell.
# Data preprocessing







