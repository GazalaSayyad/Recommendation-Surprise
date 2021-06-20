import numpy as np
import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate
from surprise import SVD
import joblib

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('./dataset/u.data',  sep='\t', names=r_cols,
encoding='latin-1')
ratings.head()
i_cols = ['movie_id', 'title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('./dataset/u.item',  sep='|', names=i_cols, encoding='latin-1')
movies.head()
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./dataset/u.user', sep='|', names=u_cols,
encoding='latin-1')



ratings = ratings.drop(columns='timestamp')
reader = Reader()
data = Dataset.load_from_df(ratings, reader)
svd = SVD()
#Evaluate the performance in terms of RMSE
cross_validate(svd, data, measures=['RMSE'], cv = 3)
trainset = data.build_full_trainset()
svd.fit(trainset)

joblib.dump(svd, 'scoresvd.pkl')

