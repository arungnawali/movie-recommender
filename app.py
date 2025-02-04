pip install nltk
   
import pickle
import streamlit as st
import requests
import numpy as np
import pandas as pd
from urllib.request import urlopen
movies=pd.read_csv('tmdb_5000_movies.csv',on_bad_lines='skip', engine='python')
crediturl=urlopen('https://drive.google.com/uc?export=download&id=1wrawnTW5jH0ldDk4_ttaJaG2o3bdkye2')
credits=pd.read_csv(crediturl,on_bad_lines='skip', engine='python')
movies=pd.merge(movies,credits,on='title')
movies.head(1)



#extracting required columns
movies=movies[['genres','id','keywords','title','overview','cast','crew']]
movies.isnull().sum()
movies.dropna(inplace=True)

import ast
def convert(obj):
  L=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L

movies['genres']=movies['genres'].apply(convert)

movies['keywords']=movies['keywords'].apply(convert)

import ast
def convert3(obj):
  L=[]
  counter=0
  for i in ast.literal_eval(obj):
    if(counter!=3):
      L.append(i['name'])
      counter+=1
    else:
      break
  return L

movies['cast']=movies['cast'].apply(convert3)

movies.head()

import ast
def convertdir(obj):
  L=[]
  for i in ast.literal_eval(obj):
    if(i['job']=="Director"):
      L.append(i['name'])

    else:
      continue
  return L

movies['crew']=movies['crew'].apply(convertdir)

movies.head()

movies['overview']=movies['overview'].apply(lambda x:x.split())

movies.head()

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])

movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

new_df=movies[['id','title','tags']]

new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))

new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(stop_words='english',max_features=5000)
vectors=cv.fit_transform(new_df["tags"]).toarray()

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

np.set_printoptions(threshold=10)  # Display only 5 elements

similarity=cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])

def recommend(movie):
  movie_index=new_df[new_df['title']==movie].index[0]
  distances=similarity[movie_index]
  movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  for i in movies_list:

    print(new_df.iloc[i[0]].title)

recommend("Avatar")

import pickle

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))



def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters


st.header('Movie Recommender System')
movies = pickle.load(open('movie_list.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])




