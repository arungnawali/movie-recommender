import streamlit as st
import pickle
import pandas as pd
import requests
def fetch_poster(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(movie_id))
    data=response.json()
    return "https://image.tmdb.org/t/p/w500/"+data['poster_path']

import requests
import pickle

# Direct link to the Google Drive file
url = 'https://drive.google.com/uc?export=download&id=1LxuSIP7Z35d1pEu5wAjuei4GARB7YKJh'

# Request the file
response = requests.get(url, stream=True)
response.raise_for_status()  # Ensure we notice bad responses

# Save the file to disk
with open('similarity.pkl', 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# Load the file
with open('similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

# Now the similarity_data variable contains the loaded data


movies_dict=pickle.load(open('movie_dict.pkl','rb'))
movies=pd.DataFrame(movies_dict)

st.title('Movie Recommender System')


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended=[]
    recommended_posters=[]
    for i in movies_list:
        movie_id=movies.iloc[i[0]].id
        recommended.append((movies.iloc[i[0]].title))
        recommended_posters.append(fetch_poster(movie_id))
    return recommended,recommended_posters

selected_movie = st.selectbox(
    "Select the movie",movies['title'].values)
if st.button("Recommend"):
    names,posters=recommend(selected_movie)
    col1, col2, col3,col4,col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(posters[0])

    with col2:
        st.text(names[1])
        st.image(posters[1])

    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])

    with col5:
        st.text(names[4])
        st.image(posters[4])

