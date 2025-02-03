import streamlit as st
import pickle
import pandas as pd
import requests
import gzip

# Function to fetch movie poster
def fetch_poster(movie_id):
    response = requests.get(
        'https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

# Helper function to download and decompress chunk files
def download_and_decompress(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we notice bad responses

    # Save the file to disk
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Decompress the file
    with gzip.open(filename, 'rb') as f_in:
        return pickle.load(f_in)

# URLs to the chunk files on GitHub
chunk_urls = [
 ' https://github.com/arungnawali/movie-recommender/blob/master/similarity_part_0.pkl.gz','https://github.com/arungnawali/movie-recommender/blob/master/similarity_part_0.pkl.gz'
]

# Download, decompress, and combine the chunks
similarity_data = []
for idx, url in enumerate(chunk_urls):
    filename = f'similarity_part_{idx}.pkl.gz'
    chunk = download_and_decompress(url, filename)
    similarity_data.extend(chunk)

# Load movies data
try:
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
except Exception as e:
    st.error("Error loading movie_dict.pkl: " + str(e))
    movies = pd.DataFrame()

# Streamlit app setup
st.title('Movie Recommender System')

# Define the recommend function
def recommend(movie):
    if not similarity_data:
        st.error("Error: similarity_data is not loaded correctly.")
        return [], []

    try:
        movie_index = movies[movies['title'] == movie].index[0]
    except IndexError:
        st.error("Error: Movie not found.")
        return [], []

    distances = similarity_data[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_posters

# User selects a movie
selected_movie = st.selectbox("Select the movie", movies['title'].values)

# Recommend movies
if st.button("Recommend"):
    names, posters = recommend(selected_movie)
    if names and posters:
        col1, col2, col3, col4, col5 = st.columns(5)

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
