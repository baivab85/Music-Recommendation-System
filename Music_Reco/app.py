import os
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import spotipy
import altair as alt
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from yellowbrick.target import FeatureCorrelation

# Load Data
data = pd.read_csv("data.csv")
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')
artist_data = pd.read_csv('data_by_artist.csv')

# Set up Spotify API
os.environ["SPOTIFY_CLIENT_ID"] = "da9314febe574c278b2e3c410851cbf1"
os.environ["SPOTIFY_CLIENT_SECRET"] = "7bcd13f0c85d4784a891cb64b0afe133"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ["SPOTIFY_CLIENT_ID"],
                                                           client_secret=os.environ["SPOTIFY_CLIENT_SECRET"]))

# Define Functions
def get_decade(year):
    period_start = int(year/10) * 10
    decade = '{}s'.format(period_start)
    return decade

def find_song(name, year):
    song_data = defaultdict()
    query = 'track:{} {}'.format(name, 'year')
    print("Search Query:", query)  # Add this line to print the search query

    # Fetch a broader set of tracks
    results = sp.search(q=query, limit=10)
    if results['tracks']['items'] == []:
        print("No results found.")
        return None

    # Filter tracks based on release year
    filtered_tracks = [track for track in results['tracks']['items'] if track['album']['release_date'][:4] == str(year)]
    if not filtered_tracks:
        print("No tracks found for the specified year.")
        return None

    # Select the first track after filtering
    track_info = filtered_tracks[0]

    # Extract track information
    track_id = track_info['id']
    audio_features = sp.audio_features(track_id)[0]

    # Populate song data
    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(track_info['explicit'])]
    song_data['duration_ms'] = [track_info['duration_ms']]
    song_data['popularity'] = [track_info['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)



# Add custom CSS for dark background
# Streamlit App
st.set_page_config(
    page_title="Music Analysis App",
    page_icon=":musical_note:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set background color to black
st.markdown(
    """
    <style>
    [theme]
     primaryColor="#ffff44"
     backgroundColor="#000000"
     secondaryBackgroundColor="#ff00ff"
      textColor="#ffffff"
    </style>
    """,
    unsafe_allow_html=True
)



# Sidebar
st.sidebar.title("Select Section")
section = st.sidebar.selectbox("", ["Feature Correlation", "Decade Count Plot", "Sound Feature Trends", 
                                     "Loudness Trend", "Genre Clustering", "Song Clustering", "Song Recommendation"])

# Main Content
st.title("🎶 MUSIC ANALYSIS APP 🎶")
st.markdown("---")

if section == "Feature Correlation":
    feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                 'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                 'duration_ms', 'explicit', 'key', 'mode', 'year']

    X, y = data[feature_names], data['popularity']

# Create a list of the feature names
    features = np.array(feature_names)

# Instantiate the visualizer
    visualizer = FeatureCorrelation(labels=features)

# Fit the data to the visualizer
    visualizer.fit(X, y)

# Calculate the correlation matrix from the feature data
    correlation_matrix = np.corrcoef(X, rowvar=False)

# Convert the correlation matrix to a DataFrame for Plotly Express
    correlation_df = pd.DataFrame(correlation_matrix, columns=features, index=features)

# Plot the correlation matrix using Plotly Express heatmap
    fig = px.imshow(correlation_df, x=features, y=features)

# Update the layout
    fig.update_layout(
    title="Feature Correlation",
    xaxis_title="Features",
    yaxis_title="Features"
)

# Display the Plotly figure
    st.plotly_chart(fig)

elif section == "Decade Count Plot":
   data['decade'] = data['year'].apply(get_decade)

# Count the number of songs by decade
   decade_counts = data['decade'].value_counts().reset_index()
   decade_counts.columns = ['decade', 'count']

# Create a bar plot using Plotly Express
   fig = px.bar(decade_counts, x='decade', y='count', 
             labels={'decade': 'Decade', 'count': 'Count'},
             title='Number of Songs by Decade',
             color='decade',
             color_discrete_map={'decade': 'viridis'})

# Update layout
   fig.update_layout(xaxis_title='Decade', yaxis_title='Count')

# Display the Plotly figure
   st.plotly_chart(fig)

elif section == "Sound Feature Trends":
    st.subheader("Sound Feature Trends")
    
    sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
    fig = px.line(year_data, x='year', y=sound_features, title='Trend of various sound features over decades')
    st.plotly_chart(fig)  # Display the Plotly figure within the Streamlit app

elif section == "Loudness Trend":
      st.subheader("Loudness Trend")
    
      data['decade'] = data['year'].apply(get_decade)

      fig = px.line(data, x='decade', y='loudness', title='Trend of Loudness Over Decades')
      fig.update_layout(
        xaxis_title='Decade',
        yaxis_title='Loudness',
        width=800,  # Adjust the width of the plot
        height=400  # Adjust the height of the plot
    )
      st.plotly_chart(fig)
elif section == "Genre Clustering":
    st.subheader("Genre Clustering")
    
    top15_genres = genre_data.nlargest(15, 'popularity')

    fig = px.bar(top15_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], 
                 barmode='group', title='Trend of various features over top 15 genres')
    
    st.plotly_chart(fig)  # Display the Plotly figure within the Streamlit app

elif section == "Song Clustering":
    st.subheader("Song Clustering")
    
    cluster_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=10))
    ])

    X = genre_data.select_dtypes(np.number)
    cluster_pipeline.fit(X)
    genre_data['cluster'] = cluster_pipeline.named_steps['kmeans'].predict(X)

    tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
    genre_embedding = tsne_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
    projection['genres'] = genre_data['genres']
    projection['cluster'] = genre_data['cluster']

    fig = px.scatter(
        projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'], title='Clusters of genres'
    )
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.plotly_chart(fig)  # Display the Plotly figure within the Streamlit app

elif section == "Song Recommendation":
    st.subheader("Song Recommendation")
    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
                   'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

    # Authenticate with Spotify API
    client_credentials_manager = SpotifyClientCredentials(client_id='da9314febe574c278b2e3c410851cbf1',
                                                          client_secret='7bcd13f0c85d4784a891cb64b0afe133')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Function to find song data
    def get_song_data(song, spotify_data):
        try:
            song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                     & (spotify_data['year'] == song['year'])].iloc[0]
            return song_data
        except IndexError:
            return None
        
    # Function to recommend songs
    def recommend_songs(song_entry, spotify_data, n_songs=11):
        song_center = get_song_data(song_entry, spotify_data)
        if song_center is not None:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(spotify_data[number_cols])
            scaled_song_center = scaler.transform(song_center[number_cols].values.reshape(1, -1))
            distances = cdist(scaled_song_center, scaled_data, 'cosine')
            indices = np.argsort(distances)[0][:n_songs]
            recommended_songs = spotify_data.iloc[indices].to_dict(orient='records')
            unique_recommended_songs = []
            seen = set()
            for song in recommended_songs:
                if song['name'] not in seen and song['name'] != song_entry['name']:
                    unique_recommended_songs.append(song)
                    seen.add(song['name'])
            return unique_recommended_songs
        else:
            return []

    # Example of using recommend_songs function
    st.subheader("Enter Your Song Recommendation")
    song_name = st.text_input("Song Name")
    song_year = st.number_input("Song Year", min_value=1900, max_value=2024, step=1)

    # Button to trigger song recommendation
    if st.button("Recommend"):
        if song_name != '' and song_year != '':
            song_entry = {'name': song_name, 'year': song_year}
            recommended_songs = recommend_songs(song_entry, data, n_songs=10)
            if recommended_songs:
                st.write("Recommended Songs:")
                num_columns = 5
                num_recommended_songs = len(recommended_songs)
                num_rows = (num_recommended_songs + num_columns - 1) // num_columns
                for i in range(num_rows):
                    row = recommended_songs[i*num_columns : (i+1)*num_columns]
                    col1, col2, col3, col4, col5= st.columns(5)
                    for j, song in enumerate(row):
                        with col1 if j % 5 == 0 else col2 if j % 5 == 1 else col3 if j % 5 == 2 else col4 if j % 5 == 3 else col5:
                            # Retrieve image URL from Spotify API
                            results = sp.search(q=song['name'], limit=1, type='track')
                            if results['tracks']['items']:
                                track = results['tracks']['items'][0]
                                if track['album']['images']:
                                    image_url = track['album']['images'][0]['url']
                                    
                                    st.image(image_url, width=150, use_column_width=True, output_format="JPEG")
                                    st.markdown(f"**{song['name']}**")
                                    
                                else:
                                    st.write(f"{song['name']} (No image available)")
                            else:
                                st.write(f"{song['name']} (No image available)")
            else:
                st.write("No recommendations found for the given song.")
        else:
            st.write("Please enter the name and year of the song.")