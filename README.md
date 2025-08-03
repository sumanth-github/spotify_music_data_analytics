**Spotify Music Trend Analyzer**
A full-stack interactive web application for analyzing Spotify music trends, artist statistics, and predicting song popularity using machine learning.

This project blends real-time Spotify playlist analysis with preprocessed audio features from Kaggle datasets to deliver insightful visualizations and predictive models — all within a user-friendly Streamlit interface.

**Overview**
This app allows users to:
-Upload and analyze public Spotify playlists using the Spotify Web API.
-Explore artists’ track and album trends from a large curated dataset.
-Predict the popularity of a song based on audio features.
-Discover similar tracks through a recommendation engine.
-Visualize various audio metrics with dynamic plots and dashboards.

**Features**
Upload Playlist
Analyze any Spotify playlist by entering a URL or ID.
Visual outputs include:
-Distribution of track popularity
-Top contributing artists
-Popularity of individual tracks
-Artist Explorer

Search any artist from the Kaggle dataset to get:
-Popularity timeline
-Audio feature radar charts
-Top tracks and pseudo-albums
-Release timeline visualization

ML Popularity Predictor
Use a trained Random Forest model to predict the expected popularity of a track based on input features:
-Danceability
-Energy
-Valence
-Acousticness
-Instrumentalness
-Liveness
-Tempo

### Track Recommender
Get similar songs based on cosine similarity of audio features. Ideal for playlist expansion and discovery.

### Visual Analytics
Custom plots include:
-Popularity histograms
-Bar charts of top artists/tracks
-Timeline charts
-Radar plots for audio signatures

### Tech Stack
Frontend: Streamlit
Backend/Data Handling: Python, Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: scikit-learn (Random Forest)
Spotify API Integration: Spotipy
Dataset Sources:
Kaggle Spotify Dataset
Spotify Web API

