import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import plotly.express as px
import time
import streamlit as st
import random
import ast

client_id = "44ff9a105fa9440fb1262e78085c1bc3"
client_secret = "3a5d85a572c24690a2ae7adb2cec53a7"

# Set up credentials manager
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

def get_playlist_tracks(playlist_id):
    results = sp.playlist_items(playlist_id, limit=50)
    tracks = results['items']

    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    return tracks

def parse_tracks(tracks_data):
    track_list = []

    for item in tracks_data:
        track = item.get('track')
        if track:
            track_info = {
                'track_name': track.get('name', 'N/A'),
                'artist_name': track.get('artists', [{'name': 'N/A'}])[0].get('name', 'N/A'),
                'album': track.get('album', {}).get('name', 'N/A'),
                'release_date': track.get('album', {}).get('release_date', 'N/A'),
                'duration_ms': track.get('duration_ms', 0),
                'popularity': track.get('popularity', 0),
                'spotify_url': track.get('external_urls', {}).get('spotify', 'N/A'),
                'track_id': track.get('id', 'N/A')
            }
            track_list.append(track_info)

    return pd.DataFrame(track_list)
# 🎧 Artist Explorer - Phase 2


    


def plot_correlation_heatmap(df):
    numerical_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'popularity']
    df_subset = df[numerical_features]
    corr = df_subset.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap of Audio Features", fontsize=14)
    return fig

def load_scaled_features(df):
    features = df[[
        'danceability', 'energy', 'valence',
        'acousticness', 'instrumentalness',
        'liveness', 'tempo'
    ]].copy()
    
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)

# Global cached features (used in recommend_tracks)
df = pd.read_csv(r"spotify_dataset\tracks.csv")  # For your artist explorer and rec engine
scaled_features = load_scaled_features(df)

# Update recommend_tracks to work with global df and scaled_features
def recommend_tracks(track_name, n=5):
    track_idx = df[df['track_name'].str.lower() == track_name.lower()].index
    if track_idx.empty:
        return f"Track '{track_name}' not found in the dataset."

    track_vector = scaled_features[track_idx[0]].reshape(1, -1)
    similarities = cosine_similarity(track_vector, scaled_features).flatten()

    similar_indices = similarities.argsort()[::-1][1:n+1]
    return df.iloc[similar_indices][['track_name', 'artist_name', 'popularity']]






def plot_popularity_distribution(df_tracks):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_tracks['popularity'], bins=10, kde=True, color='#00FFFF')
    plt.title("Distribution of Track Popularity")
    plt.xlabel("Popularity")
    plt.ylabel("Count")
    plt.tight_layout()
    return plt.gcf()


def plot_tempo_vs_popularity(df):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='tempo', y='popularity', hue='explicit', palette="coolwarm", ax=ax)
    ax.set_title("Tempo vs Popularity", fontsize=14)
    return fig






def extract_playlist_id(url):
    return url.split("/")[-1].split("?")[0]

def clean_dataset(tracks_df, artists_df=None):
    if artists_df is not None and 'artist_id' in tracks_df.columns and 'id' in artists_df.columns:
        combined_df = tracks_df.merge(artists_df, left_on='artist_id', right_on='id', how='left')
    else:
        combined_df = tracks_df.copy()

    # Clean it up
    if 'release_date' in combined_df.columns:
        combined_df['release_date'] = pd.to_datetime(combined_df['release_date'], errors='coerce')

    combined_df.drop_duplicates(inplace=True)
    combined_df.dropna(subset=['popularity'], inplace=True)  # drop if popularity missing

    if 'duration_ms' in combined_df.columns:
        combined_df['duration_min'] = combined_df['duration_ms'] / 60000

    return combined_df




def plot_track_duration(df):
    if 'duration_min' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['duration_min'], bins=50, color='green', ax=ax)
        ax.set_title("Distribution of Track Durations")
        ax.set_xlabel("Duration (minutes)")
        ax.set_ylabel("Frequency")
        return fig





# 📈 Explore Artist
def explore_artist(artist_name, df):
    artist_tracks = df[df['artists'].str.contains(artist_name, case=False, na=False)]

    if artist_tracks.empty:
        st.warning(f"No tracks found for artist: {artist_name}")
        return

    st.success(f"Found {artist_tracks.shape[0]} tracks for artist: {artist_name}")
    st.dataframe(artist_tracks[['name', 'release_date', 'popularity', 'duration_ms']].sort_values(by='popularity', ascending=False).head(10))

    artist_tracks['duration_min'] = artist_tracks['duration_ms'] / 60000

    st.markdown("### 📊 Average Stats")
    st.dataframe(artist_tracks[['danceability', 'energy', 'valence', 'tempo', 'popularity']].mean().round(2))

    # Popularity Over Time
    plt.figure(figsize=(12, 5))
    artist_tracks_sorted = artist_tracks.sort_values(by='release_date')
    sns.lineplot(x='release_date', y='popularity', data=artist_tracks_sorted)
    plt.xticks(rotation=45)
    plt.title(f"📈 Popularity Over Time - {artist_name.title()}")
    st.pyplot(plt.gcf())
    plt.clf()

    # Radar Chart
    audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness']
    mean_vals = artist_tracks[audio_features].mean().values
    angles = np.linspace(0, 2 * np.pi, len(audio_features), endpoint=False).tolist()
    mean_vals = np.concatenate((mean_vals, [mean_vals[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, mean_vals, color='blue', linewidth=2)
    ax.fill(angles, mean_vals, color='skyblue', alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(audio_features)
    ax.set_title(f"Audio Profile: {artist_name.title()}")
    st.pyplot(fig)

# 🏆 Top Tracks
def get_top_tracks(df, top_n=20):
    top_tracks = df.dropna(subset=["track_name", "popularity"]).sort_values("popularity", ascending=False).head(top_n)
    return top_tracks[["track_name", "popularity"]]

# 💿 Albums by Artist
def get_artist_albums(artist_name, df):
    artist_tracks = df[df["artists"].str.contains(artist_name, case=False, na=False)].copy()
    artist_tracks["release_year"] = pd.to_datetime(artist_tracks["release_date"], errors='coerce').dt.year
    artist_tracks["pseudo_album"] = artist_tracks["name"] + " (" + artist_tracks["release_year"].astype(str) + ")"
    artist_tracks.dropna(subset=["release_year"], inplace=True)
    albums = artist_tracks[["pseudo_album", "release_year"]].drop_duplicates().sort_values(by="release_year")

    return albums

# 🕒 Timeline of Albums
def plot_artist_album_timeline(artist_name, df, max_albums=15):
    artist_tracks = df[df["artists"].str.contains(artist_name, case=False, na=False)].copy()
    artist_tracks["release_date"] = pd.to_datetime(artist_tracks["release_date"], errors="coerce")
    artist_tracks.dropna(subset=["release_date"], inplace=True)
    unique_albums = artist_tracks[["name", "release_date"]].drop_duplicates().sort_values("release_date").tail(max_albums)
    unique_albums["short_name"] = unique_albums["name"].apply(lambda x: x if len(x) <= 30 else x[:27] + "...")

    fig, ax = plt.subplots(figsize=(12, 0.5 * max_albums))
    ax.scatter(unique_albums["release_date"], range(len(unique_albums)), color="darkblue")
    ax.set_yticks(range(len(unique_albums)))
    ax.set_yticklabels(unique_albums["short_name"])
    ax.set_xlabel("Release Date")
    ax.set_title(f"Timeline of Last {max_albums} Albums - {artist_name}")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

# 🎨 General Insights
def plot_track_popularity(df):
    top_tracks = get_top_tracks(df)
    plt.figure(figsize=(12, 8))
    sns.barplot(x="popularity", y="track_name", data=top_tracks, palette="viridis")
    plt.title("Top 20 Tracks by Popularity", fontsize=16)
    plt.xlabel("Popularity")
    plt.ylabel("Track")
    st.pyplot(plt.gcf())
    plt.clf()


def get_artist_top_tracks(df, top_n=20):
    top_tracks = df.dropna(subset=["name", "artists", "popularity"]).copy()
    
    # Clean up the artists column for readability
    top_tracks['artists'] = top_tracks['artists'].astype(str).str.replace(r"[\[\]']", "", regex=True)
    top_tracks['artists'] = top_tracks['artists'].str.strip().str.replace(r"\s*,\s*", ", ", regex=True)

    top_tracks = top_tracks.sort_values("popularity", ascending=False).head(top_n)
    return top_tracks[["name", "artists", "popularity"]]



def plot_artist_track_popularity(df):
    top_tracks = get_artist_top_tracks(df)
    plt.figure(figsize=(12, 8))
    sns.barplot(x="popularity", y="name", data=top_tracks, palette="viridis")
    plt.title("Top 20 Tracks by Popularity", fontsize=16)
    plt.xlabel("Popularity")
    plt.ylabel("Track")
    st.pyplot(plt.gcf())
    plt.clf()
    
def plot_top_explorer_artists(df, top_n=15):
    # Wrap artist name in a list to make it consistent
    df = df.copy()
    df["artists_cleaned"] = df["artists"].apply(lambda x: [x] if isinstance(x, str) else [])

    # Flatten all artists into a single list
    all_artists = [artist.strip() for sublist in df["artists_cleaned"] for artist in sublist]

    # Count top artists
    top_artists = pd.Series(all_artists).value_counts().head(top_n)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_artists.values, y=top_artists.index, palette='plasma')
    plt.title("Top Artists by Track Count")
    plt.xlabel("Track Count")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_duration_distribution(df):
    df["duration_min"] = df["duration_ms"] / 60000
    plt.figure(figsize=(10, 6))
    sns.histplot(df["duration_min"], bins=30, color="skyblue", kde=True)
    plt.title("Track Duration Distribution")
    plt.xlabel("Duration (Minutes)")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_top_artists(df, top_n=15):
    # Wrap artist name in a list to make it consistent
    df = df.copy()
    df["artists_cleaned"] = df["artist_name"].apply(lambda x: [x] if isinstance(x, str) else [])

    # Flatten all artists into a single list
    all_artists = [artist.strip() for sublist in df["artists_cleaned"] for artist in sublist]

    # Count top artists
    top_artists = pd.Series(all_artists).value_counts().head(top_n)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_artists.values, y=top_artists.index, palette='plasma')
    plt.title("Top Artists by Track Count")
    plt.xlabel("Track Count")
    st.pyplot(plt.gcf())
    plt.clf()




def load_tracks_data():
    return pd.read_csv(r"spotify_dataset/tracks.csv")

def get_artist_tracks(df, artist_name):
    artist_tracks = df[df["artists"].str.contains(artist_name, case=False, na=False)].copy()
    if artist_tracks.empty:
        return None
    artist_tracks["duration_min"] = artist_tracks["duration_ms"] / 60000
    artist_tracks["release_date"] = pd.to_datetime(artist_tracks["release_date"], errors="coerce")
    artist_tracks["release_year"] = artist_tracks["release_date"].dt.year
    artist_tracks["pseudo_album"] = artist_tracks["name"] + " (" + artist_tracks["release_year"].astype(str) + ")"
    return artist_tracks






@st.cache_data
def load_genre_data(path=r"spotify_dataset/features.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=['genre'])
    
    # Convert all relevant features to numeric
    audio_features = ['acousticness', 'danceability', 'duration_ms', 'energy',
                      'instrumentalness', 'key', 'liveness', 'loudness',
                      'mode', 'speechiness', 'tempo', 'time_signature', 'valence']
    
    df[audio_features] = df[audio_features].apply(pd.to_numeric, errors='coerce')
    return df


def plot_top_genres_by_popularity(df):

    top_10_genres = df.groupby('genre')['popularity'].mean().sort_values(ascending=False).head(10).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_10_genres, x='popularity', y='genre', palette='coolwarm', ax=ax)
    ax.set_title('🎵 Top 10 Most Popular Genres (Avg. Popularity)', fontsize=14)
    ax.set_xlabel('Average Popularity')
    ax.set_ylabel('Genre')
    st.pyplot(fig)

    return top_10_genres



def plot_audio_feature_radar(df, top_genres):
    import numpy as np

    radar_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness']
    filtered_df = df[df['genre'].isin(top_genres)]
    genre_features = filtered_df.groupby('genre')[radar_features].mean()

    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for genre in genre_features.index:
        values = genre_features.loc[genre].tolist()
        values += values[:1]
        ax.plot(angles, values, label=genre)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_features)
    ax.set_title("🎧 Audio Feature Profile by Genre", size=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)



def plot_valence_vs_energy(df):
    top_500 = df.sort_values(by="popularity", ascending=False).head(500)

    fig = px.scatter(top_500,
                     x="valence",
                     y="energy",
                     color="genre",
                     size="popularity",
                     hover_name="track_name",
                     title="Valence vs Energy by Genre (Top 500 Tracks)")
    st.plotly_chart(fig, use_container_width=True)

def plot_danceability_vs_energy(df):
    top_genres = df['genre'].value_counts().nlargest(10).index
    filtered_df = df[df['genre'].isin(top_genres)]
    filtered_df = filtered_df.groupby('genre').apply(lambda x: x.head(100)).reset_index(drop=True)

    fig = px.scatter(filtered_df,
                     x='danceability',
                     y='energy',
                     color='genre',
                     hover_name='track_name',
                     title='Danceability vs Energy (Top Genres, 100 Tracks Each)')
    st.plotly_chart(fig, use_container_width=True)





def assign_random_country(df, country_list=None):
    if country_list is None:
        country_list = [
            'United States', 'India', 'United Kingdom', 'Germany', 'France',
            'Brazil', 'Canada', 'Australia', 'Japan', 'Mexico',
            'South Korea', 'Spain', 'Italy', 'Netherlands', 'Sweden'
        ]
    random.seed(42)
    df = df.copy()
    df['country'] = [random.choice(country_list) for _ in range(len(df))]
    return df

def get_top_tracks_by_country(df, country_name, top_n=10):
    df_country = df[df['country'] == country_name]
    if df_country.empty:
        return pd.DataFrame()  # For error handling in app
    return df_country.sort_values(by='popularity', ascending=False).head(top_n)[['track_name', 'artist_name', 'popularity']]



import pandas as pd
import os

# Define a mapping of filenames to country names
COUNTRY_FILES = {
    "regional-ca-weekly-2025-05-15.csv": "Canada",
    "regional-br-weekly-2025-05-15.csv": "Brazil",
    "regional-fr-weekly-2025-05-15.csv": "France",
    "regional-de-weekly-2025-05-15.csv": "Germany",
    "regional-se-weekly-2025-05-15.csv": "Sweden",
    "regional-nl-weekly-2025-05-15.csv": "Netherlands",
    "regional-it-weekly-2025-05-15.csv": "Italy",
    "regional-kr-weekly-2025-05-15.csv": "South Korea",
    "regional-mx-weekly-2025-05-15.csv": "Mexico",
    "regional-gb-weekly-2025-05-15.csv": "UK",
    "regional-in-weekly-2025-05-15.csv": "India",
    "regional-us-weekly-2025-05-15.csv": "USA",
    "regional-jp-weekly-2025-05-15.csv": "Japan",
    "regional-au-weekly-2025-05-15.csv": "Australia",
    "regional-global-weekly-2025-05-15.csv": "Global",
    "regional-es-weekly-2025-05-15.csv": "Spain"
    
}

def load_country_charts(data_dir="spotify_dataset"):
    dfs = []
    for filename, country in COUNTRY_FILES.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["country"] = country
            dfs.append(df)
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    else:
        raise FileNotFoundError("No country CSVs found in the dataset folder.")

def get_top_tracks_by_country(df, country, n=10):
    country_df = df[df["country"] == country].copy()

    # Normalize columns for processing
    country_df.columns = [col.strip().lower().replace(" ", "_") for col in country_df.columns]

    # Rename to standardized column names
    column_mapping = {
        'position': 'rank',
        'track name': 'track_name',
        'artist': 'artist_names',
        'streams': 'streams'
    }

    country_df = country_df.rename(columns=column_mapping)

    # Check required columns
    expected_cols = ['rank', 'track_name', 'artist_names', 'streams']
    missing = [col for col in expected_cols if col not in country_df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # Select top n and rename for display
    display_df = country_df.nlargest(n, 'streams')[expected_cols].copy()
    display_df['country'] = country  # add back

    # Rename for display (Title Case, no underscores)
    display_df.columns = [col.replace("_", " ").title() for col in display_df.columns]

    return display_df


def plot_average_popularity_over_years(df):
    df = df.copy()
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date'])

    df['year'] = df['release_date'].dt.year
    df = df[(df['year'] >= 1950) & (df['year'] <= 2020)]

    yearly_popularity = df.groupby('year')['popularity'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly_popularity['year'], yearly_popularity['popularity'], color='mediumslateblue', linewidth=2)
    ax.set_title("📈 Average Track Popularity Over Years")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Popularity")
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_linear_regression(df):
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'speechiness', 'tempo', 'valence']
    df = df.dropna(subset=features + ['popularity'])

    X = df[features]
    y = df['popularity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    joblib.dump(model, "linear_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return model, scaler, X_test_scaled, y_test, y_pred


def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.4)
    ax.plot([0, 100], [0, 100], color='red', linestyle='--')
    ax.set_xlabel("Actual Popularity")
    ax.set_ylabel("Predicted Popularity")
    ax.set_title("Actual vs Predicted Popularity")
    ax.grid(True)
    return fig


import joblib

def evaluate_random_forest_model(model_path, X_test_scaled, y_test):
    rf = joblib.load(model_path)
    y_pred_rf = rf.predict(X_test_scaled)

    scores = {
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'rmse': mean_squared_error(y_test, y_pred_rf) ** 0.5,
        'r2': r2_score(y_test, y_pred_rf),
        'y_pred_rf': y_pred_rf,
        'rf': rf
    }
    return scores

def plot_rf_feature_importance(rf, feature_names):
    feat_importance = pd.Series(rf.feature_importances_, index=feature_names)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feat_importance.values, y=feat_importance.index, ax=ax)
    ax.set_title("Random Forest Feature Importance")
    return fig

def plot_model_comparison(lin_metrics, rf_metrics):
    models = ['Linear Regression', 'Random Forest']
    mae_scores = [lin_metrics['mae'], rf_metrics['mae']]
    rmse_scores = [lin_metrics['rmse'], rf_metrics['rmse']]
    r2_scores = [lin_metrics['r2'], rf_metrics['r2']]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    axs[0].bar(models, mae_scores, color=['skyblue', 'forestgreen'])
    axs[0].set_title('MAE')

    axs[1].bar(models, rmse_scores, color=['skyblue', 'forestgreen'])
    axs[1].set_title('RMSE')

    axs[2].bar(models, r2_scores, color=['skyblue', 'forestgreen'])
    axs[2].set_title('R² Score')

    for ax in axs:
        ax.set_ylabel('Score')
        ax.set_xticklabels(models, rotation=15)

    plt.tight_layout()
    return fig



def plot_average_popularity(csv_path='spotify_dataset/tracks.csv'):
    # Load dataset
    tracks = pd.read_csv(csv_path)

    # Clean and transform
    tracks['release_date'] = pd.to_datetime(tracks['release_date'], errors='coerce')
    tracks = tracks.dropna(subset=['release_date'])
    tracks['year'] = tracks['release_date'].dt.year
    tracks = tracks[(tracks['year'] >= 1950) & (tracks['year'] <= 2020)]

    # Group and calculate
    yearly_popularity = tracks.groupby('year')['popularity'].mean().reset_index()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly_popularity['year'], yearly_popularity['popularity'], color='mediumslateblue', linewidth=2)
    ax.set_title("Average Track Popularity Over Years", fontsize=16)
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Popularity")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    return fig


def plot_actual_vs_predicted(y_true, y_pred, model_name="Linear Regression"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.4)
    ax.plot([0, 100], [0, 100], color='red', linestyle='--')
    ax.set_xlabel("Actual Popularity")
    ax.set_ylabel("Predicted Popularity")
    ax.set_title(f"Actual vs Predicted Popularity ({model_name})")
    ax.grid(True)
    plt.tight_layout()
    return fig



from sklearn.metrics.pairwise import cosine_similarity

def recommend_tracks(track_name, df, scaled_features, n=5):
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    if 'name' not in df.columns:
        return "🛑 Column 'name' not found in the dataset."

    # Find index of the track (case-insensitive match)
    track_idx = df[df['name'].str.lower() == track_name.lower()].index
    if track_idx.empty:
        return f"🚫 Track '{track_name}' not found in the dataset."

    # Compute cosine similarity
    track_vector = scaled_features[track_idx[0]].reshape(1, -1)
    similarities = cosine_similarity(track_vector, scaled_features).flatten()
    similar_indices = similarities.argsort()[::-1][1:n+1]
    # Clean artist names
    recs = df.iloc[similar_indices][['name', 'artists', 'popularity']].copy()
    recs['artists'] = recs['artists'].apply(lambda x: ", ".join(eval(x)) if isinstance(x, str) and x.startswith("[") else x)
    return recs










































        