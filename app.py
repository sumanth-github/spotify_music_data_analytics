import streamlit as st
st.set_page_config(page_title="Spotify Music Trend Analyzer", layout="wide")

import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import joblib
from utils import (
    get_playlist_tracks, 
    parse_tracks,load_country_charts,plot_actual_vs_predicted,plot_artist_track_popularity,
    load_genre_data,get_top_tracks_by_country,plot_actual_vs_predicted,
    plot_top_genres_by_popularity,evaluate_random_forest_model,get_artist_top_tracks,
    plot_audio_feature_radar,plot_rf_feature_importance,plot_average_popularity,
    plot_valence_vs_energy,plot_model_comparison,train_test_split,
    plot_danceability_vs_energy,train_linear_regression,plot_top_explorer_artists,
    plot_popularity_distribution,plot_average_popularity_over_years,
    plot_track_popularity, plot_artist_album_timeline,load_tracks_data,
    plot_top_artists, plot_duration_distribution, get_top_tracks, get_artist_albums, explore_artist)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load your dataset
track_df = pd.read_csv("spotify_dataset/tracks.csv")

# Filter and clean up
track_df['release_date'] = pd.to_datetime(track_df['release_date'], errors='coerce')
track_df = track_df.dropna(subset=['release_date'])
track_df['year'] = track_df['release_date'].dt.year
track_df = track_df[(track_df['year'] >= 1950) & (track_df['year'] <= 2020)]

# Features and target
features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'speechiness', 'tempo', 'valence']
X = track_df[features]
y = track_df['popularity']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Load ML model
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
y_pred_lin = lin_model.predict(X_test_scaled)

# Random Forest loading or training
rf_model = joblib.load("random_forest_model.pkl")  # or train here if needed
y_pred_rf = rf_model.predict(X_test_scaled)

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lin_mae = mean_absolute_error(y_test, y_pred_lin)
lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lin))
lin_r2 = r2_score(y_test, y_pred_lin)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

# 💾 Save models and scaler
joblib.dump(lin_model, "linear_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# ✅ Store metrics and scaled test data in session state
st.session_state['lin_metrics'] = {
    "mae": lin_mae,
    "rmse": lin_rmse,
    "r2": lin_r2
}
st.session_state['rf_metrics'] = {
    "mae": rf_mae,
    "rmse": rf_rmse,
    "r2": rf_r2
}
st.session_state['X_test_scaled'] = X_test_scaled
#st.session_state["y_pred_lin"] = y_pred_lin
#st.session_state["y_pred_rf"] = y_pred_rf
#st.session_state["y_test"] = y_test

# Load dataset (you can switch this to a cached version later)
@st.cache_data
def load_data():
    df = pd.read_csv(r"spotify_dataset\tracks.csv")
    return df

df = load_data()

def genre_analytics_section(dfg=None):
    st.subheader("🎼 Genre Analytics Dashboard")

    if dfg is None:
        dfg = load_genre_data()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎧 Top Genres by Popularity",
        "📊 Audio Features by Genre (Radar)",
        "🌀 Valence vs Energy (Top 500)",
        "🎯 Danceability vs Energy (Top Genres)"
    ])

    with tab1:
        st.header("🎵 Top 10 Most Popular Genres")
        with st.expander("ℹ️ What does this show and why?"):
            st.markdown("""
            This chart displays the **top music genres based on their average popularity**.
        
            **🎯 What is it?**
            - We're calculating the average popularity of songs within each genre.
            - Popularity is a Spotify metric (0 to 100) based on play count, recentness, and engagement.
        
            **🧠 Why is it useful?**
            - Shows which genres dominate streaming platforms.
            - Great for spotting trends in listener preferences.
        
            **🔍 Use case:**
            - Understand mainstream genre trends.
            - Discover which genres are growing in popularity over time.
            """)

        top_genres_df = plot_top_genres_by_popularity(dfg)

    with tab2:
        st.header("📊 Audio Feature Profile by Genre")
        with st.expander("ℹ️ What does this chart represent?"):
            st.markdown("""
            This **radar chart compares audio characteristics** across top music genres.
        
            **🎯 What is it?**
            - Each spoke on the chart represents a musical quality: danceability, energy, valence, etc.
            - We're showing average values of these features per genre.
        
            **🧠 Why is it useful?**
            - Gives a **sonic fingerprint** of each genre.
            - See which genres are more energetic, emotional, or acoustic.
        
            **🔍 Use case:**
            - Explore how musical elements differ across genres.
            - Helpful for music producers, playlist curators, and data enthusiasts.
            """)

        top_genres = top_genres_df['genre'].head(5).tolist()
        plot_audio_feature_radar(dfg, top_genres)

    with tab3:
        st.header("🌀 Valence vs Energy - Top 500 Tracks")
        with st.expander("ℹ️ What do Valence and Energy mean?"):
            st.markdown("""
            **🎭 Valence**  
            Represents the *musical positiveness* of a track.  
            - **High Valence (→ 1.0):** Happy, cheerful, euphoric tracks (*e.g., "Happy" by Pharrell*)  
            - **Low Valence (→ 0.0):** Sad, depressed, or angry tracks (*e.g., "Someone Like You" by Adele*)
        
            **⚡ Energy**  
            Describes the *intensity and activity level* of a song.  
            - **High Energy:** Loud, fast, and intense tracks (*e.g., EDM, hard rock*)  
            - **Low Energy:** Calm, mellow, and gentle tracks (*e.g., acoustic ballads, lo-fi beats*)
        
            This chart helps you **visualize the emotional mood** of music by genre.
            """)
        plot_valence_vs_energy(dfg)

    with tab4:
        st.header("🎯 Danceability vs Energy - Top Genres")
        with st.expander("ℹ️ What does this chart show?"):
            st.markdown("""
            This scatter plot maps **danceability vs energy** for top music genres.
        
            **🎯 What is it?**
            - Each dot is a track. Its position tells how danceable and energetic it is.
            - Tracks are grouped by genre for pattern spotting.
        
            **🧠 Why is it useful?**
            - Helps discover which genres lean toward party vibes vs chill tones.
            - Visualizes the balance between movement and intensity.
        
            **🔍 Use case:**
            - Choose tracks for workouts, parties, or relaxing based on genre behavior.
            """)

        plot_danceability_vs_energy(dfg)
        
st.title("🎵 Spotify Music Trend Analyzer")
st.markdown("""
    <hr style='border: 1px solid #bbb; margin-top: 0.25rem; margin-bottom: 1.5rem;'>
""", unsafe_allow_html=True)
st.sidebar.markdown("## 🎵 Navigation Panel")
st.sidebar.markdown("---")
st.markdown("""
    <style>
    /* Sidebar title styling */
    section[data-testid="stSidebar"] .css-1d391kg, 
    section[data-testid="stSidebar"] .css-1v0mbdj {
        font-weight: bold;
        font-size: 1.2rem;
        color: #1DB954;
        margin-bottom: 1rem;
    }

    /* Add vertical spacing between radio options */
    section[data-testid="stSidebar"] .stRadio > div {
        row-gap: 0.75rem !important;
    }

    /* Optional: customize the labels */
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        font-weight: 500;
    }

    /* Hover effect for expanders if any */
    .streamlit-expanderHeader:hover {
        color: #1DB954;
    }
    </style>
""", unsafe_allow_html=True)



# Sidebar navigation
section = st.sidebar.radio("",
    [
        "🏠 Home",
        "📂 Upload Playlist", 
        "🎨 Artist Explorer", 
        "📊 Genre Analytics",
        "🌍 Top Tracks by Country",
        "🎯 Recommendation Engine", 
        "🧠 ML Popularity Predictor"
    ]
)
st.sidebar.markdown("<br><br><br><br>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("👤 **Built by:** Sumanth")
st.sidebar.markdown("📅 **Last updated:** May 2025")


# 🏠 Home
if section == "🏠 Home":
    st.markdown("## Welcome to the Ultimate Spotify Analysis App 🚀")
    st.markdown("Analyze playlists, explore artists, visualize trends, and even **predict popularity** with ML! 🎯")

    st.markdown("### 🎯 Purpose of the Project")
    st.markdown("""
    This project is built to **analyze, visualize, and predict music trends** using Spotify's extensive audio data.

    **Core objectives:**
    - Understand what makes a track popular through audio features.
    - Track changes in music taste across decades and genres.
    - Help artists and curators explore music patterns and behaviors.
    - Predict song popularity using machine learning.
    - Recommend songs based on acoustic similarity.
    """)

    st.markdown("### 🌍 Real-World Applications")
    st.markdown("""
    This system can benefit a wide variety of stakeholders:

    - 🎧 **Streaming Services**: Improve recommendation algorithms.
    - 📊 **Record Labels & Music Marketers**: Forecast hit potential.
    - 🎙️ **Artists & Producers**: Discover what elements drive success.
    - 🧠 **Data Scientists**: Use it as a sandbox for music-based ML projects.
    - 📚 **Educators**: Teach data analytics with real-world use cases.
    - 🎥 **Content Creators**: Match music with mood or theme effectively.
    """)



elif section == "📂 Upload Playlist":
    st.markdown("### Enter Spotify Playlist ID or Full URL:")
    user_input = st.text_input("Paste your Playlist ID or URL below",help="You can paste either a full Spotify URL or just the playlist ID")
    if st.button("Analyze Playlist") and user_input:
        from utils import extract_playlist_id

        # Check and extract playlist ID
        playlist_id = extract_playlist_id(user_input)
        tracks = get_playlist_tracks(playlist_id)
        df_playlist = parse_tracks(tracks)
        st.dataframe(df_playlist)

        # Import and show visualizations
        from utils import (
            plot_popularity_distribution, 
            plot_track_popularity, 
            plot_top_artists
        )

        st.subheader("📊 Playlist Insights")

        st.markdown("### 🎯 Popularity Distribution")
        fig1 = plot_popularity_distribution(df_playlist)
        st.pyplot(fig1)
        st.markdown("### 🌟 Track Popularity")
        fig2 = plot_track_popularity(df_playlist)


        st.markdown("### 👑 Top Artists by Number of Tracks")
        fig3 = plot_top_artists(df_playlist)

# 🎤 Artist Explorer
elif section == "🎨 Artist Explorer":
    st.subheader("🎧 Artist Explorer")

    # Load Dataset (Cached)
    @st.cache_data
    def load_artist_data():
        return pd.read_csv(r"spotify_dataset/tracks.csv")

    df = load_artist_data()

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 General Insights",
        "🧑‍🎤 Artist Explorer",
        "🔥 Top Tracks",
        "📀 Albums"
    ])

    # ------------------------ Tab 1: General Insights ------------------------
    with tab1:
        
        st.header("📊 General Track Insights")
    
        with st.expander("🎧 Top Tracks by Popularity"):
            st.markdown("Shows the most popular tracks in the dataset based on their Spotify popularity score.")
        plot_artist_track_popularity(df)
    
        with st.expander("⏱ Track Duration Distribution"):
            st.markdown("Displays the distribution of track durations in minutes to understand common track lengths.")
        plot_duration_distribution(df)
    
        with st.expander("🔥 Top Artists by Track Count"):
            st.markdown("Lists the artists with the highest number of tracks in the dataset.")
        plot_top_explorer_artists(df)



    # ------------------------ Tab 2: Artist Explorer ------------------------
    with tab2:
        st.header("🧑‍🎤 Explore Any Artist")
        artist_input = st.text_input("Enter artist name (Case Sensitive):", key="artist_explorer_input")

        if st.button("Explore", key="explore_artist_button"):
            if artist_input.strip() == "":
                st.warning("🚨 Please enter an artist name before exploring!")
            else:
                explore_artist(artist_input, df)

    # ------------------------ Tab 3: Top Tracks ------------------------
    with tab3:
        st.header("🔥 Top Tracks by Popularity")
        with st.expander("ℹ️ What does this section do?"):
            st.markdown("""
            This section lists the **most popular tracks** in the dataset.
        
            **🎯 What is it?**
            - Tracks are sorted by popularity score (Spotify API metric).
            - You can customize how many top tracks to view.
        
            **🧠 Why is it useful?**
            - Lets you explore what listeners love most.
            - Great for quickly identifying trends or viral hits.
        
            **🔍 Use case:**
            - Spot breakout songs by genre, artist, or audio feature.
            """)

    
        # 🔢 User input for number of top tracks (default to 20)
        n = st.number_input("Select number of top tracks to display:", min_value=5, max_value=100, value=20, step=5)
        st.subheader("🎵 Top Tracks by Popularity (with Artists)")
        top_tracks = get_artist_top_tracks(df, top_n=n)
    
        # 📊 Wider column display using streamlit dataframe styling
        st.dataframe(
            top_tracks.style.set_properties(**{'width': '300px'}),
            use_container_width=True
        )

    # ------------------------ Tab 4: Artist Albums ------------------------
    with tab4:
        st.header("📀 Artist Albums & Timeline")
    
        artist_album_input = st.text_input("Enter artist for album timeline:", key="album_input")
        timeline = st.slider("Select number of albums to show in timeline:", min_value=5, max_value=50, value=15, step=1)
    
        if st.button("Show Albums & Timeline", key="show_album_button"):
            if artist_album_input.strip() == "":
                st.warning("📛 Please enter an artist name to see albums.")
            else:
                albums = utils.get_artist_albums(artist_album_input, df)
                if not albums.empty:
                    st.success(f"✅ Found {len(albums)} pseudo-albums.")
                    st.dataframe(albums)
                    utils.plot_artist_album_timeline(artist_album_input, df, max_albums=timeline)
                else:
                    st.warning("🤷 No albums found for that artist.")


elif section == "📊 Genre Analytics":
    dfg = load_genre_data()
    genre_analytics_section(dfg)  # You can pass the full df or features.csv-specific df

elif section == "🌍 Top Tracks by Country":
    st.subheader("🌍 Top Tracks by Country")

    st.markdown("""
        This section displays the most streamed tracks in different countries based on Spotify's weekly charts.
        It's a great way to discover what's trending **globally and locally** 🌐🎶
    """)

    # Load the dataset
    country_df = load_country_charts()

    # Dropdown input for country selection
    country_list = sorted(country_df["country"].unique())
    selected_country = st.selectbox("Select a country:", country_list)

    # Slider for top N
    top_n = st.slider("Number of top tracks to show", min_value=5, max_value=200, value=10)

    # Show top tracks
    result = get_top_tracks_by_country(country_df, selected_country, n=top_n)

    if not result.empty:
        st.success(f"Top {top_n} tracks in {selected_country}:")
        st.dataframe(result, use_container_width=True)
    else:
        st.warning(f"No tracks found for {selected_country}.")



# 🤖 ML Popularity Predictor
elif section == "🧠 ML Popularity Predictor": 
    st.header("🎯 Predict Song Popularity with ML Models")
    st.markdown("""
    Use machine learning to estimate a song's popularity score based on its audio characteristics.  
    Choose between models, adjust the features, and get real-time predictions!
    """)

    # Load model + scaler
    rf_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")  # Make sure you saved this earlier during training

    # Define features in same order as during training
    feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                     'liveness', 'speechiness', 'tempo', 'valence']

    # Tabs: Predictor vs Visualizer
    tab1, tab2,tab3,tab4 = st.tabs(["🔮 Predictor", "📊 Model Visualizations","📈 Average Popularity Over Time","⚔️ Actual vs Predicted"])

    with tab1:
        model_choice = st.selectbox("Choose a Model", ["Random Forest", "Linear Regression"])

        st.markdown("### 🎛️ Audio Feature Input")
        st.caption("Use the sliders below to define your song's characteristics:")

        col1, col2 = st.columns(2)
        with col1:
            danceability = st.slider("🎵 Danceability", 0.0, 1.0, 0.5)
            energy = st.slider("⚡ Energy", 0.0, 1.0, 0.5)
            valence = st.slider("😊 Valence (positiveness)", 0.0, 1.0, 0.5)
            tempo = st.slider("🕒 Tempo (BPM)", 50, 200, 120)
        with col2:
            acousticness = st.slider("🎻 Acousticness", 0.0, 1.0, 0.5)
            instrumentalness = st.slider("🎼 Instrumentalness", 0.0, 1.0, 0.0)
            liveness = st.slider("📡 Liveness", 0.0, 1.0, 0.5)
            speechiness = st.slider("🗣️ Speechiness", 0.0, 1.0, 0.1)

        input_features = np.array([[acousticness, danceability, energy,
                                    instrumentalness, liveness, speechiness,
                                    tempo, valence]])
        input_scaled = scaler.transform(input_features)

        if st.button("Predict Popularity"):
            if model_choice == "Random Forest":
                pred = rf_model.predict(input_scaled)[0]
            else:
                lin_model = joblib.load("linear_model.pkl")
                pred = lin_model.predict(input_scaled)[0]

            st.success(f"🌟 Predicted Popularity Score: **{int(pred)} / 100**")

    with tab2:
        st.markdown("### 📊 Model Comparisons & Metrics")
        st.markdown("Explore how each model performs using MAE, RMSE, and R².")

        if "X_test_scaled" not in st.session_state:
            st.warning("Train your models to populate visualizations.")
        else:
            lin_metrics = st.session_state['lin_metrics']
            rf_metrics = st.session_state['rf_metrics']
            st.pyplot(plot_model_comparison(lin_metrics, rf_metrics))
            st.markdown("---")
            st.markdown("### 🔍 Feature Importance (Random Forest)")
            st.pyplot(plot_rf_feature_importance(rf_model, feature_names))
    with tab3:
        st.markdown("This chart shows how average track popularity has changed over the years.")
        fig = plot_average_popularity()
        st.pyplot(fig)
        
    with tab4:
        st.markdown("Compare the predicted and actual popularity scores using a selected ML model.")

        model_choice = st.selectbox("Choose model for scatter plot:", ["Linear Regression", "Random Forest"])
    
        if model_choice == "Linear Regression":
            y_pred = y_pred_lin
        elif model_choice == "Random Forest":
            y_pred = y_pred_rf
    
        fig = plot_actual_vs_predicted(y_test, y_pred, model_name=model_choice)
        st.pyplot(fig)

# 🎧 Recommendation Engine
elif section == "🎯 Recommendation Engine":
    from utils import recommend_tracks

    st.subheader("🎧 Music Recommendation Engine")
    st.markdown("Enter a track name to find similar songs based on audio features.")

    track_input = st.text_input("Enter a track name:")

    # Slider to choose number of recommended songs
    n_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

    # Load dataset
    if 'tracks_df' not in st.session_state:
        tracks_df = pd.read_csv("spotify_dataset/tracks.csv")
        tracks_df.columns = [col.strip().lower().replace(" ", "_") for col in tracks_df.columns]
        st.session_state['tracks_df'] = tracks_df
    else:
        tracks_df = st.session_state['tracks_df']

    # Define audio features
    audio_features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                      'liveness', 'speechiness', 'tempo', 'valence']

    # Scale features
    if 'scaled_features' not in st.session_state:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_feats = scaler.fit_transform(tracks_df[audio_features])
        st.session_state['scaled_features'] = scaled_feats
    else:
        scaled_feats = st.session_state['scaled_features']

    # Show recommendations
    if track_input:
        recs = recommend_tracks(track_input, tracks_df, scaled_feats, n=n_recommendations)
        if isinstance(recs, str):
            st.error(recs)
        else:
            st.success(f"🎯 Top {n_recommendations} Recommended Tracks:")
            st.dataframe(recs)

