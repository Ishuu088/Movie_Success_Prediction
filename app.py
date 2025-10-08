
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(page_title="🎬 Movie Success Predictor", layout="centered")
st.title("🎥 Movie Success Predictor")

# ------------------------------
# LOAD MODEL AND FEATURES
# ------------------------------
try:
    with open('movie_hit_model.pkl', 'rb') as f:
        model, label_encoder, features = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ------------------------------
# LOAD ORIGINAL DATA (for options)
# ------------------------------
uploaded_data = st.file_uploader("📂 Upload your original CSV (to fetch director, genres, etc.)", type=["csv"])
if uploaded_data is not None:
    df = pd.read_csv(uploaded_data, encoding="latin1")
else:
    st.info("Please upload your movie dataset CSV file to continue.")
    st.stop()

# ------------------------------
# BUILD INPUT FORM
# ------------------------------
st.subheader("🎬 Enter Movie Details")

# Dropdowns and sliders based on CSV data
budget = st.slider("💰 Budget", int(df['budget'].min()), int(df['budget'].max()), int(df['budget'].mean()))
gross = st.slider("💵 Gross", int(df['gross'].min()), int(df['gross'].max()), int(df['gross'].mean()))
duration = st.slider("⏱ Duration (minutes)", int(df['duration'].min()), int(df['duration'].max()), int(df['duration'].mean()))
director = st.selectbox("🎬 Director Name", sorted(df['director_name'].dropna().unique()))
genres = st.selectbox("🎭 Genres", sorted(df['genres'].dropna().unique()))
language = st.selectbox("🗣 Language", sorted(df['language'].dropna().unique()))
country = st.selectbox("🌍 Country", sorted(df['country'].dropna().unique()))
content_rating = st.selectbox("🔞 Content Rating", sorted(df['content_rating'].dropna().unique()))
actor_rating = st.slider("⭐ Lead Actor Popularity (1–10)", 1, 10, 5)

# ------------------------------
# PROCESS INPUT
# ------------------------------
st.subheader("🧩 Prediction Input Summary")
st.write({
    "budget": budget,
    "gross": gross,
    "duration": duration,
    "director_name": director,
    "genres": genres,
    "language": language,
    "country": country,
    "content_rating": content_rating,
    "actor_1_facebook_likes (scaled 1–10)": actor_rating
})

# Create input DataFrame
X_input = pd.DataFrame([{
    "budget": budget,
    "gross": gross,
    "duration": duration,
    "director_facebook_likes": 0,  # optional dummy values
    "actor_1_facebook_likes": actor_rating * 100,  # scaled up
    "num_user_for_reviews": 50,
    "num_voted_users": 1000,
    "imdb_score": 7.0
}])

# ------------------------------
# MAKE PREDICTION
# ------------------------------
if st.button("🎯 Predict Movie Success"):
    try:
        prediction = model.predict(X_input)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        st.success(f"🎬 **Predicted Outcome:** {predicted_label}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
