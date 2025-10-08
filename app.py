import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Define paths for artifacts - adjust if not using /tmp
model_path = '/tmp/movie_prediction_artifacts/best_movie_model.joblib'
scaler_path = '/tmp/movie_prediction_artifacts/scaler.joblib'
cols_path = '/tmp/movie_prediction_artifacts/model_feature_columns.json'


if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(cols_path):
    st.error('Model artifacts not found. Please ensure the training cell saved artifacts to /tmp/movie_prediction_artifacts.')
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(cols_path, 'r') as f:
        model_cols = json.load(f)

    # Load your full dataset (to get dropdown options and max/min values)
    # Corrected path to load the original movie_metadata.csv
    data_path = '/content/movie_metadata.csv'
    try:
        df = pd.read_csv(data_path, encoding='latin1')
    except FileNotFoundError:
        st.error(f"Error: Original data file not found at {data_path}. Please ensure movie_metadata.csv is uploaded or available.")
        st.stop() # Stop execution if data is not found


    st.header("🎞️ Enter Movie Details")

    # --- Dropdowns and Sliders based on CSV ---
    # Add checks for column existence before accessing
    budget_min, budget_max, budget_median = 0, 1000000000, 20000000 # Default values in case column is missing
    if 'budget' in df.columns and pd.api.types.is_numeric_dtype(df['budget']):
        budget_min, budget_max, budget_median = int(df['budget'].min()), int(df['budget'].max()), int(df['budget'].median())

    budget = st.slider(
        "💰 Budget",
        budget_min,
        budget_max,
        budget_median
    )

    gross_min, gross_max, gross_median = 0, 500000000, 25000000 # Default values
    if 'gross' in df.columns and pd.api.types.is_numeric_dtype(df['gross']):
        gross_min, gross_max, gross_median = int(df['gross'].min()), int(df['gross'].max()), int(df['gross'].median())

    gross = st.slider(
        "💵 Gross",
        gross_min,
        gross_max,
        gross_median
    )

    actor_rating = st.slider("⭐ Actor (Facebook Likes Rating 1–10)", 1, 10, 5)

    duration_min, duration_max, duration_median = 0, 300, 105 # Default values
    if 'duration' in df.columns and pd.api.types.is_numeric_dtype(df['duration']):
         duration_min, duration_max, duration_median = int(df['duration'].min()), int(df['duration'].max()), int(df['duration'].median())

    duration = st.slider(
        "⏱️ Duration (minutes)",
        duration_min,
        duration_max,
        duration_median
    )

    # Add checks for column existence and handle missing values before unique()
    genres_options = []
    if 'genres' in df.columns:
        genres_options = sorted(df['genres'].dropna().unique())
    genres = st.selectbox("🎭 Genres", genres_options)

    language_options = []
    if 'language' in df.columns:
        language_options = sorted(df['language'].dropna().unique())
    language = st.selectbox("🗣️ Language", language_options)

    country_options = []
    if 'country' in df.columns:
        country_options = sorted(df['country'].dropna().unique())
    country = st.selectbox("🌍 Country", country_options)

    content_rating_options = []
    if 'content_rating' in df.columns:
         content_rating_options = sorted(df['content_rating'].dropna().unique())
    content_rating = st.selectbox("🔞 Content Rating", content_rating_options)


    # --- Prepare Input DataFrame ---
    input_dict = {
        'budget': budget,
        'gross': gross,
        'actor_1_facebook_likes': actor_rating * (df['actor_1_facebook_likes'].max() / 10) if 'actor_1_facebook_likes' in df.columns else 0, # scaled
        'duration': duration,
        'genres': genres, # Keep original genre string for parsing
        'language': language,
        'country': country,
        'content_rating': content_rating
    }


    # --- Replicate Preprocessing Steps ---
    try:
        # Create a DataFrame from input_dict
        input_df_processed = pd.DataFrame([input_dict])

        # Apply the same preprocessing steps as the training notebook (simplified)
        # This part needs to be more robust to fully replicate the training preprocessing

        # Handle genres - extract first genre and apply top-k
        if 'genres' in input_df_processed.columns and 'genres_first' in model_cols:
             input_df_processed['genres_first'] = input_df_processed['genres'].fillna('Unknown').apply(lambda x: str(x).split('|')[0])
             # Need to replicate top_k_replace logic for genres_first here if it was applied in training

        # Handle categorical features with one-hot encoding - this requires knowing the categories the model was trained on
        # A more robust solution would involve saving the fitted OneHotEncoder or the list of categories for each feature
        # For simplicity here, we'll manually create dummy columns based on the input values if they are in model_cols
        categorical_input_cols = ['genres_first', 'language', 'country', 'content_rating'] # Add other categorical columns as needed

        # Create a DataFrame with all model columns, filled with 0
        X_input = pd.DataFrame(0, index=[0], columns=model_cols)

        # Populate numeric columns and one-hot encoded categorical columns
        for col in model_cols:
            if col in input_df_processed.columns:
                # Handle numeric columns
                if np.issubdtype(X_input[col].dtype, np.number):
                    X_input[col] = input_df_processed[col].iloc[0]
            elif col.startswith(tuple([c + '_' for c in categorical_input_cols])):
                 # Handle one-hot encoded columns (simplified - assuming a direct match)
                 # This is basic and needs improvement for robust handling of categories not seen in training
                 original_col = col.split('_', 1)[0]
                 category_value = col.split('_', 1)[1]
                 if original_col in input_df_processed.columns and input_df_processed[original_col].iloc[0] == category_value:
                     X_input[col] = 1
                 # Special handling for genres_first as it's derived
                 elif original_col == 'genres_first' and 'genres' in input_df_processed.columns:
                     first_genre = str(input_df_processed['genres'].iloc[0]).split('|')[0]
                     if first_genre == category_value:
                         X_input[col] = 1



        # Scale numeric columns - ensure only numeric columns present in the scaler are scaled
        numeric_cols_to_scale = [c for c in model_cols if np.issubdtype(X_input[c].dtype, np.number) and c in scaler.feature_names_in_]
        if len(numeric_cols_to_scale) > 0:
             X_input[numeric_cols_to_scale] = scaler.transform(X_input[numeric_cols_to_scale])


        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0].max()

        st.success(f"🎯 Prediction: {pred} (Confidence: {prob:.2f})")
    except Exception as e:
        st.error(f"Prediction failed during preprocessing or prediction: {e}")
        st.exception(e) # Display full traceback for debugging