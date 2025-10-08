import streamlit as st
import pandas as pd, numpy as np, joblib, json, os
pip install streamlit pandas numpy joblib scikit-learn


streamlit
pandas
numpy
joblib
scikit-learn

st.set_page_config(page_title='Movie Success Predictor', layout='centered')
st.title('🎬 Movie Success Predictor (CSV upload)')

# Define paths for artifacts - adjust if not using /tmp
model_path = '/tmp/movie_prediction_artifacts/best_movie_model.joblib'
scaler_path = '/tmp/movie_prediction_artifacts/scaler.joblib'
cols_path = '/tmp/movie_prediction_artifacts/model_feature_columns.json'


if not os.path.exists(model_path):
    st.error('Model not found. Please ensure the training cell saved artifacts to /tmp/movie_prediction_artifacts.')
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(cols_path,'r') as f:
        model_cols = json.load(f)

    uploaded = st.file_uploader('Upload processed CSV (matching model features)', type='csv')
    if uploaded is not None:
        df = pd.read_csv(uploaded, encoding='latin1')
        st.write(df.head())
        try:
            X = df[model_cols].copy()
            num_cols = [c for c in model_cols if c in X.columns and np.issubdtype(X[c].dtype, np.number)]
            if len(num_cols)>0:
                X[num_cols] = scaler.transform(X[num_cols])
            preds = model.predict(X)
            probs = model.predict_proba(X)
            df['Predicted_Class'] = preds
            df['Pred_Prob'] = [p.max() for p in probs]
            st.dataframe(df.head(50))
            st.download_button('Download predictions', df.to_csv(index=False).encode('utf-8'), 'preds.csv')
        except Exception as e:
            st.error('Prediction failed: '+str(e))
