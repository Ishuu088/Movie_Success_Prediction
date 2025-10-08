import streamlit as st
import pandas as pd



import os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from collections import Counter
import joblib, json

        

df = pd.read_csv('movie_metadata.csv')
print('Shape:', df.shape)

# Preprocessing similar to notebook (create Classify, drop imdb_score from features)
df['imdb_score'] = pd.to_numeric(df['imdb_score'], errors='coerce')
def categorize(score):
    if pd.isna(score): return np.nan
    if score <= 3.0: return 'Flop'
    if score <= 6.0: return 'Average'
    return 'Hit'
df['Classify'] = df['imdb_score'].apply(categorize)
df = df[~df['Classify'].isna()].reset_index(drop=True)

cols = ['duration','director_name','director_facebook_likes',
        'actor_1_name','actor_1_facebook_likes','actor_2_name','actor_2_facebook_likes',
        'actor_3_name','actor_3_facebook_likes','num_user_for_reviews','num_critic_for_reviews',
        'num_voted_users','cast_total_facebook_likes','movie_facebook_likes','plot_keywords',
        'facenumber_in_poster','color','genres','title_year','language','country','content_rating',
        'aspect_ratio','gross','budget','Classify']
cols = [c for c in cols if c in df.columns]
df = df[cols].copy()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

def top_k_replace(series, k=15):
    topk = series.fillna('Unknown').value_counts().nlargest(k).index
    return series.fillna('Unknown').where(series.fillna('Unknown').isin(topk), 'Other')

for c in ['director_name','actor_1_name','actor_2_name','actor_3_name']:
    if c in df.columns:
        df[c] = top_k_replace(df[c], k=15)

if 'genres' in df.columns:
    df['genres_first'] = df['genres'].fillna('Unknown').apply(lambda x: str(x).split('|')[0])
    df['genres_first'] = top_k_replace(df['genres_first'], k=12)
if 'plot_keywords' in df.columns:
    kw_series = df['plot_keywords'].dropna().astype(str).str.split('|').explode()
    top_kw = kw_series.value_counts().nlargest(30).index.tolist()
    for kw in top_kw:
        df[f'kw_{kw}'] = df['plot_keywords'].fillna('').apply(lambda s: 1 if kw in str(s).split('|') else 0)

for c in ['content_rating','language','country','color']:
    if c in df.columns:
        df[c] = df[c].fillna('Unknown')
if 'aspect_ratio' in df.columns:
    df['aspect_ratio'] = pd.to_numeric(df['aspect_ratio'], errors='coerce').fillna(df['aspect_ratio'].median())

base_numeric = [
    'duration','director_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes',
    'actor_3_facebook_likes','num_user_for_reviews','num_critic_for_reviews','num_voted_users',
    'cast_total_facebook_likes','movie_facebook_likes','facenumber_in_poster','title_year',
    'aspect_ratio','gross','budget'
]
final_numeric = [c for c in base_numeric if c in df.columns]
cat_small = [c for c in ['genres_first','content_rating','language','country','color','director_name','actor_1_name','actor_2_name','actor_3_name'] if c in df.columns]

model_df = df[final_numeric + cat_small + [c for c in df.columns if c.startswith('kw_')] + ['Classify']].copy()
model_df = pd.get_dummies(model_df, columns=cat_small, drop_first=True)

X = model_df.drop(columns=['Classify'])
y = model_df['Classify']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35, stratify=y)

# scale numeric columns
numeric_present = [c for c in final_numeric if c in X.columns]
scaler = StandardScaler()
X_train[numeric_present] = scaler.fit_transform(X_train[numeric_present])
X_test[numeric_present] = scaler.transform(X_test[numeric_present])

print('Train class counts:', Counter(y_train))

# SMOTE if available
try:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    print('After SMOTE:', Counter(y_train_bal))
except Exception as e:
    print('SMOTE not available or failed:', e)
    X_train_bal, y_train_bal = X_train, y_train

# GridSearchCV for RandomForest
param_grid = {'n_estimators':[100,200], 'max_depth':[None,10], 'min_samples_split':[2,5]}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
gs = GridSearchCV(RandomForestClassifier(random_state=37, class_weight='balanced', n_jobs=-1), param_grid, cv=cv, scoring='f1_macro', verbose=1)
gs.fit(X_train_bal, y_train_bal)
best_rf = gs.best_estimator_
print('Best RF params:', gs.best_params_)
print('Best RF report:'); print(classification_report(y_test, best_rf.predict(X_test)))



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
