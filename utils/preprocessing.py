# 1. Install & import necessary packages
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 2. Load data and basic exploration
@st.cache_data
def load_data():
    st.write("Loading app...")
    df = pd.read_csv('data/Expresso_churn_dataset.csv')

    print(df.head())
    print(df.info())
    print(df.describe())
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    return df

# 3. Profiling Report (optional)
'''profile = ProfileReport(df, title="Expresso Churn Profiling Report", explorative=True)
profile.to_file("profiling_report.html")'''

import pandas as pd

def preprocess_input(df):
    # Make a copy of the dataframe to avoid modifying original data
    df = df.copy()

    # Detect mode: training if CHURN is present
    is_training = 'CHURN' in df.columns

    # Step 1: Drop unused columns
    df.drop(columns=['ZONE1', 'ZONE2'], inplace=True, errors='ignore')

    # Step 2: Fill missing values for categorical features
    df['REGION'] = df.get('REGION', pd.Series(dtype='object')).fillna('UNKNOWN')
    df['TOP_PACK'] = df.get('TOP_PACK', pd.Series(dtype='object')).fillna('UNKNOWN')

    # Step 3: Handle numeric features
    num_cols = [
        'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE',
        'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'FREQ_TOP_PACK', 'REGULARITY'
    ]

    for col in num_cols:
        if col in df.columns:
            # Avoid median() crash if values are non-numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = 0
        else:
            df[col] = 0

    # Step 4: Encode 'TENURE' and 'ARPU_SEGMENT'
    tenure_mapping = {
        'D 3-6 month': 1, 'E 6-9 month': 2, 'F 9-12 month': 3,
        'G 12-15 month': 4, 'H 15-18 month': 5, 'I 18-21 month': 6,
        'J 21-24 month': 7, 'K > 24 month': 8
    }
    arpu_mapping = {'Low': 0, 'Medium': 1, 'High': 2}

    df['TENURE'] = df.get('TENURE', pd.Series()).map(tenure_mapping).fillna(0)
    df['ARPU_SEGMENT'] = df.get('ARPU_SEGMENT', pd.Series()).map(arpu_mapping).fillna(0)

    # Step 5: One-hot encode REGION and TOP_PACK
    df = pd.get_dummies(df, columns=['REGION', 'TOP_PACK'], drop_first=False)

    # Step 6: Add any expected missing dummy columns
    expected_region_cols = [
        'REGION_CENTRE', 'REGION_EAST', 'REGION_NORTH',
        'REGION_SOUTH', 'REGION_UNKNOWN', 'REGION_WEST'
    ]
    expected_top_pack_cols = [
        'TOP_PACK_Internet', 'TOP_PACK_Orange',
        'TOP_PACK_UNKNOWN', 'TOP_PACK_Voice'
    ]

    for col in expected_region_cols + expected_top_pack_cols:
        if col not in df.columns:
            df[col] = 0

    # Step 7: Define final expected columns
    expected_cols = [
        'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT',
        'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO',
        'REGULARITY', 'FREQ_TOP_PACK'
    ] + expected_region_cols + expected_top_pack_cols

    # Only include CHURN if training
    if is_training:
        df = df[df['CHURN'].notnull()]
        expected_cols += ['CHURN']

    # Add any missing expected column (shouldn’t happen, but just in case)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Keep only expected columns, in correct order
    df = df[expected_cols].copy()
    df.reset_index(drop=True, inplace=True)

    def remove_outliers_iqr(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            before = df.shape[0]
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            after = df.shape[0]
            print(f"{col}: removed {before - after} rows")

    return df
# Preprocess the data
df_raw = load_data()

df = preprocess_input(df_raw.copy())
# Prépare Features et Target
X = df.drop(columns=['CHURN'])
y = df['CHURN']

# Contrôle rapide des shapes pour éviter erreurs
print(f"Shape X: {X.shape}")
print(f"Shape y: {y.shape}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Classifier
clf = RandomForestClassifier(random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# 11. Evaluate Model
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Sauvegarder le modèle entraîné
joblib.dump(clf, "model/churn_model.pkl",compress=3)
print("✅ Modèle sauvegardé dans model/churn_model.pkl")

