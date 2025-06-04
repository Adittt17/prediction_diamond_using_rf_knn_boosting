import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Load model, scaler, pca, fitur kolom
model = joblib.load('model_rf.joblib')
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')
with open('feature_columns.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

numerical_features = ['carat', 'table', 'dimension']  # sesuai training

# Pilihan kategori yang valid (harus sesuai data training)
cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_options = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_options = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

st.title("Diamond Price Prediction")

# Input user
carat = st.number_input("Carat (0.2 - 5.01)", min_value=0.2, max_value=5.01, value=1.0, step=0.01)
cut = st.selectbox("Cut Quality", cut_options)
color = st.selectbox("Color (J worst - D best)", color_options)
clarity = st.selectbox("Clarity", clarity_options)
x = st.number_input("Length (x) in mm (0 - 10.74)", min_value=0.0, max_value=10.74, value=5.0, step=0.01)
y = st.number_input("Width (y) in mm (0 - 58.9)", min_value=0.0, max_value=58.9, value=5.0, step=0.01)
z = st.number_input("Depth (z) in mm (0 - 31.8)", min_value=0.0, max_value=31.8, value=3.0, step=0.01)
depth = 2 * z / (x + y) if (x + y) != 0 else 0.0
table = st.number_input("Table (%) (43 - 95)", min_value=43, max_value=95, value=55, step=1)

# Preprocessing input

# Hitung dimension dari PCA x,y,z
dim = pca.transform(np.array([[x, y, z]]))[0, 0]

# Buat DataFrame kosong dengan semua kolom fitur yang dipakai model
input_df = pd.DataFrame(columns=feature_cols)
input_df.loc[0] = 0  # init 0 semua

# Masukkan fitur numerik
input_df.loc[0, 'carat'] = carat
input_df.loc[0, 'table'] = table
input_df.loc[0, 'dimension'] = dim

# Isi dummy encoding fitur kategorikal
# Contoh: 'cut_Fair', 'color_J', 'clarity_I1' dll harus ada di feature_cols
cut_col = 'cut_' + cut
color_col = 'color_' + color
clarity_col = 'clarity_' + clarity

if cut_col in feature_cols:
    input_df.loc[0, cut_col] = 1
if color_col in feature_cols:
    input_df.loc[0, color_col] = 1
if clarity_col in feature_cols:
    input_df.loc[0, clarity_col] = 1

# Scaling fitur numerik
input_df[numerical_features] = scaler.transform(input_df[numerical_features])

# Prediksi
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Diamond Price: ${prediction:,.2f}")

# Tampilkan info MSE evaluasi model (example)
st.markdown("""
### Model Evaluation (MSE)
- KNN: 203.76 (train), 239.53 (test)
- Random Forest: 52.29 (train), 130.79 (test) **<- Best**
- AdaBoost: 904.84 (train), 846.21 (test)
  
Lower MSE means better model performance on that dataset.
Random Forest has lowest MSE, so generally more accurate.
""")
