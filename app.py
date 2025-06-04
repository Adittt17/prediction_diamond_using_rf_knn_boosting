import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Load models, scaler, pca, feature columns
models = {
    'Random Forest': joblib.load('model_rf.joblib'),
    'KNN': joblib.load('model_knn.joblib'),
    'AdaBoost': joblib.load('model_adaboost.joblib')
}
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')
with open('feature_columns.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

numerical_features = ['carat', 'table', 'dimension']

# Valid categories matching training data
cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_options = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_options = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

st.title("💎 Diamond Price Predictor")

st.markdown("""
### Choose your model:
Select one of the three algorithms to predict diamond price.
- **Random Forest**: Best accuracy with lowest MSE.
- **K-Nearest Neighbors (KNN)**: Simple, interpretable.
- **AdaBoost**: Boosting method, usually slower and less accurate here.
""")

model_choice = st.selectbox("Select Model", list(models.keys()))

st.markdown("---")
st.header("Input Diamond Features")

col1, col2, col3 = st.columns(3)

with col1:
    carat = st.number_input("Carat (0.2 - 5.01)", 0.2, 5.01, 1.0, 0.01)
    cut = st.selectbox("Cut Quality", cut_options)

with col2:
    color = st.selectbox("Color (J worst - D best)", color_options)
    x = st.number_input("Length (x) in mm (0 - 10.74)", 0.0, 10.74, 5.0, 0.01)

with col3:
    clarity = st.selectbox("Clarity", clarity_options)
    y = st.number_input("Width (y) in mm (0 - 58.9)", 0.0, 58.9, 5.0, 0.01)

z = st.number_input("Depth (z) in mm (0 - 31.8)", 0.0, 31.8, 3.0, 0.01)
depth = 2 * z / (x + y) if (x + y) != 0 else 0.0
table = st.number_input("Table (%) (43 - 95)", 43, 95, 55, 1)

# Preprocess input
dim = pca.transform(np.array([[x, y, z]]))[0, 0]

input_df = pd.DataFrame(columns=feature_cols)
input_df.loc[0] = 0

input_df.loc[0, 'carat'] = carat
input_df.loc[0, 'table'] = table
input_df.loc[0, 'dimension'] = dim

for cat, val in [('cut', cut), ('color', color), ('clarity', clarity)]:
    col_name = f"{cat}_{val}"
    if col_name in feature_cols:
        input_df.loc[0, col_name] = 1

input_df[numerical_features] = scaler.transform(input_df[numerical_features])

if st.button("Predict Price"):
    model = models[model_choice]
    pred = model.predict(input_df)[0]
    st.success(f"💰 Predicted Price using **{model_choice}**: ${pred:,.2f}")

st.markdown("---")
st.header("Model Evaluation Summary (MSE)")

st.markdown("""
| Model          | Train MSE | Test MSE | Notes                        |
|----------------|-----------|----------|------------------------------|
| KNN            | 203.76    | 239.53   | Simple, decent but less accurate |
| Random Forest  | 52.29     | 130.79   | Best performer, lowest error  |
| AdaBoost       | 904.84    | 846.21   | Higher error, less accurate    |

**MSE (Mean Squared Error)** measures average squared difference between predicted and true prices — lower is better.  
Random Forest is best here, meaning it predicts prices closest to actual values.
""")

st.markdown("---")
st.header("Why Input Limits?")

st.markdown("""
We restrict numerical inputs to ranges seen during training to:
- **Avoid extrapolation errors:** Models perform poorly on data too different from training.
- **Maintain prediction reliability:** Inputs outside these bounds might produce unrealistic results.
- **Prevent app crashes:** Out-of-range or nonsensical values can break preprocessing or prediction steps.

These limits ensure the model stays in its "comfort zone" for accurate and stable predictions.
""")
