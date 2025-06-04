import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load trained models
models = {
    'Random Forest': load('model_rf.joblib'),
    'KNN': load('model_knn.joblib'),
    'AdaBoost': load('model_adaboost.joblib')
}

# Load MSE data (assume you've saved it from training phase)
mse = pd.read_csv('mse_summary.csv', index_col=0)

# Set app title
st.title("Diamond Price Predictor 💎")

st.markdown("""
This app allows you to predict the price of a diamond based on its physical attributes. 
Choose a model from the sidebar, input diamond characteristics, and get the predicted price.
""")

# Sidebar: model selection
selected_model_name = st.sidebar.selectbox("Choose a model:", list(models.keys()))
model = models[selected_model_name]

# Sidebar: show MSE
st.sidebar.markdown("### Model Evaluation (MSE/1000)")
st.sidebar.dataframe(mse)
st.sidebar.markdown("""
- Lower MSE indicates better model performance.
- MSE is calculated on both training and testing datasets.
""")

# User inputs
st.subheader("Enter Diamond Features")
carat = st.slider("Carat (0.2 - 5.01)", 0.2, 5.01, 0.5)
cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox("Color", ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
x = st.slider("Length (x in mm)", 0.0, 10.74, 5.0)
y = st.slider("Width (y in mm)", 0.0, 58.9, 5.0)
z = st.slider("Depth (z in mm)", 0.0, 31.8, 3.0)
depth = st.slider("Depth (%)", 43.0, 79.0, 60.0)
table = st.slider("Table (%)", 43.0, 95.0, 57.0)

# Convert categorical to numerical (assume this is same as during training)
cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

# Build input DataFrame
input_df = pd.DataFrame({
    'carat': [carat],
    'cut': [cut_map[cut]],
    'color': [color_map[color]],
    'clarity': [clarity_map[clarity]],
    'x': [x],
    'y': [y],
    'z': [z],
    'depth': [depth],
    'table': [table]
})

# Predict
if st.button("Predict Price"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Diamond Price: ${pred:,.2f}")

    st.info("You are using the **{}** model.".format(selected_model_name))
    st.markdown("""
    **Evaluation Notes:**
    - **Train MSE**: `{:.2f}`
    - **Test MSE**: `{:.2f}`
    - These values are divided by 1000 for display purposes.
    """.format(mse.loc[selected_model_name, 'train'], mse.loc[selected_model_name, 'test']))
