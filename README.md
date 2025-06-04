# 💎 Diamond Price Prediction App

This project is a **machine learning regression app** that predicts the price of diamonds based on their physical and quality features. Users can interact with a clean UI to select between three ML models: **Random Forest**, **K-Nearest Neighbors (KNN)**, and **AdaBoost**.

Built using **Python, Streamlit, scikit-learn**, and deployed on **Streamlit Cloud**.

---

## 🚀 Features

- Predict diamond prices with high accuracy
- Choose between 3 ML models:
  - 🔥 Random Forest (best performer)
  - 📍 K-Nearest Neighbors (KNN)
  - ⚡ AdaBoost
- Real-time predictions with input validation
- Displays MSE scores for transparency

---

## 📊 Models & Evaluation (MSE)

| Model          | Train MSE | Test MSE | Notes                          |
|----------------|-----------|----------|--------------------------------|
| KNN            | 203.76    | 239.53   | Moderate accuracy              |
| Random Forest  | 52.29     | 130.79   | ⭐ Best performer (lowest MSE) |
| AdaBoost       | 904.84    | 846.21   | High error, less reliable      |

> MSE = Mean Squared Error  
> Lower MSE means better prediction performance.

---

## 📥 Input Features Explained

| Feature   | Description |
|-----------|-------------|
| **Carat** | Weight of the diamond (0.2 – 5.01 carats). Heavier diamonds usually cost more. |
| **Cut**   | Quality of the cut (Fair, Good, Very Good, Premium, Ideal). Affects brilliance. |
| **Color** | Diamond color grade, from J (worst) to D (best). Less color means higher value. |
| **Clarity** | Measures inclusions/blemishes, from I1 (worst) to IF (best). Higher clarity increases price. |
| **Length (x)** | Length in millimeters (0 – 10.74 mm). Used in dimension analysis. |
| **Width (y)**  | Width in mm (0 – 58.9 mm). Used in PCA to compute 3D shape. |
| **Depth (z)**  | Depth in mm (0 – 31.8 mm). Also contributes to overall dimensions. |
| **Table** | Top flat width relative to widest point (43 – 95%). Impacts sparkle. |
| **Dimension (PCA)** | Internally computed from x, y, z using PCA to reduce dimensionality while preserving shape info. |

All numerical inputs are **bounded** to values observed during training to:
- Avoid extrapolation errors
- Improve model reliability
- Keep predictions realistic

---

## 🌐 Deployment

This app is deployed using **Streamlit Cloud**. You can run it in your browser without installing anything.

🔗 [Live Demo (Streamlit Cloud)](https://your-streamlit-url.streamlit.app)

> *(Replace the link above with your actual Streamlit app link.)*

---

## ✨ Author

Made with 💎 by **Adityo Pangestu**  
👨‍💻 Visionary, Machine Learning Explorer

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
