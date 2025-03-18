 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Judul aplikasi
st.title("Prediksi Pengelolaan Persediaan dengan Machine Learning")

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV data stok", type=["csv"])

if uploaded_file is not None:
    # Membaca data
    df = pd.read_csv(uploaded_file)
    st.write("Data Stok:")
    st.dataframe(df.head())

    # Visualisasi Data
    st.subheader("Visualisasi Data Stok")
    fig, ax = plt.subplots()
    df.plot(x='Tanggal', y='Stok', kind='line', ax=ax)
    st.pyplot(fig)

    # Heatmap Korelasi
    st.subheader("Heatmap Korelasi")
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Machine Learning Model
    st.subheader("Prediksi Stok")
    feature_cols = ["Stok_Awal", "Penjualan"]  # Pilih fitur
    X = df[feature_cols]
    y = df["Stok"]  # Target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluasi Model
    error = mean_absolute_error(y_test, predictions)
    st.write(f"Mean Absolute Error: {error:.2f}")

    # Prediksi untuk input pengguna
    st.subheader("Prediksi Stok Baru")
    stok_awal = st.number_input("Masukkan Stok Awal:", min_value=0)
    penjualan = st.number_input("Masukkan Penjualan:", min_value=0)

    if st.button("Prediksi"):
        hasil_prediksi = model.predict([[stok_awal, penjualan]])
        st.success(f"Prediksi Stok yang Dibutuhkan: {hasil_prediksi[0]:.2f} unit")

