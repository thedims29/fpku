import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load model
model = load_model('model_pupuk.keras')

# Load data untuk scaler
data = pd.read_csv('DataPupuk.csv', sep=';', encoding='latin-1')
X = data['Luas Tanah (m²)'].values.reshape(-1, 1)
y = data[['Banyak Pupuk (kg)', 'Air (liter)', 'Waktu (hari)']].values

# Normalisasi data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(X)
scaler_y.fit(y)

# Judul Aplikasi
st.title('Prediksi Kebutuhan Pupuk Menggunakan Streamlit')

# Input Luas Tanah
tanah_input = st.number_input('Masukkan Luas Tanah (m²):', min_value=0.0, step=0.1)

if st.button('Prediksi'):
    # Normalisasi input
    luas_tanah_scaled = scaler_X.transform([[tanah_input]])
    luas_tanah_scaled = luas_tanah_scaled.reshape((1, 1, 1))

    # Prediksi
    prediksi_scaled = model.predict(luas_tanah_scaled)
    prediksi = scaler_y.inverse_transform(prediksi_scaled)

    banyak_pupuk = prediksi[0][0]
    air = prediksi[0][1]
    waktu = prediksi[0][2]

    # Tampilkan Hasil Prediksi
    st.success(f'Prediksi untuk Luas Tanah {tanah_input} m²:')
    st.write(f'Banyak Pupuk: {banyak_pupuk:.2f} kg')
    st.write(f'Air: {air:.2f} liter')
    st.write(f'Waktu: {waktu:.2f} hari')
