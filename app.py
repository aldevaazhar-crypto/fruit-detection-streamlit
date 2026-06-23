import os
import gdown
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="AI Deteksi Gambar Buah",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 AI Deteksi Gambar Buah")
st.write(
    "Aplikasi ini digunakan untuk mengklasifikasikan gambar apakah termasuk "
    "gambar buah atau bukan buah menggunakan model CNN berformat H5."
)

# =========================
# KONFIGURASI MODEL
# =========================
MODEL_PATH = "model_buah_cnn.h5"

# ID Google Drive file model .h5
FILE_ID = "1a6L-Yy0hb7X5PE4VMEX-N2usdvDzLLzs"

# Sesuai input model kamu
IMG_SIZE = (277, 277)

# Label prediksi
# Kalau hasilnya nanti kebalik, cukup tukar dua teks di bawah ini.
LABEL_0 = "BUKAN BUAH"
LABEL_1 = "BUAH"


# =========================
# DOWNLOAD DAN LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"

        with st.spinner("Mengunduh model AI dari Google Drive..."):
            gdown.download(url, MODEL_PATH, quiet=False)

    # compile=False supaya model H5 lebih aman dibaca di Streamlit Cloud
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


try:
    model = load_model()
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error("Model gagal dimuat.")
    st.warning(
        "Pastikan file model di Google Drive sudah disetel menjadi "
        "'Anyone with the link' atau 'Siapa saja yang memiliki link'."
    )
    st.write("Detail error:")
    st.code(str(e))
    st.stop()


# =========================
# FUNGSI PREPROCESS GAMBAR
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# =========================
# UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader(
    "Upload gambar untuk dideteksi",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Gambar yang Diupload")
    st.image(image, use_container_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    prediction_value = float(prediction[0][0])

    # Model sigmoid:
    # Nilai mendekati 0 = LABEL_0
    # Nilai mendekati 1 = LABEL_1
    fruit_probability = prediction_value
    not_fruit_probability = 1 - fruit_probability

    st.subheader("Hasil Prediksi")

    if fruit_probability >= 0.5:
        st.success(f"HASIL: {LABEL_1}")
        st.write(f"Tingkat keyakinan {LABEL_1.lower()}: **{fruit_probability * 100:.2f}%**")
    else:
        st.error(f"HASIL: {LABEL_0}")
        st.write(f"Tingkat keyakinan {LABEL_0.lower()}: **{not_fruit_probability * 100:.2f}%**")

    st.write("Probabilitas Buah:")
    st.progress(fruit_probability)

    with st.expander("Detail nilai prediksi"):
        st.write(f"Probabilitas {LABEL_1}: {fruit_probability * 100:.2f}%")
        st.write(f"Probabilitas {LABEL_0}: {not_fruit_probability * 100:.2f}%")

    st.caption(
        "Catatan: aplikasi ini merupakan model klasifikasi gambar, sehingga hanya "
        "menentukan apakah gambar termasuk buah atau bukan buah."
    )

else:
    st.info("Silakan upload gambar terlebih dahulu.")
