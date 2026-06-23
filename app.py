import os
import json
import shutil
import h5py
import gdown
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="AI Klasifikasi Buah",
    page_icon="🍉",
    layout="centered"
)

st.title("🍉 AI Klasifikasi Buah")
st.write(
    "Aplikasi ini digunakan untuk mengklasifikasikan gambar buah "
    "menggunakan model CNN berformat H5."
)

# =========================
# KONFIGURASI MODEL
# =========================
MODEL_PATH = "model_buah_cnn.h5"
FIXED_MODEL_PATH = "model_buah_cnn_fixed.h5"

# ID file Google Drive
FILE_ID = "1a6L-Yy0hb7X5PE4VMEX-N2usdvDzLLzs"

# Ukuran input gambar
IMG_SIZE = (277, 277)

# Kelas asli model
# Jika hasil terbalik, tinggal tukar urutannya
CLASS_NAMES = ["Buah Naga", "Mangga"]


# =========================
# FUNGSI FIX MODEL H5
# =========================
def remove_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for value in obj.values():
            remove_quantization_config(value)
    elif isinstance(obj, list):
        for item in obj:
            remove_quantization_config(item)


def fix_h5_model(original_path, fixed_path):
    if os.path.exists(fixed_path):
        return fixed_path

    shutil.copy(original_path, fixed_path)

    with h5py.File(fixed_path, "r+") as h5file:
        model_config = h5file.attrs.get("model_config")

        if model_config is None:
            return fixed_path

        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

        model_config_json = json.loads(model_config)
        remove_quantization_config(model_config_json)

        h5file.attrs.modify("model_config", json.dumps(model_config_json))

    return fixed_path


# =========================
# DOWNLOAD DAN LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        with st.spinner("Mengunduh model dari Google Drive..."):
            gdown.download(url, MODEL_PATH, quiet=False)

    fixed_path = fix_h5_model(MODEL_PATH, FIXED_MODEL_PATH)
    model = tf.keras.models.load_model(fixed_path, compile=False)
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
# PREPROCESS GAMBAR
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
    "Upload gambar buah",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Gambar yang Diupload")
    st.image(image, use_container_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image, verbose=0)
    pred = prediction[0]

    st.subheader("Hasil Prediksi")

    # Dukungan untuk 2 kemungkinan output model:
    # 1. Binary sigmoid -> output 1 angka
    # 2. Softmax 2 kelas -> output 2 angka
    if len(pred.shape) == 0:
        pred = np.array([float(pred)])

    if len(pred) == 1:
        score = float(pred[0])

        if score >= 0.5:
            predicted_class = CLASS_NAMES[1]
            confidence = score
        else:
            predicted_class = CLASS_NAMES[0]
            confidence = 1 - score

        st.success(f"HASIL: {predicted_class}")
        st.write(f"Tingkat keyakinan: **{confidence * 100:.2f}%**")

        st.write("Detail probabilitas:")
        st.write(f"{CLASS_NAMES[0]}: {(1 - score) * 100:.2f}%")
        st.progress(float(1 - score))

        st.write(f"{CLASS_NAMES[1]}: {score * 100:.2f}%")
        st.progress(float(score))

    elif len(pred) == 2:
        predicted_index = int(np.argmax(pred))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(pred))

        st.success(f"HASIL: {predicted_class}")
        st.write(f"Tingkat keyakinan: **{confidence * 100:.2f}%**")

        st.write("Detail probabilitas:")
        for class_name, prob in zip(CLASS_NAMES, pred):
            st.write(f"{class_name}: {prob * 100:.2f}%")
            st.progress(float(prob))

    else:
        st.error(
            f"Jumlah output model terdeteksi {len(pred)}. "
            "Kode ini disiapkan untuk 2 kelas."
        )

else:
    st.info("Silakan upload gambar buah terlebih dahulu.")
