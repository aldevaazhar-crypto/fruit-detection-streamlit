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
FIXED_MODEL_PATH = "model_buah_cnn_fixed.h5"

# ID Google Drive file model .h5
FILE_ID = "1a6L-Yy0hb7X5PE4VMEX-N2usdvDzLLzs"

# Sesuai input model buah kamu
IMG_SIZE = (277, 277)

# Label prediksi
# Kalau hasilnya kebalik, tukar dua label ini.
LABEL_0 = "BUKAN BUAH"
LABEL_1 = "BUAH"


# =========================
# FUNGSI FIX MODEL H5
# =========================
def remove_quantization_config(obj):
    """
    Menghapus key quantization_config dari konfigurasi model H5.
    Ini untuk mengatasi error:
    Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
    """
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for value in obj.values():
            remove_quantization_config(value)
    elif isinstance(obj, list):
        for item in obj:
            remove_quantization_config(item)


def fix_h5_model(original_path, fixed_path):
    """
    Membuat salinan model H5 lalu membersihkan model_config dari quantization_config.
    """
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
        with st.spinner("Mengunduh model AI dari Google Drive..."):
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
    # mendekati 0 = LABEL_0
    # mendekati 1 = LABEL_1
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
