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
    page_title="AI Klasifikasi Jenis Buah",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 AI Klasifikasi Jenis Buah")
st.write(
    "Aplikasi ini digunakan untuk mengklasifikasikan jenis buah dari gambar "
    "menggunakan model CNN berformat H5."
)

# =========================
# KONFIGURASI MODEL
# =========================
MODEL_PATH = "model_buah_cnn.h5"
FIXED_MODEL_PATH = "model_buah_cnn_fixed.h5"

# GANTI dengan ID Google Drive model H5 multiclass kamu
FILE_ID = "1a6L-Yy0hb7X5PE4VMEX-N2usdvDzLLzs"

# Sesuaikan dengan ukuran input waktu training
IMG_SIZE = (277, 277)

# =========================
# DAFTAR KELAS BUAH
# =========================
# PENTING:
# Urutan ini harus sama dengan urutan class_names saat training.
# Kalau dataset kamu foldernya cuma buah_naga dan mangga, pakai:
# CLASS_NAMES = ["Buah Naga", "Mangga"]

CLASS_NAMES = [
    "Buah Naga",
    "Mangga",
    "Apel",
    "Pisang",
    "Jeruk"
]

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
# CEK JUMLAH OUTPUT MODEL
# =========================
try:
    output_shape = model.output_shape
    jumlah_output = output_shape[-1]

    if jumlah_output == 1:
        st.error(
            "Model ini masih model binary classification, jadi hanya bisa membedakan "
            "2 kondisi seperti Buah/Bukan Buah. Untuk hasil Buah Naga, Mangga, Apel, dll, "
            "model harus dilatih ulang sebagai multiclass classification."
        )
        st.stop()

    if jumlah_output != len(CLASS_NAMES):
        st.warning(
            f"Jumlah output model adalah {jumlah_output}, tetapi jumlah CLASS_NAMES adalah {len(CLASS_NAMES)}. "
            "Pastikan daftar CLASS_NAMES sesuai dengan jumlah kelas saat training."
        )
except Exception as e:
    st.warning("Tidak bisa membaca output shape model.")
    st.write(e)

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
    "Upload gambar buah untuk diklasifikasikan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Gambar yang Diupload")
    st.image(image, use_container_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0]

    predicted_index = int(np.argmax(prediction))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(prediction))

    st.subheader("Hasil Prediksi")

    st.success(f"HASIL: {predicted_class}")
    st.write(f"Tingkat keyakinan: **{confidence * 100:.2f}%**")

    st.write("Detail probabilitas:")

    for class_name, prob in zip(CLASS_NAMES, prediction):
        st.write(f"{class_name}: {prob * 100:.2f}%")
        st.progress(float(prob))

    st.caption(
        "Catatan: aplikasi ini menggunakan model klasifikasi multiclass, "
        "sehingga hasilnya berupa nama jenis buah."
    )

else:
    st.info("Silakan upload gambar buah terlebih dahulu.")
