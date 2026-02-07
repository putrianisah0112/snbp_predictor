import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="SNBP Predictor + Rekomendasi", layout="wide")

# ================= DATABASE =================
conn = sqlite3.connect("snbp.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS siswa (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nama TEXT,
    nilai REAL,
    ranking INTEGER,
    jumlah INTEGER,
    prestasi TEXT,
    akreditasi TEXT,
    ptn TEXT,
    jurusan TEXT,
    peluang REAL,
    rekomendasi TEXT
)
""")
conn.commit()

# ================= MODEL =================
X = np.array([
    [85,2,30,1,1],
    [70,10,30,0,0],
    [90,1,30,1,1],
    [60,15,30,0,0],
    [88,3,30,1,1],
    [75,8,30,0,1]
])
y = np.array([1,0,1,0,1,0])

model = LogisticRegression()
model.fit(X,y)

# ================= REKOMENDASI =================
rekomendasi_ptn = {
    "tinggi":[("UI","Teknik Informatika"),("ITB","Teknik Elektro"),("UGM","Kedokteran")],
    "sedang":[("UNNES","Pendidikan Fisika"),("UNY","Pendidikan Matematika"),("UNS","Teknik Lingkungan")],
    "aman":[("UNM","Pendidikan IPA"),("UNESA","Fisika"),("UIN","Sistem Informasi")]
}

# ================= SIDEBAR =================
menu = st.sidebar.selectbox("Menu",["Home","Upload Excel","Data & Grafik"])
st.sidebar.info("Prediksi hanya estimasi, bukan keputusan resmi SNBP")

# ================= HOME =================
if menu=="Home":
    st.title("🎓 SNBP Predictor + Rekomendasi PTN & Jurusan")
    st.write("
