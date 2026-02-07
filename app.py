import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np

st.set_page_config(page_title="SNBP Predictor v2", layout="wide")

# ================= DATABASE =================
conn = sqlite3.connect("snbp.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS siswa (
    nama TEXT,
    nilai REAL,
    ranking INTEGER,
    jumlah INTEGER,
    prestasi TEXT,
    akreditasi TEXT,
    ptn TEXT,
    jurusan TEXT,
    peluang REAL
)
""")
conn.commit()

# ================= LOGIN =================
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    st.title("🔐 Login Guru")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "guru" and pw == "123":
            st.session_state.login = True
            st.success("Login berhasil")
        else:
            st.error("Username / Password salah")

if not st.session_state.login:
    login()
    st.stop()

# ================= SIDEBAR =================
menu = st.sidebar.selectbox("Menu", ["Home", "Input Data", "Upload Excel", "Data & Grafik", "Logout"])
st.sidebar.info("⚠️ Prediksi hanya estimasi, bukan hasil resmi SNBP")

# ================= MODEL ML (DUMMY TRAINING) =================
# Data contoh training (simulasi)
X = np.array([
    [85, 2, 30, 1, 1],
    [70, 10, 30, 0, 0],
    [90, 1, 30, 1, 1],
    [60, 15, 30, 0, 0],
    [88, 3, 30, 1, 1],
    [75, 8, 30, 0, 1]
])
y = np.array([1,0,1,0,1,0])

model = LogisticRegression()
model.fit(X, y)

# ================= HOME =================
if menu == "Home":
    st.title("🎓 SNBP Predictor v2")
    st.write("""
    Website ini membantu guru memprediksi peluang siswa lolos SNBP berdasarkan:
    - Nilai rapor
    - Ranking
    - Prestasi
    - Akreditasi
    - PTN & Jurusan
    """)

# ================= INPUT DATA =================
elif menu == "Input Data":
    st.title("📝 Input Data Siswa")

    nama = st.text_input("Nama Siswa")
    nilai = st.number_input("Rata-rata Nilai Rapor", 0.0, 100.0, 80.0)
    ranking = st.number_input("Ranking", 1, 50, 5)
    jumlah = st.number_input("Jumlah siswa kelas", 1, 50, 30)

    prestasi = st.selectbox("Prestasi", ["Tidak", "Ya"])
    akreditasi = st.selectbox("Akreditasi Sekolah", ["A", "B", "C"])

    ptn = st.selectbox("Pilih PTN", ["UI", "UGM", "ITB", "UNPAD", "UNNES"])
    jurusan = st.selectbox("Pilih Jurusan", ["Teknik", "Kedokteran", "Pendidikan", "Ekonomi", "Sains"])

    prestasi_num = 1 if prestasi == "Ya" else 0
    akreditasi_num = 1 if akreditasi == "A" else 0

    if st.button("Prediksi"):
        input_data = np.array([[nilai, ranking, jumlah, prestasi_num, akreditasi_num]])
        prob = model.predict_proba(input_data)[0][1] * 100

        if prob >= 75:
            kategori = "Tinggi"
            rekomendasi = "Bisa mencoba PTN favorit"
        elif prob >= 50:
            kategori = "Sedang"
            rekomendasi = "Pilih PTN peluang aman"
        else:
            kategori = "Rendah"
            rekomendasi = "Perlu alternatif SNBT / kampus lain"

        st.success(f"Peluang diterima: {prob:.1f}%")
        st.write("Kategori:", kategori)
        st.write("Rekomendasi:", rekomendasi)

        c.execute("INSERT INTO siswa VALUES (?,?,?,?,?,?,?,?,?)",
                  (nama, nilai, ranking, jumlah, prestasi, akreditasi, ptn, jurusan, prob))
        conn.commit()

#============ INPUT DATA ==============
elif menu == "Upload Excel":
    st.title("📂 Upload Data Excel Siswa")

    file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

    if file:
        df = pd.read_excel(file)
        st.dataframe(df)

        if st.button("Proses & Prediksi"):
            for i, row in df.iterrows():
                prestasi_num = 1 if row["Prestasi"] == "Ya" else 0
                akreditasi_num = 1 if row["Akreditasi"] == "A" else 0

                input_data = np.array([[row["Nilai"], row["Ranking"], row["Jumlah"], prestasi_num, akreditasi_num]])
                prob = model.predict_proba(input_data)[0][1] * 100

                c.execute("INSERT INTO siswa VALUES (?,?,?,?,?,?,?,?,?)",
                          (row["Nama"], row["Nilai"], row["Ranking"], row["Jumlah"],
                           row["Prestasi"], row["Akreditasi"], row["PTN"], row["Jurusan"], prob))

            conn.commit()
            st.success("Data Excel berhasil diproses & disimpan!")


# ================= DATA & GRAFIK =================
elif menu == "Data & Grafik":
    st.title("📊 Data & Grafik")

    df = pd.read_sql("SELECT * FROM siswa", conn)

    if df.empty:
        st.warning("Belum ada data.")
    else:
        st.dataframe(df)

        fig, ax = plt.subplots()
        ax.bar(df["nama"], df["peluang"])
        ax.set_ylabel("Peluang (%)")
        ax.set_xlabel("Nama")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "data_snbp.csv",
            "text/csv"
        )

# ================= LOGOUT =================
elif menu == "Logout":
    st.session_state.login = False
    st.success("Logout berhasil")

