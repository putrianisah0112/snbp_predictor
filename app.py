import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="SNBP Predictor", layout="wide")

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
    peluang REAL
)
""")
conn.commit()

# ================= MODEL ML (dummy training) =================
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

# ================= SIDEBAR =================
menu = st.sidebar.selectbox("Menu", ["Home", "Upload Excel", "Data & Grafik"])
st.sidebar.info("⚠️ Prediksi hanya estimasi, bukan hasil resmi SNBP")

# ================= HOME =================
if menu == "Home":
    st.title("🎓 SNBP Predictor")
    st.write("""
Website ini memprediksi peluang SNBP siswa berdasarkan:
- Nilai rapor
- Ranking
- Prestasi
- Akreditasi sekolah

Gunakan menu **Upload Excel** untuk memasukkan data siswa.
""")

# ================= UPLOAD EXCEL =================
elif menu == "Upload Excel":
    st.title("📂 Upload Data Excel")

    file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

    if file:
        df = pd.read_excel(file)
        st.subheader("Preview Data")
        st.dataframe(df)

        required_cols = ["Nama","Nilai","Ranking","Jumlah","Prestasi","Akreditasi","PTN","Jurusan"]

        if not all(col in df.columns for col in required_cols):
            st.error("❌ Format kolom Excel salah!")
            st.write("Kolom wajib:", required_cols)
        else:
            if st.button("Proses & Prediksi"):
                for i, row in df.iterrows():

                    prestasi_num = 1 if str(row["Prestasi"]).lower() == "ya" else 0
                    akreditasi_num = 1 if str(row["Akreditasi"]).upper() == "A" else 0

                    input_data = np.array([[ 
                        row["Nilai"], 
                        row["Ranking"], 
                        row["Jumlah"], 
                        prestasi_num, 
                        akreditasi_num
                    ]])

                    prob = model.predict_proba(input_data)[0][1] * 100

                    c.execute("""
                        INSERT INTO siswa 
                        (nama,nilai,ranking,jumlah,prestasi,akreditasi,ptn,jurusan,peluang)
                        VALUES (?,?,?,?,?,?,?,?,?)
                    """, (
                        row["Nama"], row["Nilai"], row["Ranking"], row["Jumlah"],
                        row["Prestasi"], row["Akreditasi"], row["PTN"], row["Jurusan"], round(prob,2)
                    ))

                conn.commit()
                st.success("✅ Data berhasil diproses & disimpan!")

# ================= DATA & GRAFIK =================
elif menu == "Data & Grafik":
    st.title("📊 Data & Grafik")

    df = pd.read_sql("SELECT * FROM siswa", conn)

    if df.empty:
        st.warning("Belum ada data.")
    else:
        st.dataframe(df)

        st.subheader("Grafik Peluang SNBP")
        fig, ax = plt.subplots()
        ax.bar(df["nama"], df["peluang"])
        ax.set_ylabel("Peluang (%)")
        ax.set_xlabel("Nama")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "hasil_snbp.csv",
            "text/csv"
        )
