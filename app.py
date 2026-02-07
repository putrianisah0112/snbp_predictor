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
    st.write("Upload Excel → Prediksi peluang → Rekomendasi PTN & Jurusan")

# ================= UPLOAD =================
elif menu=="Upload Excel":
    st.title("📂 Upload Excel")
    file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

    if file:
        df = pd.read_excel(file)
        st.dataframe(df)

        kolom_wajib = ["Nama","Nilai","Ranking","Jumlah","Prestasi","Akreditasi","PTN","Jurusan"]

        if not all(k in df.columns for k in kolom_wajib):
            st.error("Format kolom salah!")
            st.write(kolom_wajib)
        else:
            if st.button("Proses & Prediksi"):
                for i,row in df.iterrows():

                    prestasi_num = 1 if str(row["Prestasi"]).lower()=="ya" else 0
                    akreditasi_num = 1 if str(row["Akreditasi"]).upper()=="A" else 0

                    X_input = np.array([[row["Nilai"],row["Ranking"],row["Jumlah"],prestasi_num,akreditasi_num]])
                    prob = model.predict_proba(X_input)[0][1]*100

                    if prob>=80:
                        kategori="tinggi"
                    elif prob>=60:
                        kategori="sedang"
                    else:
                        kategori="aman"

                    rekom_list = rekomendasi_ptn[kategori]
                    rekom_text = "; ".join([f"{p}-{j}" for p,j in rekom_list])

                    c.execute("""
                    INSERT INTO siswa
                    (nama,nilai,ranking,jumlah,prestasi,akreditasi,ptn,jurusan,peluang,rekomendasi)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,(
                        row["Nama"],
                        row["Nilai"],
                        row["Ranking"],
                        row["Jumlah"],
                        row["Prestasi"],
                        row["Akreditasi"],
                        row["PTN"],
                        row["Jurusan"],
                        round(prob,2),
                        rekom_text
                    ))

                conn.commit()
                st.success("✅ Data berhasil diproses & disimpan")

# ================= DATA =================
elif menu=="Data & Grafik":
    st.title("📊 Data & Grafik")

    df = pd.read_sql("SELECT * FROM siswa", conn)

    if df.empty:
        st.warning("Belum ada data")
    else:
        st.dataframe(df)

        fig, ax = plt.subplots()
        ax.bar(df["nama"], df["peluang"])
        ax.set_ylabel("Peluang (%)")
        ax.set_xlabel("Nama")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.download_button("Download CSV", df.to_csv(index=False), "hasil_snbp.csv")
