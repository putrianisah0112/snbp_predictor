import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from openpyxl import Workbook

st.set_page_config(page_title="SNBP Predictor Advanced ML", layout="wide")

# ================= DATABASE =================
conn = sqlite3.connect("snbp.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS riwayat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nama_siswa TEXT,
    rata_nilai REAL,
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
    user = st.text_input("Username:guru")
    pw = st.text_input("Password:123", type="password")
    if st.button("Login"):
        if user == "guru" and pw == "123":
            st.session_state.login = True
            st.success("Login berhasil")
        else:
            st.error("Username / Password salah")

if not st.session_state.login:
    login()
    st.stop()

# ================= TEMPLATE GENERATOR =================
def create_template():
    wb = Workbook()
    ws = wb.active
    ws.title = "Nilai Rapor"
    headers = ["Mapel","Sem1","Sem2","Sem3","Sem4","Sem5"]
    ws.append(headers)

    subjects = ["Matematika","Bahasa Indonesia","Bahasa Inggris","Fisika","Kimia","Biologi"]
    for s in subjects:
        ws.append([s,"","","","",""])

    wb.save("template_nilai_rapor.xlsx")

create_template()

# ================= SIDEBAR =================
menu = st.sidebar.selectbox("Menu", ["Home","Upload & Prediksi","Data Riwayat","Logout"])

# ================= PTN & JURUSAN =================
ptn_list = [
    "Universitas Indonesia","Universitas Gadjah Mada","Universitas Sriwijaya",
    "Universitas Airlangga","Universitas Diponegoro","Institut Teknologi Bandung",
    "Institut Pertanian Bogor","Universitas Brawijaya",
    "Universitas Pendidikan Indonesia","Universitas Surabaya"
]

jurusan_list = [
    "Kedokteran","Teknik","Hukum","Ekonomi","Manajemen","Akuntansi",
    "Pendidikan","Matematika","Fisika","Kimia","Biologi","Informatika",
    "Statistika","Pertanian","Peternakan","Farmasi","Keperawatan",
    "Psikologi","Ilmu Komunikasi","Hubungan Internasional"
]

ptn_map = {ptn:i+1 for i,ptn in enumerate(ptn_list)}
jurusan_map = {jur:i+1 for i,jur in enumerate(jurusan_list)}

# ================= TRAINING DATA SIMULASI =================
np.random.seed(42)
X_train = []
y_train = []

for _ in range(500):
    nilai = np.random.uniform(65, 95)
    akreditasi = np.random.randint(0,2)
    ptn = np.random.randint(1,11)
    jurusan = np.random.randint(1,21)

    skor = nilai + (akreditasi*5) + ptn + jurusan
    diterima = 1 if skor > 120 else 0

    X_train.append([nilai, akreditasi, ptn, jurusan])
    y_train.append(diterima)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ================= HOME =================
if menu == "Home":
    st.title("🎓 SNBP Predictor Advanced ML")
    st.write("""
    Fitur:
    - Upload nilai rapor Excel
    - Input nama siswa
    - Prediksi peluang SNBP semua PTN & jurusan
    - Machine Learning RandomForest
    - Riwayat tersimpan database
    - Grafik & ranking jurusan
    """)

    with open("template_nilai_rapor.xlsx","rb") as f:
        st.download_button("📥 Download Template Excel", f, "template_nilai_rapor.xlsx")

# ================= UPLOAD & PREDIKSI =================
elif menu == "Upload & Prediksi":
    st.title("📤 Upload & Prediksi")

    nama_siswa = st.text_input("Nama Siswa")
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])

    akreditasi = st.selectbox("Akreditasi Sekolah", ["A","B","C"])
    akreditasi_num = 1 if akreditasi == "A" else 0

    if uploaded_file and nama_siswa:
        df = pd.read_excel(uploaded_file)

        st.subheader("📄 Data Nilai")
        st.dataframe(df)

        nilai_cols = ["Sem1","Sem2","Sem3","Sem4","Sem5"]
        df["Rata-rata"] = df[nilai_cols].mean(axis=1)

        rata_rata_siswa = df["Rata-rata"].mean()
        st.success(f"Rata-rata Nilai Rapor: {rata_rata_siswa:.2f}")

        hasil = []

        for ptn in ptn_list:
            for jurusan in jurusan_list:
                X_input = np.array([[rata_rata_siswa, akreditasi_num, ptn_map[ptn], jurusan_map[jurusan]]])
                prob = model.predict_proba(X_input)[0][1] * 100

                hasil.append([nama_siswa, akreditasi, ptn, jurusan, round(prob,2)])

                c.execute("""
                INSERT INTO riwayat (nama_siswa, rata_nilai, akreditasi, ptn, jurusan, peluang)
                VALUES (?,?,?,?,?,?)
                """, (nama_siswa, rata_rata_siswa, akreditasi, ptn, jurusan, prob))

        conn.commit()

        hasil_df = pd.DataFrame(
            hasil,
            columns=["Nama Siswa","Akreditasi","PTN","Jurusan","Peluang (%)"]
        )

        st.subheader("📊 Hasil Prediksi")
        st.dataframe(hasil_df)

        top10 = hasil_df.sort_values("Peluang (%)", ascending=False).head(10)
        st.subheader("🏆 Top 10 Jurusan Terbaik")
        st.table(top10)

        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(top10["Jurusan"], top10["Peluang (%)"])
        plt.xticks(rotation=45)
        ax.set_ylabel("Peluang (%)")
        st.pyplot(fig)

        st.download_button(
            "Download Hasil Prediksi CSV",
            hasil_df.to_csv(index=False),
            "hasil_prediksi_snbp.csv",
            "text/csv"
        )

    else:
        st.info("Masukkan nama siswa dan upload file Excel.")

# ================= DATA RIWAYAT =================
elif menu == "Data Riwayat":
    st.title("📁 Riwayat Prediksi")

    df = pd.read_sql("SELECT * FROM riwayat", conn)

    if df.empty:
        st.warning("Belum ada data.")
    else:
        st.dataframe(df)

        st.download_button(
            "Download Riwayat CSV",
            df.to_csv(index=False),
            "riwayat_prediksi.csv",
            "text/csv"
        )

# ================= LOGOUT =================
elif menu == "Logout":
    st.session_state.login = False
    st.success("Logout berhasil")
