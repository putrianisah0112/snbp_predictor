import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="SNBP Predictor Machine Learning", layout="wide")

# ================= DATABASE =================
conn = sqlite3.connect("snbp_ml.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS hasil (
    nama TEXT,
    akreditasi TEXT,
    rata_nilai REAL,
    universitas TEXT,
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
            st.rerun()
        else:
            st.error("Username / Password salah")

if not st.session_state.login:
    login()
    st.stop()

# ================= SIDEBAR =================
menu = st.sidebar.selectbox(
    "Menu",
    ["Home", "Download Template", "Upload & Prediksi (ML)", "Data & Grafik", "Logout"]
)
st.sidebar.info("⚠️ Prediksi hanya estimasi, bukan hasil resmi SNBP")

# ================= DATA UNIVERSITAS & JURUSAN =================
universitas_list = [
    "Universitas Indonesia","Universitas Gadjah Mada","Universitas Sriwijaya",
    "Universitas Airlangga","Universitas Diponegoro","Institut Teknologi Bandung",
    "Institut Pertanian Bogor","Universitas Brawijaya",
    "Universitas Pendidikan Indonesia","Universitas Surabaya"
]

jurusan_list = [
    "Kedokteran","Farmasi","Keperawatan",
    "Teknik Informatika","Teknik Sipil","Teknik Mesin",
    "Matematika","Fisika","Kimia","Biologi",
    "Ekonomi","Manajemen","Akuntansi",
    "Pendidikan Matematika","Pendidikan Fisika","Pendidikan Biologi"
]

# ================= TRAIN MODEL ML =================
@st.cache_resource
def train_model():
    # Dataset simulasi (nanti bisa diganti dataset SNBP asli)
    X = np.array([
        [90,3],[88,3],[85,3],[82,2],[80,2],[78,2],
        [75,2],[72,1],[70,1],[68,1],[65,1],[60,1]
    ])
    y = np.array([1,1,1,1,1,0,0,0,0,0,0,0])

    model = LogisticRegression()
    model.fit(X, y)
    return model

model = train_model()

# ================= HOME =================
if menu == "Home":
    st.title("🎓 SNBP Predictor (Machine Learning)")
    st.write("""
    Website ini memprediksi peluang kelulusan SNBP berdasarkan:
    - File Excel nilai rapor semester 1–5
    - Akreditasi sekolah
    - Machine Learning (Logistic Regression)
    """)

# ================= DOWNLOAD TEMPLATE =================
elif menu == "Download Template":
    st.title("📥 Download Template Excel Nilai Rapor")

    template_df = pd.DataFrame({
        "Mapel": ["Matematika","Bahasa Indonesia","Bahasa Inggris","Fisika","Kimia","Biologi"],
        "Sem1": ["","","","","",""],
        "Sem2": ["","","","","",""],
        "Sem3": ["","","","","",""],
        "Sem4": ["","","","","",""],
        "Sem5": ["","","","","",""]
    })

    st.download_button(
        "Download Template Excel",
        template_df.to_excel(index=False),
        file_name="template_nilai_rapor.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ================= UPLOAD & PREDIKSI =================
elif menu == "Upload & Prediksi (ML)":
    st.title("📤 Upload Excel & Prediksi SNBP (Machine Learning)")

    nama = st.text_input("Nama Siswa")
    akreditasi = st.selectbox("Akreditasi Sekolah", ["A","B","C"])
    file = st.file_uploader("Upload file Excel nilai rapor", type=["xlsx"])

    if st.button("Prediksi"):
        if nama.strip() == "" or file is None:
            st.error("Nama siswa dan file Excel wajib diisi")
        else:
            df = pd.read_excel(file)
            st.subheader("📄 Data Nilai dari Excel")
            st.dataframe(df)

            try:
                nilai = df.iloc[:,1:6].values.flatten()
                nilai = nilai[~np.isnan(nilai)]
                rata_rata = np.mean(nilai)

                st.success(f"📊 Rata-rata nilai rapor: {rata_rata:.2f}")

                akreditasi_map = {"A":3,"B":2,"C":1}
                akr = akreditasi_map[akreditasi]

                hasil = []

                for univ in universitas_list:
                    for jur in jurusan_list:
                        input_data = np.array([[rata_rata, akr]])
                        prob = model.predict_proba(input_data)[0][1] * 100

                        hasil.append([
                            nama, akreditasi, rata_rata,
                            univ, jur, round(prob,2)
                        ])

                hasil_df = pd.DataFrame(
                    hasil,
                    columns=["Nama","Akreditasi","Rata-rata Nilai","Universitas","Jurusan","Peluang (%)"]
                )

                st.subheader("🎯 Hasil Prediksi SNBP")
                st.dataframe(hasil_df)

                for row in hasil:
                    c.execute("INSERT INTO hasil VALUES (?,?,?,?,?,?)", row)
                conn.commit()

                st.download_button(
                    "Download Hasil Prediksi",
                    hasil_df.to_csv(index=False),
                    "hasil_prediksi_snbp_ml.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error("Format Excel tidak sesuai template!")
                st.write(e)

# ================= DATA & GRAFIK =================
elif menu == "Data & Grafik":
    st.title("📊 Data & Grafik")

    df = pd.read_sql("SELECT * FROM hasil", conn)
    if df.empty:
        st.warning("Belum ada data.")
    else:
        st.dataframe(df)

        fig, ax = plt.subplots()
        ax.bar(df["Universitas"], df["Peluang (%)"])
        plt.xticks(rotation=45)
        ax.set_ylabel("Peluang (%)")
        st.pyplot(fig)

# ================= LOGOUT =================
elif menu == "Logout":
    st.session_state.login = False
    st.success("Logout berhasil")
    st.rerun()

