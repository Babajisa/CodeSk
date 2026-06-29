import streamlit as st
import numpy as np
import torch
import faiss
import os
import pickle
import requests
import json
import uuid
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor
from dotenv import load_dotenv
import re

# Membaca file .env di awal aplikasi
load_dotenv()

# Tampilan halaman Streamlit minimalis berfokus di tengah
st.set_page_config(page_title="Asisten Tanya Jawab Islami", layout="centered")

# Custom styling untuk tampilan premium
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Amiri:ital,wght@0,400;0,700;1,400;1,700&display=swap');

/* Main font styling */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Background gradient styling */
.stApp {
    background: linear-gradient(135deg, #0b1329 0%, #1c1538 50%, #030712 100%) !important;
    color: #e2e8f0 !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: rgba(11, 19, 41, 0.95) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Header text inside sidebar */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #10b981 !important;
}

/* Chat container and message cards styling */
.stChatMessage {
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    margin-bottom: 15px !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stChatMessage:hover {
    transform: translateY(-2px);
    border-color: rgba(16, 185, 129, 0.4) !important;
    box-shadow: 0 6px 24px rgba(16, 185, 129, 0.15) !important;
}

/* Chat avatar custom shape */
[data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] {
    background-color: #10b981 !important;
}

/* Make Arabic text beautiful and readable */
.arabic-text {
    font-family: 'Amiri', serif !important;
    font-size: 1.65rem !important;
    line-height: 2.3 !important;
    direction: rtl !important;
    text-align: right !important;
    color: #ffd700 !important; /* gold color for Arabic text */
    padding: 12px 18px !important;
    background: rgba(255, 255, 255, 0.02) !important;
    border-radius: 12px !important;
    margin-top: 12px !important;
    margin-bottom: 12px !important;
    border-right: 4px solid #ffd700 !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.1) !important;
}

/* Custom header/title gradient styling */
.main-title {
    background: linear-gradient(90deg, #3b82f6 0%, #10b981 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 700 !important;
    font-size: 2.6rem !important;
    margin-bottom: 0.5rem !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1) !important;
}

.sub-title {
    color: #94a3b8 !important;
    font-size: 1.1rem !important;
    margin-bottom: 2rem !important;
}

/* Reference card */
.reference-card {
    background-color: rgba(16, 185, 129, 0.08) !important;
    border-left: 4px solid #10b981 !important;
    padding: 12px 18px !important;
    border-radius: 4px 12px 12px 4px !important;
    margin-top: 15px !important;
    font-size: 0.95rem !important;
    color: #a7f3d0 !important;
    border: 1px solid rgba(16, 185, 129, 0.15) !important;
}

/* Custom buttons styling */
.stButton>button {
    background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(16, 185, 129, 0.35) !important;
}
</style>
""", unsafe_allow_html=True)

# Pengaturan User ID di URL query parameters untuk riwayat chat persisten
if "user_id" not in st.query_params:
    st.query_params["user_id"] = str(uuid.uuid4())
user_id = st.query_params["user_id"]

DB_DIR = "faiss_vdb"
INDEX_FILE = os.path.join(DB_DIR, "index.faiss")
METADATA_FILE = os.path.join(DB_DIR, "metadata.pkl")
HISTORY_DIR = "data/history"

if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

def load_chat_history(uid):
    filepath = os.path.join(HISTORY_DIR, f"{uid}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Gagal memuat riwayat chat: {e}")
    return []

def save_chat_history(uid, messages):
    filepath = os.path.join(HISTORY_DIR, f"{uid}.json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Gagal menyimpan riwayat chat: {e}")

# ==========================================
# 1. SIDEBAR BERSIH (PILIHAN API)
# ==========================================
st.sidebar.header("🔑 Akses Sistem")

# Membaca OpenAI API Key dari file .env atau Streamlit Secrets
openai_api_key = os.getenv("OPENAI_API_KEY") or ""

# Coba ambil dari st.secrets jika berjalan di Streamlit Cloud
if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]

# Bersihkan whitespace/tanda kutip manual jika ada
openai_api_key = openai_api_key.strip('"').strip("'").strip()

# Menampilkan status koneksi di sidebar
if openai_api_key:
    st.sidebar.success("✅ OpenAI Terhubung")
    ganti_key = st.sidebar.checkbox("Ganti OpenAI API Key?")
    if ganti_key:
        openai_api_key = st.sidebar.text_input("Masukkan OpenAI API Key Baru:", type="password", value=openai_api_key)
else:
    openai_api_key = st.sidebar.text_input("Masukkan OpenAI API Key:", type="password")
    if openai_api_key:
        openai_api_key = openai_api_key.strip()
        st.sidebar.success("✅ API Key dimasukkan")
    else:
        st.sidebar.error("❌ API Key tidak ditemukan!")
        st.sidebar.info("Silakan isi API Key di atas atau tambahkan di file `.env` / Streamlit Secrets.")

st.sidebar.write("---")

# Tombol untuk menghapus riwayat chat
if st.sidebar.button("🧹 Hapus Riwayat Chat", use_container_width=True):
    filepath = os.path.join(HISTORY_DIR, f"{user_id}.json")
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            pass
    st.session_state.messages = []
    st.success("Riwayat chat berhasil dibersihkan!")
    st.rerun()

st.sidebar.write("---")
st.sidebar.info("Aplikasi ini menggunakan model GPT-4o mini resmi dari OpenAI untuk menjawab pertanyaan berdasarkan rujukan Al-Qur'an dan Tafsir.")

# ==========================================
# 2. LOAD MODEL & DATABASE (Latar Belakang)
# ==========================================
@st.cache_resource
def load_resources():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    preprocessor = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02")
    
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
    else:
        index, metadata = None, None
        
    return preprocessor, tokenizer, model, index, metadata

preprocessor, tokenizer, model, index, metadata = load_resources()

# Fungsi untuk melakukan pooling (mean pooling) guna merata-ratakan nilai embedding token
# Ini mengubah embedding tingkat token menjadi embedding representasi satu kalimat utuh
def mean_pooling(model_output, attention_mask):
    # Mengambil token embeddings (dimensi output pertama dari model)
    token_embeddings = model_output[0]
    
    # Memperluas dimensi attention mask agar cocok dengan bentuk token embeddings
    # Mengubahnya menjadi tipe float untuk perkalian elemen-per-elemen
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Mengalikan embedding token dengan mask agar token padding diabaikan (bernilai 0),
    # kemudian menjumlahkan seluruh token di sumbu baris (dimensi 1), lalu dibagi dengan
    # jumlah total token riil (yang tidak dipadding) untuk mendapatkan rata-rata.
    # torch.clamp digunakan untuk mencegah pembagian dengan nilai 0 (nilai minimum dibatasi 1e-9).
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Fungsi untuk menghasilkan vektor embedding numerik dari sebuah teks input
def get_embedding(text):
    # Jika terdeteksi karakter/tulisan bahasa Arab (rentang Unicode Arab \u0600 - \u06FF)
    if any("\u0600" <= c <= "\u06FF" for c in str(text)):
        # Lakukan preprocessing teks Arab menggunakan ArabertPreprocessor (menghilangkan harakat, tatweel, dll.)
        cleaned_text = preprocessor.preprocess(str(text))
    else:
        # Jika bukan bahasa Arab, gunakan teks asli dalam bentuk string
        cleaned_text = str(text)
        
    # Lakukan tokenisasi teks input, menambahkan padding, pemotongan teks (truncation) max 512 token,
    # dan mengembalikan outputnya dalam format PyTorch tensors ('pt')
    inputs = tokenizer(cleaned_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    # Jalankan model ekstraksi tanpa menghitung gradien (torch.no_grad) untuk menghemat memori dan meningkatkan kecepatan
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Panggil fungsi mean_pooling untuk menggabungkan token embeddings menjadi satu vektor kalimat,
    # ubah ke array NumPy, dan ratakan (flatten) menjadi array 1 dimensi
    return mean_pooling(outputs, inputs['attention_mask']).numpy().flatten()

# Fungsi pencarian berbasis kata kunci (lexical/keyword search) sebagai pelengkap pencarian semantik
def keyword_search(query, metadata, k=5):
    # Mengekstrak seluruh kata alfanumerik dari query pengguna, mengubahnya ke huruf kecil,
    # dan hanya mengambil kata yang memiliki panjang karakter lebih dari 2 huruf
    words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
    
    # Menyaring dan mengabaikan stop words umum dalam bahasa Indonesia agar pencarian lebih fokus pada kata kunci penting
    stop_words = {'mengapa', 'bagaimana', 'apakah', 'adakah', 'yang', 'dalam', 'dan', 'atau', 'untuk', 'dengan', 'dari', 'pada', 'saya', 'bisa', 'dapat'}
    keywords = [w for w in words if w not in stop_words]
    
    # Jika tidak ada kata kunci tersisa setelah disaring, kembalikan list kosong
    if not keywords:
        return []
        
    scores = []
    # Lakukan pencarian sekuensial pada setiap dokumen di metadata
    for idx, item in enumerate(metadata):
        text_lower = item["teks"].lower()
        sumber_lower = item["sumber"].lower()
        score = 0
        for kw in keywords:
            # Jika kata kunci ada dalam teks rujukan
            if kw in text_lower:
                score += 1 # Berikan poin dasar +1
                # Berikan tambahan skor +2 jika kata kunci tersebut cocok sebagai kata utuh (bukan bagian dari kata lain)
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    score += 2
            # Jika kata kunci cocok dengan nama sumber dokumen (misal kategori subjek)
            if kw in sumber_lower:
                score += 1 # Berikan poin tambahan +1
        # Jika dokumen memiliki kecocokan (skor > 0), simpan indeks dan skornya
        if score > 0:
            scores.append((idx, score))
            
    # Urutkan daftar dokumen berdasarkan skor tertinggi ke terendah
    scores.sort(key=lambda x: x[1], reverse=True)
    # Kembalikan indeks sebanyak maksimal k dokumen teratas
    return [idx for idx, score in scores[:k]]

# Fungsi untuk mendeteksi teks Arab dalam respons dan membungkusnya dengan styling HTML/CSS
def format_output_with_arabic(text):
    # Pola regex untuk mendeteksi karakter Arab berurutan sepanjang 8 karakter atau lebih
    # Pola ini juga mendeteksi harakat, tanda baca Arab, spasi, dan angka Arab
    arabic_pattern = r'([\u0600-\u06FF][\u0600-\u06FF\s\u064B-\u065F\u0670\u06D6-\u06ED\u06F0-\u06F9\u060C\u061B\u061F\(\)]*[\u0600-\u06FF])'
    
    # Fungsi internal untuk membungkus teks Arab yang cocok
    def wrap_arabic(match):
        val = match.group(0).strip()
        if val:
            # Ganti baris baru (\n) menjadi tag break HTML (<br>) agar format paragraf terjaga
            val_clean = val.replace('\n', '<br>')
            # Bungkus teks Arab dengan div berkelas "arabic-text" dan arah penulisan kanan-ke-kiri (rtl)
            return f'<div class="arabic-text" dir="rtl">{val_clean}</div>'
        return match.group(0)
        
    # Gantikan semua kecocokan teks Arab dalam string dengan format HTML yang baru
    return re.sub(arabic_pattern, wrap_arabic, text)

# Fungsi guardrail/filter topik untuk menolak pertanyaan di luar bidang Al-Qur'an dan keislaman
def check_prompt_relevance(prompt, api_key):
    # Alamat endpoint API resmi untuk Chat Completions OpenAI
    url_openai = "https://api.openai.com/v1/chat/completions"
    
    # Header permintaan HTTP berisi Bearer Token autentikasi dan format konten JSON
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Instruksi sistem yang memaksa model bertindak sebagai filter klasifikasi biner ketat
    system_instruction = (
        "Anda adalah filter klasifikasi pertanyaan. Tugas Anda adalah menentukan apakah "
        "pertanyaan pengguna berkaitan dengan Al-Qur'an, Tafsir"
        "Jawablah HANYA dengan kata 'YES' jika berkaitan, atau 'NO' jika "
        "tidak berkaitan. Jangan berikan penjelasan apa pun, cukup satu kata saja."
    )
    
    # Payload parameter API OpenAI (memakai model gpt-4o-mini dengan temperature 0 untuk hasil konsisten)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": 5, # Batasi token respons sangat kecil untuk menghemat biaya API
        "temperature": 0.0
    }
    try:
        # Kirim HTTP POST request secara sinkron dengan batas waktu timeout 15 detik
        response = requests.post(url_openai, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            res_data = response.json()
            choices = res_data.get("choices", [])
            if choices and isinstance(choices, list):
                # Ambil teks jawaban filter, bersihkan whitespace, dan ubah ke huruf kapital
                result = choices[0].get("message", {}).get("content", "").strip().upper()
                # Bersihkan tanda baca titik atau tanda seru opsional dari LLM
                result = result.replace(".", "").replace("!", "")
                # Kembalikan True jika respons bernilai YES
                return "YES" in result or result == "YES"
    except Exception as e:
        # Jika terjadi kendala jaringan/koneksi API, loloskan filter (default True) agar sistem tetap bisa mencoba melayani pengguna
        return True
    return True


# ==========================================
# 3. ANTARMUKA CHAT UTAMA
# ==========================================
st.markdown('<div class="main-title">🕌 Asisten Pintar Al-Qur\'an & Tafsir AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Silakan masukkan pertanyaan Anda di kolom bawah untuk mencari rujukan ayat secara otomatis.</div>', unsafe_allow_html=True)

if index is None or metadata is None:
    st.warning("⚠️ Database sistem (`faiss_vdb`) belum dibangun atau tidak ditemukan di server.")
    st.write("Untuk mencari rujukan ayat secara otomatis, sistem memerlukan database pencarian (FAISS).")
    
    if st.button("Membangun Database Sekarang (Butuh beberapa menit)", type="primary"):
        with st.status("Sedang memproses database...", expanded=True) as status:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def streamlit_callback(completed, total, msg):
                pct = completed / total if total > 0 else 0.0
                pct = max(0.0, min(1.0, pct))
                progress_bar.progress(pct)
                status_text.write(msg)
                
            try:
                import build_db
                build_db.build_database(
                    custom_model=model,
                    custom_tokenizer=tokenizer,
                    custom_preprocessor=preprocessor,
                    progress_callback=streamlit_callback
                )
                status.update(label="✅ Database sukses dibangun!", state="complete", expanded=False)
                st.success("Sukses! Memuat ulang sistem...")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                status.update(label="❌ Gagal membangun database!", state="error", expanded=True)
                st.error(f"Terjadi kesalahan: {e}")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history(user_id)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(format_output_with_arabic(message["content"]), unsafe_allow_html=True)

    # 1. Mendeteksi jika user memasukkan pertanyaan baru lewat kolom input Streamlit
    # Menggunakan operator walrus (:=) untuk langsung menugaskan input ke variabel 'query_user'
    if query_user := st.chat_input("Tanyakan sesuatu tentang Al-Qur'an atau Tafsir..."):
        # Menampilkan gelembung chat dari sisi pengguna (user) di UI Streamlit
        with st.chat_message("user"):
            # Render pesan pengguna dengan format teks Arab (jika ada) menggunakan HTML
            st.markdown(format_output_with_arabic(query_user), unsafe_allow_html=True)
        # Menambahkan pesan pengguna ke daftar riwayat chat di session state
        st.session_state.messages.append({"role": "user", "content": query_user})
        # Menyimpan riwayat chat terbaru ke dalam file JSON lokal
        save_chat_history(user_id, st.session_state.messages)

        # Inisialisasi status kesiapan API sebelum memproses database
        api_ready = True
        # Jika OpenAI API Key belum terisi, tandai tidak siap dan siapkan pesan error
        if not openai_api_key:
            api_ready = False
            msg = "⚠️ Sistem tidak bisa menjawab karena `OPENAI_API_KEY` belum diisi."
        
        # Jika API Key siap, lakukan validasi relevansi topik (guardrail/filter awal)
        elif api_ready:
            # Tampilkan spinner pemrosesan di Streamlit
            with st.spinner("Memvalidasi topik pertanyaan..."):
                # Panggil LLM filter untuk mengecek apakah pertanyaan bertopik Islam/Al-Qur'an
                is_relate = check_prompt_relevance(query_user, openai_api_key)
                # Jika tidak berkaitan dengan keislaman, ubah status kesiapan menjadi False
                if not is_relate:
                    api_ready = False
                    msg = "Maaf, sistem ini dirancang khusus untuk menjawab pertanyaan seputar Al-Qur'an, Tafsir, dan keislaman. Pertanyaan Anda tampaknya berada di luar topik tersebut."

        # Jika API tidak siap (karena kunci kosong atau topik tidak relevan)
        if not api_ready:
            # Tampilkan pesan penjelasan di gelembung chat asisten
            with st.chat_message("assistant"):
                st.markdown(msg)
                # Masukkan pesan penolakan/error ke session state dan simpan riwayat chat
                st.session_state.messages.append({"role": "assistant", "content": msg})
                save_chat_history(user_id, st.session_state.messages)
        else:
            # Jika semua validasi lolos, mulai proses pencarian dokumen lokal (RAG)
            with st.spinner("Sedang menelusuri rujukan database lokal..."):
                # Ubah teks kueri pengguna menjadi vektor embedding (float32) berdimensi 384
                query_vector = np.array([get_embedding(query_user)]).astype('float32')
                # Hitung nilai norma L2 (panjang vektor) dari query_vector
                norm_q = np.linalg.norm(query_vector)
                # Lakukan normalisasi L2 pada vektor (vektor dibagi normanya) agar perhitungan kemiripan kosinus di FAISS akurat
                if norm_q > 1e-9:
                    query_vector = query_vector / norm_q
                
                # Tahap 1: Jalankan pencarian semantik menggunakan indeks FAISS (ambil 5 terdekat)
                # D berisi jarak/kemiripan, I berisi daftar indeks baris dokumen hasil pencarian
                D, I = index.search(query_vector, 5)
                # Saring indeks hasil pencarian agar tidak bernilai -1 (tidak ketemu) dan tidak melebihi batas ukuran metadata
                semantic_indices = [idx for idx in I.flatten() if idx != -1 and idx < len(metadata)]
                
                # Tahap 2: Jalankan pencarian berbasis kata kunci (keyword/lexical matching) (ambil 5 terdekat)
                keyword_indices = keyword_search(query_user, metadata, k=5)
                
                # Tahap 3: Gabungkan kedua hasil pencarian (semantik diprioritaskan di awal)
                combined_indices = []
                seen_idx = set()
                # Looping gabungan list indeks semantik dan kata kunci
                for idx in semantic_indices + keyword_indices:
                    # Pastikan tidak ada indeks duplikat yang dimasukkan kembali
                    if idx not in seen_idx:
                        seen_idx.add(idx)
                        combined_indices.append(idx)
                
                # Batasi rujukan dokumen yang diambil maksimal hanya 10 rujukan teratas untuk prompt context
                combined_indices = combined_indices[:10]
                
                # Inisialisasi list untuk menyimpan teks konten dan nama sumber dokumen rujukan
                konteks_list = []
                sumber_list = []
                seen_texts = set()
                
                # Ambil teks asli dari objek metadata berdasarkan list indeks gabungan
                for idx in combined_indices:
                    item = metadata[idx]
                    # Hindari redundansi jika ada teks dokumen yang sama persis
                    if item['teks'] not in seen_texts:
                        seen_texts.add(item['teks'])
                        konteks_list.append(item['teks'])
                        sumber_list.append(item['sumber'])

            # Render gelembung chat dari sisi asisten AI
            with st.chat_message("assistant"):
                # Sediakan placeholder kosong Streamlit agar respons bisa diperbarui dinamis
                message_placeholder = st.empty()
                
                # Kelompokkan dokumen rujukan menjadi ayat Al-Qur'an dan artikel tafsir
                quran_list = []
                tafsir_list = []
                for txt, src in zip(konteks_list, sumber_list):
                    # Jika nama sumber terdeteksi mengandung kata "Al-Qur'an"
                    if "Al-Qur'an" in src:
                        quran_list.append(txt) # Masukkan ke list rujukan ayat
                    else:
                        # Masukkan ke list rujukan tafsir beserta format nama artikelnya
                        tafsir_list.append(f"[{src}]\n{txt}")
                
                # Gabungkan list ayat Al-Qur'an menjadi satu teks paragraf panjang
                quran_string = "\n\n".join(quran_list) if quran_list else "Tidak ada rujukan ayat Al-Qur'an langsung yang ditemukan."
                # Gabungkan list artikel tafsir menjadi satu teks paragraf panjang
                tafsir_string = "\n\n".join(tafsir_list) if tafsir_list else "Tidak ada rujukan artikel tafsir tambahan yang ditemukan."

                # Menyusun RAG Prompt komprehensif berisi instruksi sistem, rujukan dokumen pendukung, dan pertanyaan user
                prompt_rag = f"""Anda adalah asisten ahli tafsir Al-Qur'an. Tugas Anda adalah menjawab pertanyaan pengguna secara jelas, akurat, dan sopan menggunakan Rujukan Dokumen yang disediakan di bawah.
 
Berikut adalah Rujukan Dokumen yang dibagi menjadi dua bagian:
 
--- RUJUKAN AYAT AL-QUR'AN ---
{quran_string}
 
--- RUJUKAN ARTIKEL TAFSIR ---
{tafsir_string}
 
PANDUAN MENJAWAB:
1. JIKA terdapat ayat Al-Qur'an yang relevan di bagian "RUJUKAN AYAT AL-QUR'AN", Anda WAJIB menampilkan teks Arab asli ayat tersebut beserta terjemahannya secara lengkap (verbatim) di dalam jawaban Anda.
2. Rujukan di bagian "RUJUKAN ARTIKEL TAFSIR" adalah penjelasan tambahan. Gunakan artikel tafsir ini HANYA jika ia relevan dengan pertanyaan untuk memberikan konteks penjelasan/tafsir. Jika tidak relevan, abaikan saja bagian artikel tafsir ini.
3. HATI-HATI: Jangan mengutip potongan tulisan bahasa Arab apa pun yang berada di dalam bagian "RUJUKAN ARTIKEL TAFSIR" sebagai teks ayat Al-Qur'an! Kutipan ayat Al-Qur'an asli hanya boleh diambil dari bagian "RUJUKAN AYAT AL-QUR'AN".
4. Jika informasi jawaban sama sekali tidak ada di dalam kedua rujukan di atas, katakan sejujurnya bahwa informasi tersebut tidak ditemukan dalam database Anda.
 
PERTANYAAN PENGGUNA:
{query_user}
 
JAWABAN:"""
 
                try:
                    jawaban_ai = ""
                    # Definisikan alamat URL endpoint untuk pemanggilan API OpenAI resmi
                    url_openai = "https://api.openai.com/v1/chat/completions"
                    
                    # Header HTTP request berisi otorisasi token OpenAI dan jenis konten JSON
                    headers = {
                        "Authorization": f"Bearer {openai_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # Parameter payload JSON untuk model gpt-4o-mini dengan input prompt_rag
                    payload = {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "user", "content": prompt_rag}
                        ],
                        "max_completion_tokens": 1000 # Batasi panjang token jawaban AI maksimal 1000 token
                    }
                    
                    # Jalankan HTTP POST request secara langsung ke API OpenAI dengan batas waktu timeout 30 detik
                    response = requests.post(url_openai, json=payload, headers=headers, timeout=30)
                    
                    # Jika server OpenAI merespons sukses (HTTP 200 OK)
                    if response.status_code == 200:
                        res_data = response.json()
                        choices = res_data.get("choices", [])
                        if choices and isinstance(choices, list):
                            # Ambil teks jawaban utama yang dihasilkan oleh model LLM
                            jawaban_ai = choices[0].get("message", {}).get("content", "")
                        else:
                            jawaban_ai = "Format respons choices dari OpenAI tidak sesuai."
                    else:
                        # Jika server OpenAI mengembalikan kode kesalahan (HTTP non-200)
                        jawaban_ai = f"⚠️ **Error API OpenAI ({response.status_code}):** {response.text}"
 
                    # Cek apakah user meminta referensi/tafsir secara eksplisit
                    butuh_referensi = any(kw in query_user.lower() for kw in ["referensi", "rujukan", "sumber", "tafsir"])

                    # Menyusun kartu informasi daftar rujukan dokumen lokal untuk diletakkan di akhir jawaban
                    sumber_unik = list(set(sumber_list))
                    info_sumber = ""
                    if sumber_unik and butuh_referensi:
                        # Pisahkan dan urutkan rujukan agar Al-Qur'an tampil paling atas, disusul artikel tafsir
                        sumber_quran = sorted([s for s in sumber_unik if "Al-Qur'an" in s])
                        sumber_artikel = sorted([s for s in sumber_unik if "Al-Qur'an" not in s])
                        sumber_sorted = sumber_quran + sumber_artikel
                        
                        # Bungkus nama sumber rujukan ke dalam elemen tag list <li> HTML
                        list_items = "".join([f"<li style='margin-bottom: 4px;'>{s}</li>" for s in sumber_sorted])
                        # Desain kartu rujukan berlatar glassmorphic hijau premium menggunakan tag div HTML
                        info_sumber = f'''\n\n<div class="reference-card">
📖 <b>Untuk jawaban ini, Anda dapat merujuk pada:</b>
<ul style="margin-top: 6px; margin-bottom: 0; padding-left: 20px; color: #a7f3d0;">
{list_items}
</ul>
</div>'''
                    
                    # Gabungkan teks jawaban utama dari asisten AI dengan kartu rujukan lokal
                    jawaban_final = jawaban_ai + info_sumber
                    
                    # Tampilkan jawaban akhir pada placeholder dengan memformat teks Arab terlebih dahulu agar indah
                    message_placeholder.markdown(format_output_with_arabic(jawaban_final), unsafe_allow_html=True)
                    # Simpan jawaban asisten ke session state untuk persistensi tampilan
                    st.session_state.messages.append({"role": "assistant", "content": jawaban_final})
                    # Simpan seluruh riwayat percakapan terbaru ke file JSON lokal
                    save_chat_history(user_id, st.session_state.messages)
                except Exception as e:
                    # Tangani error jika terjadi pengecualian saat eksekusi API atau pemrosesan output
                    st.error(f"Terjadi kesalahan saat memanggil API: {e}")
                    