import os
import streamlit as st
import numpy as np
import torch
import faiss
import pickle
import requests
import json
import uuid
import re
import shutil
from dotenv import load_dotenv
import generate_sources  # Import helper untuk generate file Juz Amma

# Meningkatkan batas waktu (timeout) koneksi ke Hugging Face Hub (defaultnya sangat pendek: 10s)
os.environ["HF_HUB_ETAG_TIMEOUT"] = "1000"
# Hapus overriding endpoint mirror jika ada, gunakan endpoint resmi HF demi kecocokan sertifikat SSL
if "HF_ENDPOINT" in os.environ:
    del os.environ["HF_ENDPOINT"]

# Membaca file .env di awal aplikasi
load_dotenv()

# Tampilan halaman Streamlit dengan Layout Wide (NotebookLM Style)
st.set_page_config(page_title="Asisten Tanya Jawab Islami", layout="wide")

# Custom styling untuk tampilan premium NotebookLM
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Amiri:ital,wght@0,400;0,700;1,400;1,700&display=swap');

/* Main font styling */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Background gradient styling */
.stApp {
    background: linear-gradient(135deg, #070d19 0%, #0f0b24 50%, #02040a 100%) !important;
    color: #e2e8f0 !important;
}

/* Sidebar (Sources Panel) Styling */
[data-testid="stSidebar"] {
    background-color: #0b111e !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    transition: width 0.3s ease, min-width 0.3s ease !important;
}

/* Wide sizing ONLY when expanded */
[data-testid="stSidebar"][aria-expanded="true"], 
[data-testid="stSidebar"][data-collapsed="false"] {
    width: 420px !important;
    min-width: 400px !important;
    max-width: 500px !important;
}

/* Make sidebar content scrollable and formatted */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 12px !important;
}

/* Source Card Container */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    margin-bottom: 4px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(16, 185, 129, 0.4) !important;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.15) !important;
    background-color: rgba(255, 255, 255, 0.04) !important;
}

/* Custom header/title gradient styling */
.main-title {
    background: linear-gradient(90deg, #60a5fa 0%, #34d399 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 700 !important;
    font-size: 2.2rem !important;
    margin-bottom: 0.2rem !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1) !important;
}

.sub-title {
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    margin-bottom: 1.5rem !important;
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
    transform: translateY(-1px);
    border-color: rgba(16, 185, 129, 0.3) !important;
    box-shadow: 0 6px 24px rgba(16, 185, 129, 0.1) !important;
}

/* Make Arabic text beautiful and readable */
.arabic-text {
    font-family: 'Amiri', serif !important;
    font-size: 1.65rem !important;
    line-height: 2.3 !important;
    direction: rtl !important;
    text-align: right !important;
    color: #ffd700 !important;
    padding: 12px 18px !important;
    background: rgba(255, 255, 255, 0.02) !important;
    border-radius: 12px !important;
    margin-top: 12px !important;
    margin-bottom: 12px !important;
    border-right: 4px solid #ffd700 !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.1) !important;
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
    border-radius: 10px !important;
    padding: 6px 16px !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15) !important;
    transition: all 0.3s ease !important;
}

.stButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(16, 185, 129, 0.25) !important;
}

/* Compact Checkboxes in Sidebar */
.stCheckbox {
    margin-bottom: -10px !important;
}

/* Prompt Badge */
.prompt-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    color: #a7f3d0;
    background-color: rgba(16, 185, 129, 0.08);
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid rgba(16, 185, 129, 0.2);
    width: fit-content;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.05);
}

.prompt-badge-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
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
SOURCES_DIR = "data/sources"

if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)
if not os.path.exists(SOURCES_DIR):
    os.makedirs(SOURCES_DIR)

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
# LOAD MODEL & DATABASE (Latar Belakang)
# ==========================================
@st.cache_resource
def load_resources():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    from transformers import AutoTokenizer, AutoModel
    from arabert.preprocess import ArabertPreprocessor
    
    # Mencoba memuat model dan tokenizer dari Hugging Face (menggunakan mirror jika dikonfigurasi)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        st.error(f"DEBUG - Error Koneksi HF (repr): {repr(e)}")
        # Fallback menggunakan file yang sudah terunduh di cache lokal (mode offline) jika koneksi internet terganggu
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception as e2:
            st.error(f"DEBUG - Error Cache Lokal (repr): {repr(e2)}")
            st.stop()
            
    preprocessor = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02")
    
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
    else:
        index, metadata = None, None
        
    return preprocessor, tokenizer, model, index, metadata

preprocessor, tokenizer, model, index, metadata = load_resources()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text):
    if any("\u0600" <= c <= "\u06FF" for c in str(text)):
        cleaned_text = preprocessor.preprocess(str(text))
    else:
        cleaned_text = str(text)
    inputs = tokenizer(cleaned_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return mean_pooling(outputs, inputs['attention_mask']).numpy().flatten()

def keyword_search(query, metadata, k=5):
    words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
    stop_words = {'mengapa', 'bagaimana', 'apakah', 'adakah', 'yang', 'dalam', 'dan', 'atau', 'untuk', 'dengan', 'dari', 'pada', 'saya', 'bisa', 'dapat'}
    keywords = [w for w in words if w not in stop_words]
    if not keywords:
        return []
        
    scores = []
    for idx, item in enumerate(metadata):
        text_lower = item["teks"].lower()
        sumber_lower = item["sumber"].lower()
        score = 0
        for kw in keywords:
            if kw in text_lower:
                score += 1
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    score += 2
            if kw in sumber_lower:
                score += 1
        if score > 0:
            scores.append((idx, score))
            
    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, score in scores[:k]]

def format_output_with_arabic(text):
    arabic_pattern = r'([\u0600-\u06FF][\u0600-\u06FF\s\u064B-\u065F\u0670\u06D6-\u06ED\u06F0-\u06F9\u060C\u061B\u061F\(\)]*[\u0600-\u06FF])'
    def wrap_arabic(match):
        val = match.group(0).strip()
        if val:
            val_clean = val.replace('\n', '<br>')
            return f'<div class="arabic-text" dir="rtl">{val_clean}</div>'
        return match.group(0)
    return re.sub(arabic_pattern, wrap_arabic, text)

def check_prompt_relevance(prompt, api_key):
    url_openai = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    system_instruction = (
        "Anda adalah filter klasifikasi pertanyaan. Tugas Anda adalah menentukan apakah "
        "pertanyaan pengguna berkaitan dengan Al-Qur'an, Tafsir, Islam, atau dokumen rujukan yang disediakan. "
        "Jawablah HANYA dengan kata 'YES' jika berkaitan, atau 'NO' jika "
        "tidak berkaitan. Jangan berikan penjelasan apa pun, cukup satu kata saja."
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": 5,
        "temperature": 0.0
    }
    try:
        response = requests.post(url_openai, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            res_data = response.json()
            choices = res_data.get("choices", [])
            if choices:
                result = choices[0].get("message", {}).get("content", "").strip().upper()
                result = result.replace(".", "").replace("!", "")
                return "YES" in result or result == "YES"
    except:
        return True
    return True

# ==========================================
# MANAJEMEN SUMBER (LOAD & SAVE)
# ==========================================
def scan_sources_folder():
    if not os.path.exists(SOURCES_DIR):
        os.makedirs(SOURCES_DIR)
    
    files = [f for f in os.listdir(SOURCES_DIR) if f.endswith(('.txt', '.md'))]
    sources = []
    
    # Baca checkbox state yang ada di session state agar tidak hilang saat re-scan
    existing_checked = {}
    if "sources" in st.session_state:
        for src in st.session_state.sources:
            existing_checked[src["filename"]] = src["checked"]
            
    for idx, filename in enumerate(sorted(files)):
        filepath = os.path.join(SOURCES_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            try:
                with open(filepath, "r", encoding="latin-1") as f:
                    content = f.read()
            except Exception:
                content = "[Gagal membaca konten berkas]"
                
        # Clean title
        title = filename.replace(".txt", "").replace(".md", "").replace("_", " ")
        if title.startswith("QS "):
            # Format QS_78_An-Naba_Part_1 -> QS. 78 An-Naba [Part 1]
            title = title.replace("Part ", "[Part ").replace("Lengkap", "[Lengkap")
            if "[" in title:
                title += "]"
                
        word_count = len(content.split())
        checked = existing_checked.get(filename, True) # Checked by default jika baru
        
        sources.append({
            "id": filename,
            "filename": filename,
            "title": title,
            "content": content,
            "checked": checked,
            "word_count": word_count,
            "index": idx
        })
    return sources

# Inisialisasi daftar sumber di session state
if "sources" not in st.session_state:
    st.session_state.sources = scan_sources_folder()

# ==========================================
# DIALOG MODAL PREVIEW (Streamlit 1.34+)
# ==========================================
if "preview_source" not in st.session_state:
    st.session_state.preview_source = None

# Fallback Preview Modal (jika st.dialog tidak tersedia)
if st.session_state.preview_source:
    src_data = st.session_state.preview_source
    # Coba st.dialog
    if hasattr(st, "dialog"):
        @st.dialog("👁️ Detail Sumber Rujukan", width="large")
        def show_dialog(src):
            st.markdown(f"### {src['title']}")
            st.markdown(f"**Nama Berkas:** `{src['filename']}` | **Jumlah Kata:** {src['word_count']}")
            st.divider()
            st.text_area("Konten Berkas:", src["content"], height=380, disabled=True)
            if st.button("Tutup", key="close_dialog"):
                st.session_state.preview_source = None
                st.rerun()
        show_dialog(src_data)
    else:
        # Fallback exp/info
        st.sidebar.info(f"👁️ **Preview: {src_data['title']}**\n\n{src_data['content'][:500]}...")
        if st.sidebar.button("Tutup Preview", key="close_fallback_prev"):
            st.session_state.preview_source = None
            st.rerun()

# ==========================================
# 1. PANEL KIRI: SOURCES SIDEBAR
# ==========================================
st.sidebar.markdown(f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 5px;'><h3 style='margin:0; font-size:1.5rem; color:white; font-weight:600; font-family:\"Outfit\", sans-serif;'>Sources</h3><span style='color: #94a3b8; font-size: 0.95rem; font-weight: 500;'>({len(st.session_state.sources)})</span></div>", unsafe_allow_html=True)

# OpenAI API Key Input & Status
openai_api_key = os.getenv("OPENAI_API_KEY") or ""
if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
openai_api_key = openai_api_key.strip('"').strip("'").strip()

if openai_api_key:
    st.sidebar.caption("✅ OpenAI Terhubung")
else:
    openai_api_key = st.sidebar.text_input("Masukkan OpenAI API Key:", type="password")
    if openai_api_key:
        openai_api_key = openai_api_key.strip()
        st.sidebar.success("✅ API Key dimasukkan")

# Aksi Penambahan Sumber
with st.sidebar.popover("➕ Add sources", use_container_width=True):
    st.markdown("##### Unggah File Baru")
    uploaded_files = st.file_uploader("Pilih file (.txt, .md):", accept_multiple_files=True, type=["txt", "md"])
    if uploaded_files:
        for uploaded_file in uploaded_files:
            target_path = os.path.join(SOURCES_DIR, uploaded_file.name)
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.session_state.sources = scan_sources_folder()
        st.success(f"Berhasil mengunggah {len(uploaded_files)} file!")
        st.rerun()
        
    st.divider()
    st.markdown("##### Tulis Catatan Manual")
    note_title = st.text_input("Judul Catatan:")
    note_content = st.text_area("Isi Catatan:")
    if st.button("Simpan Catatan", use_container_width=True):
        if note_title and note_content:
            filename = f"catatan_{note_title.replace(' ', '_').lower()}.txt"
            target_path = os.path.join(SOURCES_DIR, filename)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(note_content)
            st.session_state.sources = scan_sources_folder()
            st.success("Catatan berhasil disimpan!")
            st.rerun()
        else:
            st.error("Judul dan isi catatan tidak boleh kosong!")
            
    st.divider()
    st.markdown("##### Database Skripsi")
    if st.button("✨ Load Contoh Juz Amma (72 files)", use_container_width=True):
        with st.spinner("Mengekstrak terjemahan Juz Amma..."):
            try:
                count = generate_sources.generate_juz_amma_sources()
                st.session_state.sources = scan_sources_folder()
                st.success(f"Sukses memuat {count} file terjemahan Juz Amma!")
                st.rerun()
            except Exception as e:
                st.error(f"Gagal: {e}")

# Tombol Aksi Massal (Pilih Semua / Bersihkan)
col_sel_all, col_sel_none = st.sidebar.columns(2)
if col_sel_all.button("Pilih Semua", use_container_width=True):
    for src in st.session_state.sources:
        src["checked"] = True
    st.rerun()
if col_sel_none.button("Kosongkan", use_container_width=True):
    for src in st.session_state.sources:
        src["checked"] = False
    st.rerun()

# Kolom Pencarian Sumber
search_query = st.sidebar.text_input("🔍 Cari sumber...", value="", placeholder="Cari nama surah/file...")

# Filter sumber berdasarkan pencarian
filtered_sources = []
for src in st.session_state.sources:
    if not search_query or search_query.lower() in src["title"].lower() or search_query.lower() in src["filename"].lower():
        filtered_sources.append(src)

# Tampilkan list sumber rujukan
st.sidebar.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
for src in filtered_sources:
    with st.sidebar.container():
        # Kolom checkbox, preview eye, dan delete trash
        col_chk, col_eye, col_del = st.columns([7, 1.5, 1.5])
        
        # Checkbox untuk memilih sumber
        with col_chk:
            val = st.checkbox(
                f"📄 {src['title'][:25]}...", 
                value=src["checked"], 
                key=f"check_{src['filename']}",
                help=f"{src['title']} ({src['word_count']} kata)"
            )
            # Update status di session state langsung jika ada perubahan
            if val != src["checked"]:
                st.session_state.sources[src["index"]]["checked"] = val
                st.rerun()
                
        # Tombol Eye untuk Detail Preview
        with col_eye:
            if st.button("👁️", key=f"eye_{src['filename']}", help="Preview isi berkas"):
                st.session_state.preview_source = src
                st.rerun()
                
        # Tombol Trash untuk Hapus Sumber
        with col_del:
            if st.button("🗑️", key=f"trash_{src['filename']}", help="Hapus berkas"):
                filepath = os.path.join(SOURCES_DIR, src["filename"])
                if os.path.exists(filepath):
                    os.remove(filepath)
                st.session_state.sources = scan_sources_folder()
                st.success(f"Terhapus: {src['title']}")
                st.rerun()

# Tombol Hapus Riwayat Chat & Reset Database di bagian bawah sidebar
st.sidebar.write("---")
if st.sidebar.button("🧹 Hapus Riwayat Chat", use_container_width=True):
    filepath = os.path.join(HISTORY_DIR, f"{user_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
    st.session_state.messages = []
    st.sidebar.success("Riwayat chat berhasil dibersihkan!")
    st.rerun()

# ==========================================
# 2. PANEL UTAMA: CHAT & PROMPTING
# ==========================================
st.markdown('<div class="main-title">🕌 Asisten Pintar Tafsir AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Model grounded AI: Ajukan pertanyaan Anda berdasarkan berkas rujukan aktif di panel kiri.</div>', unsafe_allow_html=True)

# Hitung jumlah sumber aktif
active_sources = [s for s in st.session_state.sources if s["checked"]]
num_active = len(active_sources)

# Tampilkan warning jika database FAISS utama kosong
if index is None or metadata is None:
    st.warning("⚠️ Database utama (`faiss_vdb`) belum dibangun. Memerlukan database utama untuk fallback pencarian global.")
    if st.button("Membangun Database Sekarang", type="primary"):
        with st.status("Sedang memproses database...", expanded=True) as status:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def streamlit_callback(completed, total, msg):
                pct = completed / total if total > 0 else 0.0
                progress_bar.progress(max(0.0, min(1.0, pct)))
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
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                status.update(label="❌ Gagal membangun database!", state="error", expanded=True)
                st.error(f"Terjadi kesalahan: {e}")

# Inisialisasi chat messages
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history(user_id)

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(format_output_with_arabic(message["content"]), unsafe_allow_html=True)

# Tampilan Status Sumber Aktif (NotebookLM style badge) diatas Chat Input
badge_text = f"Menanyakan {num_active} sumber rujukan aktif" if num_active > 0 else "Pencarian global (tidak ada sumber terpilih)"
st.markdown(f"""
<div class="prompt-badge">
    <span class="prompt-badge-dot" style="background-color: {'#10b981' if num_active > 0 else '#f59e0b'}; box-shadow: 0 0 8px {'#10b981' if num_active > 0 else '#f59e0b'};"></span>
    {badge_text}
</div>
""", unsafe_allow_html=True)

# Input Chat
if query_user := st.chat_input("Ketik pertanyaan Anda tentang Al-Qur'an di sini..."):
    # Tampilkan chat user di UI
    with st.chat_message("user"):
        st.markdown(format_output_with_arabic(query_user), unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": query_user})
    save_chat_history(user_id, st.session_state.messages)
    
    # Validasi API Key
    if not openai_api_key:
        with st.chat_message("assistant"):
            msg = "⚠️ OpenAI API Key tidak ditemukan. Harap masukkan API Key di sidebar kiri."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            save_chat_history(user_id, st.session_state.messages)
    else:
        # Validasi Relevansi
        with st.spinner("Memvalidasi topik pertanyaan..."):
            is_relate = check_prompt_relevance(query_user, openai_api_key)
            
        if not is_relate:
            with st.chat_message("assistant"):
                msg = "Maaf, sistem ini dirancang khusus untuk menjawab pertanyaan seputar keislaman, Al-Qur'an, dan dokumen rujukan aktif Anda. Pertanyaan Anda tampaknya berada di luar topik tersebut."
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                save_chat_history(user_id, st.session_state.messages)
        else:
            with st.spinner("Menelusuri dan merumuskan jawaban..."):
                context_str = ""
                sources_used = []
                
                # JIKA ADA SUMBER RUJUKAN YANG DICENTANG (NotebookLM Mode)
                if num_active > 0:
                    # Lakukan pencarian relevansi kata kunci pada sumber aktif
                    words = [w.lower() for w in re.findall(r'\w+', query_user) if len(w) > 2]
                    stop_words = {'mengapa', 'bagaimana', 'apakah', 'adakah', 'yang', 'dalam', 'dan', 'atau', 'untuk', 'dengan', 'dari', 'pada', 'saya', 'bisa', 'dapat', 'tuliskan', 'terjemahan', 'surat', 'surah'}
                    keywords = [w for w in words if w not in stop_words]
                    
                    scored_sources = []
                    for s in active_sources:
                        score = 0
                        content_lower = s["content"].lower()
                        title_lower = s["title"].lower()
                        
                        for kw in keywords:
                            if kw in title_lower:
                                score += 15 # Prioritas utama jika keyword ada di judul
                            if kw in content_lower:
                                score += 2 # Jika keyword ada di isi konten
                                
                        if score > 0:
                            scored_sources.append((s, score))
                            
                    scored_sources.sort(key=lambda x: x[1], reverse=True)
                    
                    # Tentukan sumber yang relevan untuk dikirim ke LLM
                    if scored_sources:
                        # Ambil maksimal 5 sumber teratas yang relevan
                        relevant_sources = [item[0] for item in scored_sources[:5]]
                    else:
                        # Jika tidak ada kecocokan kata kunci (misal tanya umum/ringkasan),
                        # ambil maksimal 5 sumber pertama dari sumber aktif
                        relevant_sources = active_sources[:5]
                        
                    total_chars = sum(len(s["content"]) for s in relevant_sources)
                    
                    # Kasus A: Ukuran total sumber terpilih kecil -> masukkan semua teks
                    if total_chars <= 100000:
                        selected_texts = []
                        for s in relevant_sources:
                            selected_texts.append(f"--- SUMBER: {s['title']} ({s['filename']}) ---\n{s['content']}\n")
                            sources_used.append(s["title"])
                        context_str = "\n".join(selected_texts)
                    
                    # Kasus B: Ukuran total besar -> lakukan RAG keyword / overlap search
                    else:
                        st.info("Ukuran dokumen rujukan cukup besar. Mencari paragraf paling relevan dari sumber aktif...")
                        paragraphs = []
                        for s in active_sources:
                            # Split konten menjadi paragraf
                            paras = [p.strip() for p in s["content"].split("\n") if len(p.strip()) > 30]
                            for p in paras:
                                paragraphs.append({"title": s["title"], "filename": s["filename"], "text": p})
                        
                        # Hitung relevansi kata kunci sederhana
                        words = [w.lower() for w in re.findall(r'\w+', query_user) if len(w) > 2]
                        scored_paras = []
                        for p in paragraphs:
                            score = 0
                            for w in words:
                                if w in p["text"].lower():
                                    score += 1
                            if score > 0:
                                scored_paras.append((p, score))
                                
                        scored_paras.sort(key=lambda x: x[1], reverse=True)
                        top_paras = scored_paras[:15]
                        
                        selected_texts = []
                        for p, score in top_paras:
                            selected_texts.append(f"--- RUJUKAN DOKUMEN: {p['title']} ---\n{p['text']}\n")
                            if p["title"] not in sources_used:
                                sources_used.append(p["title"])
                        context_str = "\n".join(selected_texts) if selected_texts else "Tidak ditemukan paragraf yang relevan di berkas aktif."
                
                # JIKA TIDAK ADA SUMBER RUJUKAN YANG DICENTANG -> Fallback ke Database FAISS Utama
                else:
                    query_vector = np.array([get_embedding(query_user)]).astype('float32')
                    norm_q = np.linalg.norm(query_vector)
                    if norm_q > 1e-9:
                        query_vector = query_vector / norm_q
                    
                    # Pencarian FAISS & Keyword
                    D, I = index.search(query_vector, 5)
                    semantic_indices = [idx for idx in I.flatten() if idx != -1 and idx < len(metadata)]
                    keyword_indices = keyword_search(query_user, metadata, k=5)
                    
                    combined_indices = []
                    seen_idx = set()
                    for idx in semantic_indices + keyword_indices:
                        if idx not in seen_idx:
                            seen_idx.add(idx)
                            combined_indices.append(idx)
                    
                    combined_indices = combined_indices[:10]
                    seen_texts = set()
                    quran_list = []
                    tafsir_list = []
                    
                    for idx in combined_indices:
                        item = metadata[idx]
                        if item['teks'] not in seen_texts:
                            seen_texts.add(item['teks'])
                            if "Al-Qur'an" in item['sumber']:
                                quran_list.append(item['teks'])
                            else:
                                tafsir_list.append(f"[{item['sumber']}]\n{item['teks']}")
                            sources_used.append(item['sumber'])
                            
                    quran_string = "\n\n".join(quran_list) if quran_list else "Tidak ada rujukan ayat Al-Qur'an langsung."
                    tafsir_string = "\n\n".join(tafsir_list) if tafsir_list else "Tidak ada rujukan tafsir."
                    
                    context_str = f"--- RUJUKAN AYAT AL-QUR'AN ---\n{quran_string}\n\n--- RUJUKAN ARTIKEL TAFSIR ---\n{tafsir_string}"

                # Menyusun prompt grounding
                if num_active > 0:
                    prompt_rag = f"""Anda adalah asisten AI yang sangat patuh dan hanya boleh bersumber pada Rujukan Dokumen yang diberikan di bawah ini. Anda TIDAK diperbolehkan menggunakan pengetahuan eksternal Anda atau berimprovisasi.
 
Berikut adalah Rujukan Dokumen aktif Anda:
 
{context_str}
 
ATURAN GROUNDING SANGAT KETAT:
1. Jawablah pertanyaan pengguna HANYA menggunakan fakta dan informasi yang tertulis langsung di dalam Rujukan Dokumen di atas.
2. JANGAN menggunakan pengetahuan umum, asumsi, penafsiran di luar dokumen, atau sumber rujukan eksternal lainnya.
3. JIKA Rujukan Dokumen di atas tidak mengandung informasi yang dibutuhkan untuk menjawab pertanyaan, Anda WAJIB menjawab secara jujur dan singkat: "Maaf, informasi tidak ditemukan dalam dokumen rujukan aktif Anda."
4. JIKA terdapat kutipan ayat Al-Qur'an bahasa Arab yang relevan di dalam rujukan dokumen, Anda wajib menampilkan teks Arab asli beserta terjemahannya secara lengkap (verbatim) di dalam jawaban Anda.
 
PERTANYAAN PENGGUNA:
{query_user}
 
JAWABAN GROUNDED KETAT:"""
                else:
                    prompt_rag = f"""Anda adalah asisten ahli tafsir Al-Qur'an yang andal dan objektif. Tugas Anda adalah menjawab pertanyaan pengguna secara jelas dan akurat berdasarkan Rujukan Dokumen yang disediakan di bawah.
 
Berikut adalah Rujukan Dokumen yang Anda miliki:
 
{context_str}
 
PANDUAN MENJAWAB:
1. Jawablah pertanyaan HANYA menggunakan informasi dari Rujukan Dokumen di atas.
2. JIKA terdapat kutipan ayat Al-Qur'an bahasa Arab yang relevan di dalam rujukan dokumen, Anda WAJIB menampilkan teks Arab asli beserta terjemahannya secara lengkap (verbatim) di dalam jawaban Anda.
3. Berikan jawaban yang terstruktur dan mudah dipahami.
4. Jika jawaban sama sekali tidak dapat disimpulkan dari dokumen rujukan di atas, katakan secara jujur dan sopan bahwa informasi tidak ditemukan dalam rujukan dokumen aktif Anda.
 
PERTANYAAN PENGGUNA:
{query_user}
 
JAWABAN:"""

                # API Call ke OpenAI (gpt-4o-mini)
                try:
                    jawaban_ai = ""
                    url_openai = "https://api.openai.com/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {openai_api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt_rag}],
                        "max_completion_tokens": 1000,
                        "temperature": 0.3
                    }
                    
                    response = requests.post(url_openai, json=payload, headers=headers, timeout=30)
                    if response.status_code == 200:
                        res_data = response.json()
                        choices = res_data.get("choices", [])
                        if choices:
                            jawaban_ai = choices[0].get("message", {}).get("content", "")
                        else:
                            jawaban_ai = "Format respons OpenAI tidak sesuai."
                    else:
                        jawaban_ai = f"⚠️ **Error API OpenAI ({response.status_code}):** {response.text}"
                except Exception as e:
                    jawaban_ai = f"Terjadi kesalahan saat memanggil API: {e}"

                # Buat kartu rujukan
                info_sumber = ""
                if sources_used:
                    list_items = "".join([f"<li style='margin-bottom: 2px;'>{s}</li>" for s in sorted(list(set(sources_used)))])
                    info_sumber = f'''\n\n<div class="reference-card">
📖 <b>Dirujuk dari sumber aktif:</b>
<ul style="margin-top: 4px; margin-bottom: 0; padding-left: 20px; color: #a7f3d0; font-size: 0.9rem;">
{list_items}
</ul>
</div>'''
                
                jawaban_final = jawaban_ai + info_sumber
                
                # Render jawaban di UI
                with st.chat_message("assistant"):
                    st.markdown(format_output_with_arabic(jawaban_final), unsafe_allow_html=True)
                    
                st.session_state.messages.append({"role": "assistant", "content": jawaban_final})
                save_chat_history(user_id, st.session_state.messages)
                st.rerun()