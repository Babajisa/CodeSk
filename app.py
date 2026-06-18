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

# Membaca file .env di awal aplikasi
load_dotenv()

# Tampilan halaman Streamlit minimalis berfokus di tengah
st.set_page_config(page_title="Asisten Tanya Jawab Islami", layout="centered")

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
st.sidebar.info("Aplikasi ini menggunakan model GPT-5.4 mini resmi dari OpenAI untuk menjawab pertanyaan berdasarkan rujukan Al-Qur'an dan Tafsir.")

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

def get_embedding(text):
    if any("\u0600" <= c <= "\u06FF" for c in str(text)):
        cleaned_text = preprocessor.preprocess(str(text))
    else:
        cleaned_text = str(text)
    inputs = tokenizer(cleaned_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy().flatten()

def check_prompt_relevance(prompt, api_key):
    url_openai = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    system_instruction = (
        "Anda adalah filter klasifikasi pertanyaan. Tugas Anda adalah menentukan apakah "
        "pertanyaan pengguna berkaitan dengan Al-Qur'an, Tafsir, Hadits, keislaman, sejarah Islam, "
        "ibadah, atau hukum Islam. Jawablah HANYA dengan kata 'YES' jika berkaitan, atau 'NO' jika "
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
            if choices and isinstance(choices, list):
                result = choices[0].get("message", {}).get("content", "").strip().upper()
                result = result.replace(".", "").replace("!", "")
                return "YES" in result or result == "YES"
    except Exception as e:
        return True
    return True


# ==========================================
# 3. ANTARMUKA CHAT UTAMA
# ==========================================
st.title("🕌 Asisten Pintar Al-Qur'an & Tafsir AI")
st.write("Silakan masukkan pertanyaan Anda di kolom bawah untuk mencari rujukan ayat secara otomatis.")

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
            st.markdown(message["content"])

    if query_user := st.chat_input("Tanyakan sesuatu tentang Al-Qur'an atau Tafsir..."):
        with st.chat_message("user"):
            st.markdown(query_user)
        st.session_state.messages.append({"role": "user", "content": query_user})
        save_chat_history(user_id, st.session_state.messages)

        # Cek ketersediaan API key sebelum melakukan pencarian database
        api_ready = True
        if not openai_api_key:
            api_ready = False
            msg = "⚠️ Sistem tidak bisa menjawab karena `OPENAI_API_KEY` belum diisi."
        
        # Validasi relevansi prompt ke topik Al-Qur'an & Tafsir (Guardrail Cost)
        elif api_ready:
            with st.spinner("Memvalidasi topik pertanyaan..."):
                is_relate = check_prompt_relevance(query_user, openai_api_key)
                if not is_relate:
                    api_ready = False
                    msg = "Maaf, sistem ini dirancang khusus untuk menjawab pertanyaan seputar Al-Qur'an, Tafsir, dan keislaman. Pertanyaan Anda tampaknya berada di luar topik tersebut."

        if not api_ready:
            with st.chat_message("assistant"):
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                save_chat_history(user_id, st.session_state.messages)
        else:
            with st.spinner("Sedang menelusuri rujukan database lokal..."):
                query_vector = np.array([get_embedding(query_user)]).astype('float32')
                k_actual = min(3, len(metadata))
                D, I = index.search(query_vector, k_actual)
                
                konteks_list = []
                sumber_list = []
                seen_texts = set()
                
                for idx in I.flatten():
                    if idx != -1 and idx < len(metadata):
                        item = metadata[idx]
                        if item['teks'] not in seen_texts:
                            seen_texts.add(item['teks'])
                            konteks_list.append(item['teks'])
                            sumber_list.append(item['sumber'])

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                konteks_string = "\n".join([f"- {txt}" for txt in konteks_list])
                prompt_rag = f"""Anda adalah asisten ahli tafsir Al-Qur'an. Jawablah pertanyaan pengguna dengan sopan, jelas, dan akurat berdasarkan Rujukan Dokumen yang disediakan di bawah ini. Jika jawabannya tidak ada di dalam dokumen, katakan sejujurnya bahwa informasi tersebut tidak ditemukan dalam database Anda.

RUJUKAN DOKUMEN:
{konteks_string}

PERTANYAAN PENGGUNA:
{query_user}

JAWABAN:"""

                try:
                    jawaban_ai = ""
                    # Eksekusi Endpoint API OpenAI Resmi
                    url_openai = "https://api.openai.com/v1/chat/completions"
                    
                    headers = {
                        "Authorization": f"Bearer {openai_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "model": "gpt-5.4-mini",
                        "messages": [
                            {"role": "user", "content": prompt_rag}
                        ],
                        "max_completion_tokens": 1000
                    }
                    
                    response = requests.post(url_openai, json=payload, headers=headers, timeout=30)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        choices = res_data.get("choices", [])
                        if choices and isinstance(choices, list):
                            jawaban_ai = choices[0].get("message", {}).get("content", "")
                        else:
                            jawaban_ai = "Format respons choices dari OpenAI tidak sesuai."
                    else:
                        jawaban_ai = f"⚠️ **Error API OpenAI ({response.status_code}):** {response.text}"

                    # Menambahkan rujukan dokumen lokal di bagian bawah jawaban teks AI
                    sumber_unik = list(set(sumber_list))
                    info_sumber = ""
                    if sumber_unik:
                        info_sumber = f"\n\n*📚 Rujukan Referensi: {', '.join(sumber_unik)}*"
                    
                    jawaban_final = jawaban_ai + info_sumber
                    message_placeholder.markdown(jawaban_final)
                    st.session_state.messages.append({"role": "assistant", "content": jawaban_final})
                    save_chat_history(user_id, st.session_state.messages)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memanggil API: {e}")
                    