import pandas as pd
import numpy as np
import torch
import faiss
import os
import pickle
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor

# Jalur Berkas Dokumen
TAFSIR_FILE = "data/artikel_tafsir_clean.csv"
QURAN_FILE = "data/quran_clean.csv"
DB_DIR = "faiss_vdb"
INDEX_FILE = os.path.join(DB_DIR, "index.faiss")
METADATA_FILE = os.path.join(DB_DIR, "metadata.pkl")

# Deteksi perangkat GPU untuk mempercepat proses embedding jika tersedia (diinisialisasi malas)
device = None
tokenizer = None
model = None
preprocessor = None

def init_models(custom_model=None, custom_tokenizer=None, custom_preprocessor=None):
    global device, tokenizer, model, preprocessor
    if custom_model is not None:
        model = custom_model
        tokenizer = custom_tokenizer
        preprocessor = custom_preprocessor
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        if model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Menggunakan perangkat: {device.upper()}")
            print("Memuat model embedding...")
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            preprocessor = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02")

def get_embedding(text):
    init_models()
    if any("\u0600" <= c <= "\u06FF" for c in str(text)):
        cleaned_text = preprocessor.preprocess(str(text))
    else:
        cleaned_text = str(text)
    inputs = tokenizer(cleaned_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

def get_embeddings_batched(texts, batch_size=128, custom_model=None, custom_tokenizer=None, custom_preprocessor=None, progress_callback=None):
    init_models(custom_model, custom_tokenizer, custom_preprocessor)
    embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]
        cleaned_texts = []
        for text in batch_texts:
            if any("\u0600" <= c <= "\u06FF" for c in str(text)):
                cleaned_texts.append(preprocessor.preprocess(str(text)))
            else:
                cleaned_texts.append(str(text))
        
        inputs = tokenizer(cleaned_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
        
        completed = min(i + batch_size, total)
        progress_msg = f"Proses embedding: {completed} / {total} teks..."
        print(progress_msg, end="\r")
        if progress_callback:
            progress_callback(completed, total, progress_msg)
    print()
    return np.vstack(embeddings).astype('float32')

def build_database(custom_model=None, custom_tokenizer=None, custom_preprocessor=None, progress_callback=None):
    texts_to_embed = []
    metadata = []

    # 1. MEMPROSES FILE ARTIKEL TAFSIR
    if os.path.exists(TAFSIR_FILE):
        print(f"Membaca file artikel tafsir dari '{TAFSIR_FILE}'...")
        if progress_callback:
            progress_callback(0, 100, "Membaca file artikel tafsir...")
        df_tafsir = pd.read_csv(TAFSIR_FILE)
        print("Memotong artikel menjadi paragraf...")
        for idx, row in df_tafsir.iterrows():
            judul_artikel = row['judul']
            isi_artikel = str(row['isi'])
            
            if pd.isna(isi_artikel) or isi_artikel.strip() == "":
                continue
                
            # STRATEGI CHUNKING: Potong teks artikel berdasarkan paragraf baru (\n)
            paragraphs = [p.strip() for p in isi_artikel.split("\n") if p.strip()]
            
            # Jika teks tidak memiliki enter, pecah berdasarkan tanda titik (.)
            if len(paragraphs) == 1:
                paragraphs = [p.strip() + "." for p in isi_artikel.split(".") if p.strip()]

            for chunk in paragraphs:
                if len(chunk) < 20: # Abaikan potongan teks/kalimat yang terlalu pendek
                    continue
                
                texts_to_embed.append(chunk)
                metadata.append({
                    "teks": chunk,
                    "sumber": f"Artikel: {judul_artikel}"
                })
    else:
        print(f"Warning: File artikel tafsir '{TAFSIR_FILE}' tidak ditemukan!")
        if progress_callback:
            progress_callback(0, 100, "Warning: File artikel tafsir tidak ditemukan!")

    # 2. MEMPROSES FILE QURAN_CLEAN
    if os.path.exists(QURAN_FILE):
        print(f"Membaca file Al-Qur'an dari '{QURAN_FILE}'...")
        if progress_callback:
            progress_callback(0, 100, "Membaca file Al-Qur'an...")
        df_quran = pd.read_csv(QURAN_FILE)
        print("Mempersiapkan data ayat Al-Qur'an...")
        
        # Iterasi setiap ayat Al-Qur'an
        for idx, row in df_quran.iterrows():
            surah_name = row['surah_name']
            surah_id = row['surah_id']
            ayah = row['ayah']
            arabic = row['arabic']
            translation = row['translation']
            
            # Gabungkan ayat Arab dan terjemahan untuk isi teks yang disimpan
            teks_lengkap = f"QS. {surah_name} ({surah_id}:{ayah}):\n{arabic}\nTerjemahan: {translation}"
            
            # 1. Embedding untuk terjemahan (bahasa Indonesia)
            texts_to_embed.append(translation)
            metadata.append({
                "teks": teks_lengkap,
                "sumber": f"Al-Qur'an: QS. {surah_name} ({surah_id}:{ayah})"
            })

            # 2. Embedding untuk teks Arab (agar mendukung query bahasa Arab)
            texts_to_embed.append(arabic)
            metadata.append({
                "teks": teks_lengkap,
                "sumber": f"Al-Qur'an: QS. {surah_name} ({surah_id}:{ayah})"
            })
    else:
        print(f"Warning: File Al-Qur'an '{QURAN_FILE}' tidak ditemukan!")
        if progress_callback:
            progress_callback(0, 100, "Warning: File Al-Qur'an tidak ditemukan!")

    # 3. MEMBANGUN DAN MENYIMPAN DATABASE FAISS
    if texts_to_embed:
        print(f"Mulai proses embedding untuk total {len(texts_to_embed)} data...")
        if progress_callback:
            progress_callback(0, len(texts_to_embed), f"Mulai proses embedding untuk total {len(texts_to_embed)} data...")
            
        new_embeddings = get_embeddings_batched(
            texts_to_embed, 
            batch_size=128, 
            custom_model=custom_model, 
            custom_tokenizer=custom_tokenizer, 
            custom_preprocessor=custom_preprocessor,
            progress_callback=progress_callback
        )
        
        print("Membangun index FAISS...")
        if progress_callback:
            progress_callback(len(texts_to_embed), len(texts_to_embed), "Membangun index FAISS...")
        
        index = faiss.IndexFlatL2(384)
        index.add(new_embeddings)
        
        if not os.path.exists(DB_DIR):
            os.makedirs(DB_DIR)
            
        # Kunci file database ke folder faiss_vdb
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(metadata, f)
            
        success_msg = f"Sukses! Berhasil memproses total {len(texts_to_embed)} data dan menyimpannya ke '{DB_DIR}'."
        print(success_msg)
        if progress_callback:
            progress_callback(len(texts_to_embed), len(texts_to_embed), success_msg)
    else:
        err_msg = "Error: Tidak ada data yang berhasil diproses untuk membuat database."
        print(err_msg)
        if progress_callback:
            progress_callback(0, 100, err_msg)

if __name__ == "__main__":
    build_database()

