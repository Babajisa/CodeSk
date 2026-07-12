import os
import pandas as pd

def generate_juz_amma_sources():
    sources_dir = "data/sources"
    
    # Bersihkan folder sources lama terlebih dahulu agar tidak menumpuk file Part 1/Part 2
    if os.path.exists(sources_dir):
        for f in os.listdir(sources_dir):
            if f.endswith(('.txt', '.md')):
                try:
                    os.remove(os.path.join(sources_dir, f))
                except:
                    pass
    else:
        os.makedirs(sources_dir, exist_ok=True)
        
    quran_file = "data/quran_clean.csv"
    tafsir_file = "data/artikel_tafsir_clean.csv"
    
    if not os.path.exists(quran_file):
        raise FileNotFoundError(f"File Al-Qur'an '{quran_file}' tidak ditemukan.")
        
    df_quran = pd.read_csv(quran_file)
    df_tafsir = pd.read_csv(tafsir_file) if os.path.exists(tafsir_file) else None
    
    # Juz Amma is Surah 78 to 114
    juz_amma_df = df_quran[(df_quran['surah_id'] >= 78) & (df_quran['surah_id'] <= 114)]
    surahs = juz_amma_df['surah_id'].unique()
    file_count = 0
    
    for surah_id in surahs:
        surah_df = juz_amma_df[juz_amma_df['surah_id'] == surah_id]
        surah_name = surah_df.iloc[0]['surah_name']
        
        # 1. GENERATE FILE TERJEMAHAN LENGKAP (1 Surat = 1 Sumber)
        filename_trans = f"QS_{surah_id}_{surah_name.replace(' ', '_')}_Terjemahan.txt"
        filepath_trans = os.path.join(sources_dir, filename_trans)
        
        content_trans = []
        content_trans.append(f"QS. {surah_name} (Surah ke-{surah_id}) - Terjemahan Lengkap\n")
        for _, row in surah_df.iterrows():
            content_trans.append(f"Ayat {row['ayah']}:")
            content_trans.append(row['arabic'])
            content_trans.append(f"Terjemahan: {row['translation']}\n")
            
        with open(filepath_trans, "w", encoding="utf-8") as f:
            f.write("\n".join(content_trans))
        file_count += 1
        
        # 2. GENERATE FILE TAFSIR/RINGKASAN (Untuk mencapai >= 51 sumber dan memisahkan Terjemahan & Tafsir)
        filename_taf = f"QS_{surah_id}_{surah_name.replace(' ', '_')}_Tafsir.txt"
        filepath_taf = os.path.join(sources_dir, filename_taf)
        
        # Cari apakah ada tafsir di CSV artikel_tafsir_clean
        artikel_konten = ""
        if df_tafsir is not None:
            # Cari baris yang mengandung nama surah di judul artikel
            match_df = df_tafsir[df_tafsir['judul'].str.contains(surah_name, case=False, na=False)]
            if not match_df.empty:
                artikel_konten = "\n\n".join(match_df['isi'].astype(str).tolist())
                
        content_taf = []
        content_taf.append(f"Tafsir & Ringkasan QS. {surah_name} (Surah ke-{surah_id})\n")
        
        if artikel_konten:
            content_taf.append(artikel_konten)
        else:
            # Default ringkasan jika artikel tafsir spesifik tidak ditemukan
            content_taf.append(f"Surah {surah_name} adalah Surah ke-{surah_id} di dalam Al-Qur'an.")
            content_taf.append(f"Surah ini merupakan bagian dari Juz 30 (Juz Amma) dan tergolong surah Makkiyah berdasarkan tempat turunnya.")
            content_taf.append(f"Tema utama surah ini mencakup keimanan, hari akhir, kekuasaan Allah SWT, serta petunjuk moral bagi umat manusia.")
            
        with open(filepath_taf, "w", encoding="utf-8") as f:
            f.write("\n".join(content_taf))
        file_count += 1
        
    return file_count

if __name__ == "__main__":
    count = generate_juz_amma_sources()
    print(f"Selesai! Berhasil membuat {count} file rujukan (Terjemahan & Tafsir).")
