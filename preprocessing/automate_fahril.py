import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    if not os.path.exists(path):
        print(f"Error File tidak ditemukan di: {path}")
        print("Pastikan file 'heart.csv' ada di folder 'data_raw'.")
        return None
    return pd.read_csv(path)

def preprocess_data(df):
    print("Memulai Preprocessing...")
    
    # 1. Cleaning Data
    df_clean = df.copy()
    awal = len(df_clean)
    df_clean = df_clean.drop_duplicates().dropna()
    print(f"Data dibersihkan. Duplikat/Null dihapus: {awal - len(df_clean)} baris")

    # 2. Mendeteksi Target
    possible_targets = ['Heart Disease Status', 'output', 'target']
    target_col = None
    for col in possible_targets:
        if col in df_clean.columns:
            target_col = col
            break
            
    if target_col is None:
        target_col = df_clean.columns[-1] # Fallback
    
    print(f"Target variabel: '{target_col}'")

    # Memisahkan Fitur (X) dan Target (y)
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # 3. Encoding Fitur (One-Hot Encoding untuk Male/Female, dll)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # 4. Encoding Target
    if y.dtype == 'object':
        y_encoded = y.map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
    else:
        y_encoded = y

    # 5. Scaling (Standarisasi)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # 6. Menggabungkan Kembali Kolom
    df_final = pd.DataFrame(X_scaled, columns=X_encoded.columns)
    
    df_final['target'] = y_encoded.values 

    return df_final

def main():
    print("Memulai Otomatisasi...")
    # Mendapatkan lokasi file script ini berada (folder preprocessing)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Lokasi dataset input (Naik satu level ke data_raw)
    project_root = os.path.dirname(base_dir)
    input_path = os.path.join(project_root, 'data_raw', 'heart.csv')
    
    # Lokasi output (Di dalam folder preprocessing/data_clean)
    output_path = os.path.join(base_dir, 'data_clean', 'heart_clean.csv')

    # Eksekusi
    df = load_data(input_path)
    
    if df is not None:
        try:
            df_result = preprocess_data(df)
            
            # Simpan Hasil
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_result.to_csv(output_path, index=False)
            
            print(f"File berhasil dibuat di:")
            print(f"{output_path}")
            print(f"Jumlah Kolom: {df_result.shape[1]}")
            print("Selesai")
            
        except Exception as e:
            print(f"Terjadi error pada kode: {e}")

if __name__ == "__main__":
    main()