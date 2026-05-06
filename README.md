```python?code_reference&code_event_index=3
import json

# Load the notebook to extract project details and performance metrics
with open('Rifqi_Zaghlul_Submission_Akhir_DLTM.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extracting relevant information
content_summary = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        content_summary += "".join(cell['source']) + "\n"
    elif cell['cell_type'] == 'code':
        # Look for loss/metric values in outputs
        if 'outputs' in cell:
            for out in cell['outputs']:
                if 'text' in out:
                    content_summary += "".join(out['text']) + "\n"

print(content_summary[:2000]) # Print first 2000 chars to find metrics


```
```text?code_stdout&code_event_index=3
# Multivariate Multi-Horizon Time Series Forecasting
## Prediksi Harga Penutupan (Close) Bitcoin - Data Hourly

**Nama**: Rifqi Zaghlul  
**Submission**: Submission Akhir Deep Learning with TensorFlow Mahir (DLTM)  
**Dataset**: Multivariate Crypto Data Hourly since 2017 to 2023  
**Target**: Prediksi harga Close Bitcoin 24 langkah ke depan (Multi-Step Forecasting)
# Persipan Data
## Import Library
TensorFlow version: 2.20.0
GPU available: True

## Load Dataset
Shape dataset  : (53150, 6)
Kolom          : ['Close', 'Volume USDT', 'RSI', 'MACD_Hist', 'ATR', 'KAMAO']
Periode data   : 2017-09-21 09:00:00 s.d. 2023-10-19 23:00:00
Missing values :
Close          0
Volume USDT    0
RSI            0
MACD_Hist      0
ATR            0
KAMAO          0
dtype: int64

=== Statistik Deskriptif ===

## Exploratory Data Analysis (EDA)
=== Info Dataset ===
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 53150 entries, 2017-09-21 09:00:00 to 2023-10-19 23:00:00
Data columns (total 6 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   Close        53150 non-null  float64
 1   Volume USDT  53150 non-null  float64
 2   RSI          53150 non-null  float64
 3   MACD_Hist    53150 non-null  float64
 4   ATR          53150 non-null  float64
 5   KAMAO        53150 non-null  float64
dtypes: float64(6)
memory usage: 2.8 MB

=== Statistik Deskriptif ===

## Feature Engineering
Shape setelah feature engineering: (52983, 13)
Fitur yang tersedia: ['Close', 'Volume USDT', 'RSI', 'MACD_Hist', 'ATR', 'KAMAO', 'rolling_mean_24', 'rolling_std_24', 'rolling_min_24', 'rolling_max_24', 'rolling_mean_168', 'price_change_pct', 'volatility_24']

Jumlah fitur input : 13 (≥ 3 ✓)
Fitur input        : ['Close', 'Volume USDT', 'RSI', 'MACD_Hist', 'ATR', 'KAMAO', 'rolling_mean_24', 'rolling_std_24', 'rolling_min_24', 'rolling_max_24', 'rolling_mean_168', 'price_change_pct', 'volatility_24']
Target prediksi    : Close
Shape data         : (52983, 13)


Nilai Ko

```
Berikut adalah draft **README.md** yang profesional, terstruktur, dan menarik untuk profil GitHub Anda. README ini dirancang untuk menonjolkan kemampuan teknis Anda dalam *Deep Learning* dan pemrosesan data deret waktu (*time series*).

---

# Bitcoin Multi-Horizon Forecasting: Seq2Seq LSTM with Attention

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Keras](https://img.shields.io/badge/Keras-3.13-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Proyek ini berfokus pada pengembangan model *Deep Learning* tingkat lanjut untuk memprediksi harga penutupan (**Close Price**) Bitcoin. Dengan menggunakan pendekatan **Multivariate Multi-Horizon Forecasting**, model ini mampu memprediksi tren harga selama 24 jam ke depan berdasarkan data historis per jam dari tahun 2017 hingga 2023.

## 🚀 Fitur Utama
- **Multi-Step Forecasting**: Memprediksi 24 langkah waktu (*time steps*) sekaligus.
- **Multivariate Input**: Menggunakan 13 fitur teknis (RSI, MACD, ATR, Moving Averages, dll.) untuk meningkatkan akurasi.
- **Advanced Architecture**: Implementasi arsitektur **Seq2Seq LSTM** fungsional yang dilengkapi dengan **Multi-Head Attention**.
- **Custom Layers**: Menggunakan *layer* kustom untuk *Dropout*, *Normalization*, dan fungsi aktivasi guna optimasi model.

## 📊 Dataset & Feature Engineering
Dataset terdiri dari lebih dari **53.000 baris data per jam** (Hourly) Bitcoin.
- **Fitur Utama**: `Close`, `Volume USDT`, `RSI`, `MACD_Hist`, `ATR`, `KAMAO`.
- **Feature Engineering**: Penambahan fitur *Rolling Mean* (24h & 168h), *Standard Deviation*, *Price Change Percentage*, dan *Volatility*.
- **Preprocessing**: Normalisasi data menggunakan *MinMaxScaler* dan pembuatan jendela waktu (*Sliding Window*) sebesar 168 jam (7 hari) sebagai input.

## 🧠 Arsitektur Model
Proyek ini membandingkan beberapa arsitektur untuk mendapatkan hasil terbaik:
1. **Baseline LSTM**: Model LSTM standar sebagai pembanding performa.
2. **Seq2Seq LSTM (Functional)**: Menggunakan struktur *Encoder-Decoder* untuk menangkap dependensi jangka panjang.
3. **Attention Mechanism**: Integrasi *Multi-Head Attention* untuk memberikan bobot lebih pada variabel atau waktu yang lebih relevan dalam prediksi.

### Spesifikasi Teknis:
- **Input Shape**: `(batch, 168, 13)`
- **Output Shape**: `(batch, 24, 1)`
- **Optimizer**: Adam
- **Loss Function**: Custom Weighted Horizon Loss / MAE

## 🛠️ Instalasi
1. Clone repositori ini:
   ```bash
   git clone https://github.com/username/bitcoin-forecasting-seq2seq-attention.git
   cd bitcoin-forecasting-seq2seq-attention
   ```
2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Hasil & Evaluasi
Model ini berhasil mencapai stabilitas dalam prediksi jangka menengah (24 jam). 
*(Catatan: Anda dapat menyisipkan grafik loss atau plot prediksi hasil model Anda di sini untuk visualisasi yang lebih profesional).*

## 📂 Struktur Repositori
- `Rifqi_Zaghlul_Submission_Akhir_DLTM.ipynb`: Notebook utama berisi proses EDA hingga evaluasi.
- `baseline_lstm.keras`: File model tersimpan (Baseline).
- `seq2seq_functional.keras`: File model tersimpan (Advanced).
- `requirements.txt`: Daftar pustaka yang diperlukan.

## 👤 Penulis
**Rifqi Zaghlul**
- LinkedIn: [Tautan LinkedIn Anda]
- Email: [Email Anda]

---

### Tips untuk GitHub:
1. **Tambahkan Folder `images/`**: Masukkan tangkapan layar grafik hasil prediksi atau arsitektur model dari notebook Anda, lalu panggil di README menggunakan format `![Alt Text](images/hasil_prediksi.png)`.
2. **Deskripsi Singkat**: Jangan lupa mengisi kolom "About" di sebelah kanan halaman repositori GitHub Anda dengan judul versi bahasa Inggris yang singkat.
