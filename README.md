# 🍌 BananaLens — Klasifikasi Kematangan Pisang
> Flask + TensorFlow (MobileNetV2) · Windows · Lokal

---

## 📁 Struktur Folder

```
banana_app/
├── app.py                  ← Flask backend
├── requirements.txt        ← Dependensi Python
├── model/
│   └── banana.h5           ← ⚠️ LETAKKAN MODEL DI SINI
├── templates/
│   └── index.html          ← Halaman web
└── static/
    ├── css/
    │   └── style.css
    ├── js/
    │   └── main.js
    └── uploads/            ← Gambar yang diunggah (otomatis dibuat)
```

---

## ⚙️ Langkah Setup (Windows)

### 1. Install Python
Unduh Python 3.10 atau 3.11 dari https://python.org  
Centang **"Add Python to PATH"** saat instalasi.

### 2. Buka Command Prompt / PowerShell
Arahkan ke folder proyek:
```
cd C:\Users\NamaAnda\banana_app
```

### 3. Buat Virtual Environment (opsional tapi disarankan)
```
python -m venv venv
venv\Scripts\activate
```

### 4. Install Library
```
pip install -r requirements.txt
```
> Proses ini membutuhkan waktu beberapa menit karena TensorFlow cukup besar (~500MB).

### 5. Salin Model
Salin file `banana.h5` dari Google Drive ke folder `model/`:
```
banana_app\model\banana.h5
```

### 6. Jalankan Server
```
python app.py
```

Output sukses:
```
✅ Model berhasil dimuat!
 * Running on http://127.0.0.1:5000
```

### 7. Buka Browser
Akses di: **http://127.0.0.1:5000**

---

## 🏷️ Label & Urutan Kelas

Model menggunakan urutan **alphabetical** dari nama folder dataset:

| Index | Folder Dataset   | Label Tampil     |
|-------|-----------------|-----------------|
| 0     | matang          | Matang 🍌        |
| 1     | mentah          | Mentah 🟢        |
| 2     | setengah-matang | Setengah Matang 🌿 |

> Jika urutan kelas berbeda di model Anda, sesuaikan `class_indices` di `app.py`:
> ```python
> class_indices = {'matang': 0, 'mentah': 1, 'setengah-matang': 2}
> ```

---

## ❓ Troubleshooting

| Masalah | Solusi |
|---|---|
| `Model belum dimuat` | Pastikan `banana.h5` ada di folder `model/` |
| `pip install` error | Coba `pip install tensorflow==2.15.0` jika versi 2.17 tidak tersedia |
| Port 5000 sudah dipakai | Ubah `port=5000` menjadi `port=5001` di `app.py` |
| Halaman tidak muncul | Pastikan sudah jalankan `python app.py` terlebih dahulu |
| Error GPU warning | Normal — model akan tetap berjalan di CPU |

---

## 🔄 Cara Stop Server
Tekan `Ctrl + C` di Command Prompt.
