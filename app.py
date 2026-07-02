"""
SpiceLens — app.py (Streamlit, single-file)
Klasifikasi Biji Rempah — MobileNetV2 (.keras)

CATATAN KONVERSI DARI FLASK -> STREAMLIT
-----------------------------------------
- CSS asli (style.css) disuntikkan APA ADANYA lewat st.markdown, jadi warna,
  font (Sora/Inter), kartu, badge, bar probabilitas, dsb TIDAK diubah.
- Tab "Upload" 1:1 secara visual dengan versi Flask/JS asli (drag-drop asli
  diganti st.file_uploader bawaan Streamlit — Streamlit tidak mendukung
  drag-drop kustom seperti JS asli, tapi kartu hasil & style tetap sama).
- Tab "Ambil Foto" terpisah sudah dihapus — kamera live di tab "Real-time"
  sudah mencakup fungsi jepret foto, jadi tidak perlu tab duplikat.
- Tab "Real-time" PAKAI st.camera_input BAWAAN STREAMLIT (bukan streamlit-webrtc
  lagi). Alasannya: st.camera_input membuka kamera browser dan langsung
  menampilkan pratinjau video yang live di sisi browser (tanpa perlu klik
  "start"/"connect" seperti WebRTC, dan tanpa risiko gagal negosiasi koneksi
  WebRTC di jaringan yang rewel) — jauh lebih stabil.
  Garis scan hijau yang bergerak naik-turun (efek "memindai") SEKARANG MURNI
  CSS: sebuah ::after pseudo-element yang ditempel lewat selector
  `div[data-testid="stCameraInput"]` dan dianimasikan dengan @keyframes
  (translate posisi `top` naik-turun). Ini tidak butuh OpenCV/av/threading
  sama sekali — tidak ada frame yang digambar manual, cukup CSS di atas
  pratinjau kamera bawaan Streamlit. Label "SCANNING" di pojok kiri-atas juga
  digambar lewat ::before (CSS), meniru gaya `.cam-overlay-badge` asli.
  Karena st.camera_input hanya mengirim gambar ke Python SETELAH tombol
  jepret ditekan (bukan streaming frame terus-menerus seperti WebRTC), tidak
  ada lagi logic "lock streak / kunci N-frame-berturut-turut" ataupun
  pengaturan ambang kepercayaan — hasil ditampilkan langsung sesaat setelah
  satu jepretan diproses, menampilkan kelas teratas apa adanya. Semua ini
  tetap dirender pakai CSS/kelas SpiceLens asli (live-result-card, status
  pill, dst).
- Semua output HTML dirender lewat helper md()/render_into_slot() yang menghapus
  indentasi tiap baris sebelum dikirim ke st.markdown — mencegah Streamlit salah
  mengira blok HTML sebagai code block (markdown menganggap baris berindentasi
  >=4 spasi sebagai kode, makanya <div> sempat muncul sebagai teks mentah).
- Semua logic model (load_model, preprocess, predict) identik dengan app.py Flask lama.
- Fitur "Tips Penggunaan" sudah dihapus sepenuhnya (CSS, data, dan rendering).

Instalasi tambahan yang dibutuhkan:
    pip install streamlit tensorflow pillow numpy

Menjalankan:
    streamlit run app.py
"""

from __future__ import annotations

import io
import base64
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── TensorFlow ──────────────────────────────────
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Config ──────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "mobilenetv2_bijirempah.keras"
IMG_SIZE   = 224

# Urutan HARUS sama dengan flow_from_directory (alphabetical)
CLASS_LABELS = ["jintan", "kapulaga", "kemiri", "ketumbar", "pala"]

# Metadata kelas (nama tampilan + nama latin). Tidak ada lagi field "tips".
CLASSES = {
    "jintan":   {"name": "Jintan",   "latin": "Cuminum cyminum"},
    "kapulaga": {"name": "Kapulaga", "latin": "Elettaria cardamomum"},
    "kemiri":   {"name": "Kemiri",   "latin": "Aleurites moluccanus"},
    "ketumbar": {"name": "Ketumbar", "latin": "Coriandrum sativum"},
    "pala":     {"name": "Pala",     "latin": "Myristica fragrans"},
}


def md(html: str) -> None:
    """Wrapper st.markdown yang menghapus indentasi di awal tiap baris,
    supaya Streamlit tidak salah mengira blok HTML sebagai code block
    (markdown menganggap baris berindentasi >=4 spasi sebagai kode)."""
    cleaned = "\n".join(line.lstrip() for line in html.split("\n"))
    st.markdown(cleaned, unsafe_allow_html=True)

st.set_page_config(
    page_title="SpiceLens — Klasifikasi Biji Rempah",
    page_icon="🌿",
    layout="centered",
)

# ── Load model sekali (cache) ──────────────────
@st.cache_resource(show_spinner=False)
def load_spice_model():
    if not TF_AVAILABLE or not MODEL_PATH.exists():
        return None, None
    try:
        m = load_model(str(MODEL_PATH))
        out_units = m.output_shape[-1]
        if out_units != len(CLASS_LABELS):
            return None, (
                f"Output model ({out_units}) != jumlah kelas ({len(CLASS_LABELS)}). "
                "Periksa CLASS_LABELS di app.py."
            )
        return m, None
    except Exception as e:
        return None, str(e)

model, model_error = load_spice_model()
model_loaded = model is not None

# ── Helpers ─────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict_image(pil_img: Image.Image) -> list[dict]:
    arr = preprocess_image(pil_img)
    probs = model.predict(arr, verbose=0)[0]
    pairs = sorted(zip(CLASS_LABELS, probs.tolist()), key=lambda x: x[1], reverse=True)
    out = []
    for cls, prob in pairs:
        meta = CLASSES.get(cls, {"name": cls, "latin": ""})
        out.append({
            "class": cls,
            "label": meta["name"],
            "latin": meta["latin"],
            "probability": round(float(prob), 6),
        })
    return out


def img_to_data_url(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n/1024:.1f} KB"
    return f"{n/(1024*1024):.1f} MB"


def render_result_card(preds: list[dict], img_data_url: str, section_title: str = "Top-5 Probabilitas Kelas") -> str:
    top = preds[0]
    pct = round(top["probability"] * 100, 1)

    prob_rows = ""
    for i, p in enumerate(preds):
        pp = round(p["probability"] * 100, 1)
        prob_rows += f'''
        <div class="prob-row">
          <span class="prob-rank">{i + 1}</span>
          <span class="prob-name">{p["label"]}</span>
          <div class="prob-bar-outer">
            <div class="prob-bar-inner{" top" if i == 0 else ""}" style="width:{pp}%"></div>
          </div>
          <span class="prob-pct">{pp}%</span>
        </div>'''

    return f'''
    <div class="result-card-main visible">
      <div class="result-hero">
        <img class="result-hero-img" src="{img_data_url}" alt="Gambar rempah" />
        <div class="result-hero-info">
          <div class="result-kelas">{top["label"]}</div>
          <div class="result-badge-row"><span class="badge badge-spice">{top["latin"]}</span></div>
          <div class="conf-label">Kepercayaan Model</div>
          <div class="conf-row">
            <div class="conf-bar-outer"><div class="conf-bar-inner" style="width:{pct}%"></div></div>
            <div class="conf-pct">{pct}%</div>
          </div>
        </div>
      </div>
      <div class="prob-section">
        <div class="prob-section-title">{section_title}</div>
        <div class="prob-grid">{prob_rows}</div>
      </div>
    </div>'''


def render_error_card(msg: str) -> str:
    return f'''
    <div class="error-card visible">
      <div class="error-icon">⚠️</div>
      <p>{msg}</p>
    </div>'''


# ══════════════════════════════════════════════════
# CSS ASLI (style.css) — disuntikkan APA ADANYA
# (blok CSS untuk .tips-card / .tips-card-title / .tips-list dihapus)
# ══════════════════════════════════════════════════
ORIGINAL_CSS = r"""
/* ═══════════════════════════════════════════════
   SpiceLens — style.css
   Font: Sora (display) + Inter (body)
   Tema: hijau herbal · dark mode native
   ═══════════════════════════════════════════════ */

/* ── Reset & base ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --green-50 : #f0faf4;
  --green-100: #d5f0e0;
  --green-200: #a8dfbf;
  --green-400: #4caf78;
  --green-600: #2d7a4f;
  --green-800: #1a4f32;
  --green-900: #0f3020;

  --amber-50 : #fff8ec;
  --amber-200: #f5c97a;
  --amber-400: #e09020;
  --amber-700: #8a5500;

  --surface-page  : #f6f5f2;
  --surface-card  : #ffffff;
  --surface-input : #f0efe9;

  --text-primary  : #1a1a18;
  --text-secondary: #4a4a45;
  --text-muted    : #8a8a82;

  --border        : rgba(0,0,0,.10);
  --border-strong : rgba(0,0,0,.18);

  --radius-sm: 6px;
  --radius   : 10px;
  --radius-lg: 16px;

  --ff-display: 'Sora', system-ui, sans-serif;
  --ff-body   : 'Inter', system-ui, sans-serif;

  --shadow-card: 0 1px 3px rgba(0,0,0,.06), 0 4px 12px rgba(0,0,0,.06);
}

@media (prefers-color-scheme: dark) {
  :root {
    --surface-page : #141412;
    --surface-card : #1e1e1b;
    --surface-input: #252520;

    --text-primary  : #f0efe8;
    --text-secondary: #b0b0a8;
    --text-muted    : #6a6a62;

    --border        : rgba(255,255,255,.09);
    --border-strong : rgba(255,255,255,.16);

    --green-50 : #0d2419;
    --green-100: #163625;
    --green-200: #1f4f34;
    --green-800: #a8dfbf;
    --green-900: #d5f0e0;

    --amber-50 : #2a1e08;
    --amber-200: #7a4f10;
    --amber-700: #f5c97a;

    --shadow-card: 0 1px 4px rgba(0,0,0,.4);
  }
}

html { scroll-behavior: smooth; }

body {
  font-family: var(--ff-body);
  font-size: 15px;
  line-height: 1.6;
  color: var(--text-primary);
  background: var(--surface-page);
  min-height: 100vh;
  padding: 0 0 4rem;
}

/* ── Layout container ──────────────────────────── */
.app {
  max-width: 680px;
  margin: 0 auto;
  padding: 2rem 1.25rem;
}

/* ── HEADER ────────────────────────────────────── */
.header {
  text-align: center;
  margin-bottom: 2.5rem;
}

.logo-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin-bottom: 6px;
}

.logo-icon-wrap {
  font-size: 32px;
  line-height: 1;
  filter: saturate(1.3);
}

.logo-name {
  font-family: var(--ff-display);
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.5px;
}
.logo-name span { color: var(--green-600); }

.header-desc {
  font-size: 13px;
  color: var(--text-muted);
  letter-spacing: 0.03em;
}

.model-warning {
  margin-top: 12px;
  padding: 10px 14px;
  background: var(--amber-50);
  border: 1px solid var(--amber-200);
  border-radius: var(--radius);
  font-size: 13px;
  color: var(--amber-700);
  text-align: left;
}
.model-warning strong { font-weight: 600; }
.model-warning code {
  font-family: monospace;
  background: rgba(0,0,0,.06);
  padding: 1px 5px;
  border-radius: 4px;
}

/* ── Step label ────────────────────────────────── */
.step-label {
  font-family: var(--ff-display);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--green-600);
  margin-bottom: 10px;
}

/* ── Upload area ───────────────────────────────── */
.upload-area {
  border: 2px dashed var(--border-strong);
  border-radius: var(--radius-lg);
  background: var(--surface-card);
  padding: 2.5rem 1.5rem;
  text-align: center;
  cursor: pointer;
  transition: border-color .2s, background .2s;
}
.upload-area:hover,
.upload-area.drag-over {
  border-color: var(--green-400);
  background: var(--green-50);
}

.upload-big-icon {
  font-size: 48px;
  line-height: 1;
  margin-bottom: 14px;
}

.upload-area h3 {
  font-family: var(--ff-display);
  font-size: 17px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 6px;
}

.upload-area p {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 18px;
}

.upload-fmt {
  font-size: 11px !important;
  color: var(--text-muted) !important;
  margin-top: 12px !important;
  margin-bottom: 0 !important;
}

/* ── Buttons ───────────────────────────────────── */
.btn-primary {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 9px 20px;
  font-family: var(--ff-body);
  font-size: 14px;
  font-weight: 500;
  color: #fff;
  background: var(--green-600);
  border: none;
  border-radius: var(--radius);
  cursor: pointer;
  transition: background .15s, transform .1s;
}
.btn-primary:hover  { background: var(--green-800); }
.btn-primary:active { transform: scale(.97); }
.btn-primary i { font-size: 16px; }

.btn-ghost {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  font-family: var(--ff-body);
  font-size: 14px;
  font-weight: 400;
  color: var(--text-secondary);
  background: transparent;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius);
  cursor: pointer;
  transition: background .15s, color .15s;
}
.btn-ghost:hover {
  background: var(--surface-input);
  color: var(--text-primary);
}

/* ── Preview card ──────────────────────────────── */
.preview-card {
  display: none;
  background: var(--surface-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.25rem;
  box-shadow: var(--shadow-card);
  margin-bottom: 1.5rem;
}
.preview-card.visible { display: block; }

.preview-inner {
  display: flex;
  gap: 1rem;
  align-items: flex-start;
}

.preview-img-wrap {
  width: 110px;
  height: 110px;
  flex-shrink: 0;
  border-radius: var(--radius);
  overflow: hidden;
  background: var(--surface-input);
  border: 1px solid var(--border);
}
.preview-img-wrap img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.preview-info { flex: 1; min-width: 0; }
.preview-info h4 {
  font-family: var(--ff-display);
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 3px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.preview-info p {
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 14px;
}

.preview-actions { display: flex; gap: 8px; flex-wrap: wrap; }

/* ── Result wrap ───────────────────────────────── */
.result-wrap { display: none; }
.result-wrap.visible { display: block; }

/* ── Loading card ──────────────────────────────── */
.loading-card {
  display: none;
  background: var(--surface-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 2.5rem;
  text-align: center;
  box-shadow: var(--shadow-card);
}
.loading-card.visible { display: block; }
.loading-card p {
  font-size: 14px;
  color: var(--text-secondary);
  margin-top: 14px;
}

.spinner-ring {
  width: 44px;
  height: 44px;
  border: 3px solid var(--green-100);
  border-top-color: var(--green-600);
  border-radius: 50%;
  animation: spin .8s linear infinite;
  margin: 0 auto;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Result main card ──────────────────────────── */
.result-card-main {
  display: none;
  background: var(--surface-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: var(--shadow-card);
}
.result-card-main.visible { display: block; }

/* ─ Hero ─ */
.result-hero {
  display: flex;
  gap: 1.25rem;
  padding: 1.5rem;
  border-bottom: 1px solid var(--border);
  background: var(--green-50);
}

.result-hero-img {
  width: 130px;
  height: 130px;
  object-fit: cover;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  flex-shrink: 0;
  background: var(--surface-input);
}

.result-hero-info { flex: 1; min-width: 0; }

.result-kelas {
  font-family: var(--ff-display);
  font-size: 22px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 8px;
  line-height: 1.2;
}

.result-badge-row { margin-bottom: 12px; }

.badge {
  display: inline-block;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  padding: 3px 10px;
  border-radius: 20px;
}
.badge-spice {
  background: var(--green-100);
  color: var(--green-800);
  border: 1px solid var(--green-200);
}

.conf-label {
  font-size: 11px;
  font-weight: 500;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 5px;
}

.conf-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.conf-bar-outer {
  flex: 1;
  height: 8px;
  background: var(--green-100);
  border-radius: 4px;
  overflow: hidden;
}
.conf-bar-inner {
  height: 100%;
  border-radius: 4px;
  background: var(--green-600);
  width: 0;
  transition: width .7s cubic-bezier(.4,0,.2,1);
}

.conf-pct {
  font-family: var(--ff-display);
  font-size: 16px;
  font-weight: 700;
  color: var(--green-600);
  min-width: 48px;
  text-align: right;
}

/* ─ Prob section ─ */
.prob-section { padding: 1.25rem 1.5rem; }

.prob-section-title {
  font-family: var(--ff-display);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 14px;
}

.prob-grid { display: flex; flex-direction: column; gap: 9px; }

.prob-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.prob-rank {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted);
  width: 18px;
  text-align: center;
  flex-shrink: 0;
}
.prob-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
  width: 90px;
  flex-shrink: 0;
}
.prob-bar-outer {
  flex: 1;
  height: 6px;
  background: var(--surface-input);
  border-radius: 3px;
  overflow: hidden;
}
.prob-bar-inner {
  height: 100%;
  border-radius: 3px;
  background: var(--green-400);
  transition: width .7s cubic-bezier(.4,0,.2,1);
}
.prob-bar-inner.top { background: var(--green-600); }
.prob-pct {
  font-size: 12px;
  color: var(--text-secondary);
  width: 42px;
  text-align: right;
  flex-shrink: 0;
}

/* ─ Bottom action ─ */
.bottom-action {
  padding: 1.25rem 1.5rem;
  border-top: 1px solid var(--border);
}

/* ── Error card ────────────────────────────────── */
.error-card {
  display: none;
  background: var(--surface-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 2.5rem;
  text-align: center;
  box-shadow: var(--shadow-card);
}
.error-card.visible { display: block; }
.error-icon { font-size: 36px; margin-bottom: 12px; }
.error-card p {
  font-size: 14px;
  color: var(--text-secondary);
  margin-bottom: 18px;
}

/* ── Camera tab styles ─────────────────────────── */
.cam-section {
  background: var(--surface-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: var(--shadow-card);
  margin-bottom: 1rem;
}

.cam-video-wrap {
  position: relative;
  aspect-ratio: 4/3;
  background: #0a0a08;
  overflow: hidden;
}
.cam-video-wrap video,
.cam-video-wrap canvas {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.cam-overlay-badge {
  position: absolute;
  top: 10px;
  left: 10px;
  background: rgba(0,0,0,.55);
  color: #fff;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 3px 9px;
  border-radius: 20px;
}

.cam-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  color: #666;
  gap: 10px;
}
.cam-placeholder i { font-size: 44px; }
.cam-placeholder p  { font-size: 13px; }

/* ── Camera controls bar ───────────────────────── */
.cam-controls {
  padding: 14px 1.25rem;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 10px;
  border-top: 1px solid var(--border);
  background: var(--surface-card);
}

.cam-ctrl-left {
  display: flex;
  justify-content: flex-start;
}

.cam-ctrl-center {
  display: flex;
  justify-content: center;
}

.cam-ctrl-right {
  display: flex;
  justify-content: flex-end;
}

.btn-cam-main {
  width: 62px;
  height: 62px;
  border-radius: 50%;
  border: none;
  background: var(--green-600);
  color: #fff;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  cursor: pointer;
  box-shadow: 0 4px 14px rgba(45,122,79,.35);
  transition: background .15s, transform .1s, box-shadow .15s;
  flex-shrink: 0;
}
.btn-cam-main:hover  {
  background: var(--green-800);
  box-shadow: 0 6px 18px rgba(45,122,79,.45);
}
.btn-cam-main:active { transform: scale(.93); }
.btn-cam-main i      { font-size: 24px; line-height: 1; }
.btn-cam-main span   {
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  line-height: 1;
}

.btn-cam-sec {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  border: 1.5px solid var(--border-strong);
  background: var(--surface-input);
  color: var(--text-secondary);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  cursor: pointer;
  transition: background .15s, color .15s, transform .1s;
  flex-shrink: 0;
}
.btn-cam-sec:hover  { background: var(--surface-card); color: var(--text-primary); }
.btn-cam-sec:active { transform: scale(.93); }
.btn-cam-sec i      { font-size: 18px; line-height: 1; }
.btn-cam-sec span   {
  font-size: 8px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  line-height: 1;
}

.btn-cam-sec.danger {
  border-color: #e5534b33;
  background: #fff1f0;
  color: #c0392b;
}
.btn-cam-sec.danger:hover { background: #ffe4e2; }

@media (prefers-color-scheme: dark) {
  .btn-cam-sec.danger { background: #2a1010; color: #e57373; border-color: #e5534b44; }
  .btn-cam-sec.danger:hover { background: #361414; }
}

.cam-status {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 3px;
}
.cam-status-label {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
  line-height: 1;
}
.cam-status-sub {
  font-size: 10px;
  color: var(--text-muted);
  line-height: 1;
}

.dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--text-muted);
  flex-shrink: 0;
  display: inline-block;
}
.dot.live {
  background: var(--green-400);
  animation: dotPulse 1.2s ease-in-out infinite;
}
@keyframes dotPulse { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ─ Real-time live result ─ */
.live-result-card {
  background: var(--surface-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1rem 1.25rem;
  box-shadow: var(--shadow-card);
}
.live-top {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 6px;
}
.live-name {
  font-family: var(--ff-display);
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
}
.live-conf {
  font-family: var(--ff-display);
  font-size: 16px;
  font-weight: 700;
  color: var(--green-600);
}
.live-bar-outer {
  height: 5px;
  background: var(--surface-input);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 14px;
}
.live-bar-inner {
  height: 100%;
  border-radius: 3px;
  background: var(--green-600);
}

/* ── Tab navigation ────────────────────────────── */
.tab-nav {
  display: flex;
  gap: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 1.25rem;
  background: var(--surface-input);
}
.tab-btn {
  flex: 1;
  padding: 9px 4px;
  font-family: var(--ff-body);
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
  background: transparent;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  transition: background .15s, color .15s;
  border-right: 1px solid var(--border);
}
.tab-btn:last-child { border-right: none; }
.tab-btn.active {
  background: var(--surface-card);
  color: var(--text-primary);
}
.tab-btn i { font-size: 15px; }

.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* ── Responsive ────────────────────────────────── */
@media (max-width: 480px) {
  .result-hero { flex-direction: column; }
  .result-hero-img { width: 100%; height: 200px; }
  .preview-inner { flex-direction: column; }
  .preview-img-wrap { width: 100%; height: 180px; }
  .prob-name { width: 75px; }
}

/* ════════════════════════════════════════════════
   LIVE — pengaturan ambang deteksi
   ════════════════════════════════════════════════ */
.live-settings {
  background: var(--surface-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1rem 1.25rem;
  margin-bottom: 1rem;
  box-shadow: var(--shadow-card);
}

.live-settings-label {
  font-family: var(--ff-display);
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.07em;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.live-settings-label i { color: var(--green-600); font-size: 14px; }

.live-settings-hint {
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.5;
  margin-bottom: 10px;
}
.live-settings-hint strong { color: var(--green-600); }

/* ════════════════════════════════════════════════
   STATUS PILL — di dalam result card
   ════════════════════════════════════════════════ */
.live-status-row {
  margin-bottom: 10px;
}

.live-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  padding: 4px 10px;
  border-radius: 20px;
  transition: background .2s, color .2s;
}
.live-status-pill i { font-size: 13px; }

.live-status-pill.scanning {
  background: var(--surface-input);
  color: var(--text-muted);
}
.live-status-pill.locking {
  background: var(--amber-50);
  color: var(--amber-700);
  border: 1px solid var(--amber-200);
}
.live-status-pill.locked {
  background: var(--green-100);
  color: var(--green-800);
  border: 1px solid var(--green-200);
}
.live-status-pill.rejected {
  background: #fff1f0;
  color: #c0392b;
  border: 1px solid #e5534b33;
}
@media (prefers-color-scheme: dark) {
  .live-status-pill.locking  { background: var(--amber-50); }
  .live-status-pill.rejected { background: #2a1010; color: #e57373; border-color: #e5534b44; }
}

/* ════════════════════════════════════════════════
   LOCK DOTS & INFO ROW
   ════════════════════════════════════════════════ */
.lock-info-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 8px 0 4px;
}

.lock-dots {
  display: flex;
  gap: 5px;
}

.lock-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--border-strong);
  transition: background .2s, transform .15s;
}
.lock-dot.filled {
  background: var(--green-600);
  transform: scale(1.15);
}

.lock-info-text {
  font-size: 11px;
  color: var(--text-muted);
  flex: 1;
}

"""

# CSS tambahan: sembunyikan chrome Streamlit + remap widget bawaan
# supaya menyatu dengan tema asli (tidak mengubah warna/kartu di atas)
EXTRA_CSS = """
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 680px;}
div[data-testid="stFileUploader"] section {
  border: 2px dashed var(--border-strong);
  border-radius: var(--radius-lg);
  background: var(--surface-card);
}
div[data-testid="stFileUploader"] {margin-bottom: 1.25rem;}
.stTabs [data-baseweb="tab-list"] {
  gap: 0; border: 1px solid var(--border); border-radius: var(--radius);
  overflow: hidden; background: var(--surface-input);
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--ff-body); font-size: 13px; font-weight: 500;
  color: var(--text-secondary); border-right: 1px solid var(--border);
}
.stTabs [aria-selected="true"] {
  background: var(--surface-card); color: var(--text-primary);
}
.stCameraInput button, .stButton button {
  font-family: var(--ff-body); font-weight: 500;
  background: var(--green-600); color: #fff; border-radius: var(--radius);
  border: none;
}
.stCameraInput button:hover, .stButton button:hover {background: var(--green-800);}

/* ════════════════════════════════════════════════
   GARIS SCAN — murni CSS di atas pratinjau kamera
   bawaan Streamlit (st.camera_input), gantinya efek
   canvas/OpenCV dari versi WebRTC sebelumnya.
   ════════════════════════════════════════════════ */
div[data-testid="stCameraInput"] {
  position: relative;
  border-radius: var(--radius-lg);
  overflow: hidden;
}

/* label "SCANNING" pojok kiri-atas, meniru .cam-overlay-badge asli */
div[data-testid="stCameraInput"]::before {
  content: "● SCANNING";
  position: absolute;
  top: 10px;
  left: 10px;
  background: rgba(0,0,0,.55);
  color: #fff;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  padding: 3px 9px;
  border-radius: 20px;
  z-index: 6;
  pointer-events: none;
}

/* garis hijau yang bergerak naik-turun di atas pratinjau video */
div[data-testid="stCameraInput"]::after {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  top: 8%;
  height: 3px;
  background: linear-gradient(90deg, transparent, rgba(76,175,120,.95), transparent);
  box-shadow: 0 0 14px 2px rgba(76,175,120,.55);
  z-index: 6;
  pointer-events: none;
  animation: spicelensScanLine 2.4s ease-in-out infinite;
}

@keyframes spicelensScanLine {
  0%   { top: 8%; }
  50%  { top: 68%; }
  100% { top: 8%; }
}
"""

md(f"<style>{ORIGINAL_CSS}{EXTRA_CSS}</style>")

# ══════════════════════════════════════════════════
# HEADER (identik dengan index.html asli)
# ══════════════════════════════════════════════════
warning_html = ""
if not model_loaded:
    reason = f" ({model_error})" if model_error else ""
    warning_html = f'''
    <div class="model-warning">
      ⚠️ Model belum dimuat{reason} — letakkan <strong>mobilenetv2_bijirempah.keras</strong>
      di folder <code>model/</code>
    </div>'''

md(f'''
<div class="app">
  <div class="header">
    <div class="logo-row">
      <div class="logo-icon-wrap">🌿</div>
      <div class="logo-name">Spice<span>Lens</span></div>
    </div>
    <div class="header-desc">Identifikasi Biji Rempah — MobileNetV2 CNN</div>
    {warning_html}
  </div>
</div>
''')

tab_upload, tab_live = st.tabs(["📤  Upload", "🎥  Real-time"])

# ══════════════════════════════════════════════════
# TAB 1 — UPLOAD  (step 01 / 02 seperti index.html)
# ══════════════════════════════════════════════════
with tab_upload:
    md('<div class="app"><div class="step-label">01 — Upload Gambar</div></div>')

    uploaded = st.file_uploader(
        "Seret & Lepas Gambar Biji Rempah (JPG · PNG · WEBP · Maks 16MB)",
        type=["jpg", "jpeg", "png", "webp"],
        key="upload_file",
    )

    if uploaded is not None:
        pil_img = Image.open(uploaded)
        data_url = img_to_data_url(pil_img)

        md(f'''
        <div class="app">
        <div class="preview-card visible">
          <div class="preview-inner">
            <div class="preview-img-wrap"><img src="{data_url}" /></div>
            <div class="preview-info">
              <h4>{uploaded.name}</h4>
              <p>{format_bytes(uploaded.size)}</p>
            </div>
          </div>
        </div>
        </div>''')

        md('<div class="app"><div class="step-label" style="margin-top:8px">02 — Hasil Identifikasi</div></div>')

        if not model_loaded:
            md(f'<div class="app">{render_error_card("Model belum dimuat. Letakkan mobilenetv2_bijirempah.keras di folder model/.")}</div>')
        else:
            with st.spinner("Mengidentifikasi jenis biji rempah…"):
                preds = predict_image(pil_img)
            md(f'<div class="app">{render_result_card(preds, data_url)}</div>')

# ══════════════════════════════════════════════════
# TAB 2 — REAL-TIME
# Sekarang pakai st.camera_input bawaan Streamlit: kamera browser
# terbuka langsung dengan pratinjau live (tidak perlu tombol
# "start"/"connect" seperti WebRTC), jauh lebih stabil di jaringan
# yang tidak ramah ke koneksi WebRTC. Efek "garis scan" naik-turun
# sepenuhnya CSS (::after + @keyframes, lihat EXTRA_CSS di atas),
# ditempel di atas widget kamera lewat selector
# `div[data-testid="stCameraInput"]` — tidak ada canvas/OpenCV yang
# digambar manual per-frame lagi.
# Karena st.camera_input mengirim gambar ke Python hanya SETELAH
# tombol jepret ditekan (bukan streaming frame terus-menerus), tidak
# ada lagi logic "kunci N-frame-berturut-turut" — begitu satu foto
# terkirim, model langsung memprediksi dan hasil kelas teratas
# langsung ditampilkan (tanpa pengaturan ambang kepercayaan).
# ══════════════════════════════════════════════════
with tab_live:
    md('<div class="app"><div class="step-label">02 — Deteksi Real-time (Kamera Langsung)</div></div>')

    if not model_loaded:
        md(f'<div class="app">{render_error_card("Model belum dimuat.")}</div>')

    live_photo = st.camera_input(
        "Kamera live — jepret untuk memindai", key="live_cam", label_visibility="collapsed",
    )

    # Garis scan hanya bergerak naik-turun selagi belum ada foto yang
    # diambil (pratinjau kamera masih live). Begitu foto sudah dijepret,
    # animasinya dihentikan (dan garis/badge disembunyikan) karena yang
    # tampil di widget sudah gambar diam, bukan video live lagi.
    if live_photo is not None:
        md('''
        <style>
        div[data-testid="stCameraInput"]::after { animation: none !important; opacity: 0; }
        div[data-testid="stCameraInput"]::before { content: "● TERTANGKAP"; opacity: .85; }
        </style>''')

    result_slot = st.empty()

    def render_into_slot(html: str) -> None:
        cleaned = "\n".join(line.lstrip() for line in html.split("\n"))
        result_slot.markdown(cleaned, unsafe_allow_html=True)

    if live_photo is None:
        render_into_slot('''
        <div class="app">
        <div class="live-result-card">
          <div class="live-status-row">
            <span class="live-status-pill scanning"><i class="ti ti-radar"></i> Menunggu jepretan…</span>
          </div>
          <p style="font-size:13px;color:var(--text-muted)">Arahkan kamera ke biji rempah, lalu klik tombol jepret untuk memindai.</p>
        </div>
        </div>''')
    elif not model_loaded:
        render_into_slot(f'<div class="app">{render_error_card("Model belum dimuat.")}</div>')
    else:
        pil_img = Image.open(live_photo)
        with st.spinner("Menganalisis…"):
            preds = predict_image(pil_img)

        top = preds[0]
        pct = round(top["probability"] * 100, 1)
        status, status_text, icon = "locked", f"Teridentifikasi: {top['label']}", "ti-lock"

        bars = "".join(f'''
        <div class="prob-row">
          <span class="prob-rank">{i+1}</span>
          <span class="prob-name">{p["label"]}</span>
          <div class="prob-bar-outer"><div class="prob-bar-inner{" top" if i==0 else ""}" style="width:{round(p["probability"]*100,1)}%"></div></div>
          <span class="prob-pct">{round(p["probability"]*100,1)}%</span>
        </div>''' for i, p in enumerate(preds))

        render_into_slot(f'''
        <div class="app">
        <div class="cam-status" style="margin:0 0 10px">
          <span class="dot live"></span>
          <span class="cam-status-label">Hasil jepretan terbaru</span>
        </div>
        <div class="live-result-card">
          <div class="live-status-row">
            <span class="live-status-pill {status}"><i class="ti {icon}"></i> {status_text}</span>
          </div>
          <div class="live-top">
            <span class="live-name">{top["label"]}</span>
            <span class="live-conf">{pct}%</span>
          </div>
          <div class="live-bar-outer"><div class="live-bar-inner" style="width:{pct}%"></div></div>
          <div class="prob-section-title" style="margin:12px 0 8px">Semua kelas (5)</div>
          <div class="prob-grid">{bars}</div>
        </div>
        </div>''')
