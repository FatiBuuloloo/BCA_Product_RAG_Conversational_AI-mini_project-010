import os
import json
import re
import gradio as gr
from groq import Groq
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings as _HFEmb

PINECONE_API_KEY  = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY")
INDEX_NAME        = "chatbot-view-2"
MODEL_NAME        = "BAAI/bge-m3"
MAX_CONTEXT_CHARS = 6000

class BGEEmbeddings(_HFEmb):
    def embed_query(self, text: str):
        return super().embed_query("query: " + text)

lc_embeddings = BGEEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=lc_embeddings,
    pinecone_api_key=PINECONE_API_KEY
)
client = Groq(api_key=GROQ_API_KEY)

products = [
    # Simpanan Individu
    "Tahapan BCA","Tahapan Xpresi","Tahapan Berjangka","Tahapan Berjangka SiMuda",
    "Simpanan Pelajar","Tabungan Prestasi (Tapres)","TabunganKu",
    "Deposito Berjangka","BCA Dollar","e-Deposito",
    # Pinjaman Individu
    "Kredit Sepeda Motor (KSM)","Kredit Kendaraan Bermotor Pembelian",
    "Kredit Kendaraan Bermotor Refinancing","Kredit Pemakaian Rumah Pembelian",
    "Kredit Pemakaian Rumah (KPR) Pembelian","Kredit Pemakaian Rumah Sejahtera BCA",
    "Kredit Pemakaian Rumah Refinancing","Kredit Pemakaian Rumah Renovasi",
    "Pinjaman Kredit Tanpa Agunan Personal","BCA Secured Personal Loan",
    # Wealth — Asuransi
    "Asuransi Kebakaran","Asuransi Property All Risks",
    "Asuransi Electronic Equipment Insurance (EEI)",
    "Asuransi MyGuard","Asuransi MyGuard - Accident Care",
    "Asuransi MyGuard - Critical Care","Asuransi MyGuard - Health Care",
    "Asuransi MyGuard - Hospital Care","Asuransi BCA Life Proteksi Jiwa Optima",
    "Asuransi Maxi Infinite Link Assurance Plus (MILA Plus)",
    "Asuransi Proteksi Jiwa Maksima (JIMI)","Asuransi Credit Life",
    "Asuransi Credit Life Chubb Life","Asuransi Credit Life Paylater",
    "Asuransi Heritage Platinum Protection (Heritage+)",
    "Asuransi BCA Life Accident Safeguard","Asuransi Optima Accident Protection",
    "Asuransi Education Guard","Asuransi Household Guard","Asuransi Kendaraan Bermotor",
    "Asuransi Safety Guard Critical Cover (STAR)","Asuransi AIA Optima Protection Plus",
    "Asuransi BCA Life Perlindungan Kritis Optima","Asuransi Bima Proteksi Kesehatanku",
    "Asuransi Dental Care Plan","Asuransi Hospital 100% Refundable",
    "Asuransi Maxi Value Protection","Asuransi Optima Cancer Protection",
    "Asuransi Proteksi Kesehatan Ultima",
    "Asuransi Proteksi Penyakit Kritis Maksima Extra (Prima Extra)",
    "Asuransi Smart Eazicare Blue","Asuransi Smart Eazicare Platinum",
    "Asuransi Proteksi Edukasi Maksima","Asuransi Proteksi Retirement Maksima",
    "Asuransi Proteksi Wholelife Income Maksima","Travel Insurance",
    "Asuransi Prosper Life Guard (PROSPER)",
    # Wealth — Investasi
    "BCA Reksadana","Obligasi","Rekening Dana Nasabah","Rekening Dana Lender (RDL) BCA",
    # Kartu Kredit BCA
    "BCA Everyday Card","BCA Card Platinum","BCA Smartcash",
    "BCA Singapore Airlines KrisFlyer Visa Signature",
    "BCA Singapore Airlines KrisFlyer Visa Infinite",
    "BCA Singapore Airlines PPS Club Visa Infinite",
    "BCA Mastercard Black","BCA Blibli Mastercard","BCA tiket.com Mastercard",
    "BCA Mastercard Globe","BCA Mastercard World","BCA JCB Black",
    "BCA UnionPay","BCA American Express Platinum",
    # Uang Elektronik & Reward
    "Flazz","Sakuku","Reward BCA",
]

product_lists = {
    "Simpanan Individu": [
        "Tahapan BCA","Tahapan Xpresi","Tahapan Berjangka","Tahapan Berjangka SiMuda",
        "Simpanan Pelajar","Tabungan Prestasi (Tapres)","TabunganKu",
        "Deposito Berjangka","BCA Dollar","e-Deposito",
    ],
    "Pinjaman Individu": [
        "Kredit Sepeda Motor (KSM)","Kredit Kendaraan Bermotor Pembelian",
        "Kredit Kendaraan Bermotor Refinancing","Kredit Pemakaian Rumah Pembelian",
        "Kredit Pemakaian Rumah Sejahtera BCA","Kredit Pemakaian Rumah Refinancing",
        "Kredit Pemakaian Rumah Renovasi","Pinjaman Kredit Tanpa Agunan Personal",
        "BCA Secured Personal Loan",
    ],
    "Uang Elektronik": ["Flazz","Sakuku"],
    "Kartu Kredit BCA": [
        "BCA Everyday Card","BCA Card Platinum","BCA Smartcash",
        "BCA Singapore Airlines KrisFlyer Visa Signature",
        "BCA Singapore Airlines KrisFlyer Visa Infinite",
        "BCA Singapore Airlines PPS Club Visa Infinite",
        "BCA Mastercard Black","BCA Blibli Mastercard","BCA tiket.com Mastercard",
        "BCA Mastercard Globe","BCA Mastercard World","BCA JCB Black",
        "BCA UnionPay","BCA American Express Platinum",
    ],
    "Reward BCA": ["Reward BCA"],
}

wealth_lists = """**Wealth Management BCA** terdiri dari:

**Asuransi** (berdasarkan jenis):
- Harta Benda: Asuransi Kebakaran, Property All Risks, Electronic Equipment Insurance (EEI)
- Jiwa: MyGuard (Accident/Critical/Health/Hospital Care), BCA Life Proteksi Jiwa Optima, MILA Plus, JIMI, Credit Life, Credit Life Chubb Life, Credit Life Paylater, Heritage Platinum Protection
- Kecelakaan: BCA Life Accident Safeguard, Optima Accident Protection, Education Guard, Household Guard
- Kendaraan: Asuransi Kendaraan Bermotor
- Kesehatan: Safety Guard Critical Cover (STAR), AIA Optima Protection Plus, BCA Life Perlindungan Kritis Optima, Bima Proteksi Kesehatanku, Dental Care Plan, Hospital 100% Refundable, Maxi Value Protection, Optima Cancer Protection, Proteksi Kesehatan Ultima, Prima Extra, Smart Eazicare Blue, Smart Eazicare Platinum
- Pendidikan: Proteksi Edukasi Maksima
- Pensiun & Anuitas: Proteksi Retirement Maksima, Proteksi Wholelife Income Maksima
- Travel: Travel Insurance
- Warisan: Heritage Platinum Protection (Heritage+), Prosper Life Guard (PROSPER)

**Investasi**: BCA Reksadana, Obligasi

**Rekening Investasi**: Rekening Dana Nasabah (RDN), Rekening Dana Lender (RDL) BCA"""

def is_list_query(question: str):
    q = question.lower()
    trigger_keywords = [
        "list produk","daftar produk","sebutkan produk","apa saja produk",
        "semua produk","jenis produk","tampilkan produk","produk apa saja",
    ]
    specific_key_words = [
        "persyaratan","syarat","dokumen","biaya","bunga","suku bunga",
        "limit","plafon","manfaat","fitur","risiko","cara","berapa",
        "jelaskan","premi","denda","pengecualian","ketentuan",
    ]
    if not any(kw in q for kw in trigger_keywords): return False, None
    if any(kw in q for kw in specific_key_words):       return False, None
    if any(w in q for w in ["wealth","asuransi","investasi","reksadana","rdn","rdl"]):
        return True, "Wealth Management"
    if any(w in q for w in ["simpanan","tabungan","deposito","giro"]):
        return True, "Simpanan Individu"
    if any(w in q for w in ["pinjaman individu","kredit individu","jenis pinjaman"]):
        return True, "Pinjaman Individu"
    if "kartu kredit" in q: return True, "Kartu Kredit BCA"
    if "uang elektronik" in q: return True, "Uang Elektronik"
    if "reward" in q: return True, "Reward BCA"
    return False, None

def answer_list(category: str) -> str:
    if category == "Wealth Management":
        return "Berikut produk **Wealth Management BCA**:\n\n" + wealth_lists + \
               "\n\nInfo lebih lanjut: Halo BCA **1500 888**."
    items = "\n".join(f"- **{p}**" for p in product_lists.get(category, []))
    return f"Berikut seluruh produk **{category}**:\n\n{items}\n\nInfo lebih lanjut: Halo BCA **1500 888**."

def normalize(s: str) -> str:
    return re.sub(r'[\s\-–—()/.,]+', ' ', s.strip().lower())
product_prompt = "\n".join(f"- {p}" for p in products)
analyze_prompt = f"""Anda adalah sistem pendeteksi nama produk BCA dari pertanyaan user.

TUGAS:
1. product_name          : nama produk PERSIS dari daftar, atau null
2. needs_clarification   : true jika ambigu
3. clarification_question: kalimat tanya sopan Bahasa Indonesia
4. clarification_options : array pilihan

DAFTAR PRODUK:
{product_prompt}

PEMETAAN PENTING:
- "kpr pembelian" → "Kredit Pemakaian Rumah Pembelian"
- "kpr sejahtera"/"kpr subsidi" → "Kredit Pemakaian Rumah Sejahtera BCA"
- "kpr refinancing" → "Kredit Pemakaian Rumah Refinancing"
- "kpr renovasi" → "Kredit Pemakaian Rumah Renovasi"
- "pinjaman tanpa agunan"/"kta"/"personal loan" (bukan secured) → "Pinjaman Kredit Tanpa Agunan Personal"
- "secured personal loan" → "BCA Secured Personal Loan"
- "myguard critical"/"critical care" → "Asuransi MyGuard - Critical Care"
- "myguard accident" → "Asuransi MyGuard - Accident Care"
- "myguard health" → "Asuransi MyGuard - Health Care"
- "myguard hospital" → "Asuransi MyGuard - Hospital Care"
- "optima cancer" → "Asuransi Optima Cancer Protection"
- "aia optima"/"optima protection plus" → "Asuransi AIA Optima Protection Plus"
- "tapres" → "Tabungan Prestasi (Tapres)"
- "jimi" → "Asuransi Proteksi Jiwa Maksima (JIMI)"
- "mila plus" → "Asuransi Maxi Infinite Link Assurance Plus (MILA Plus)"
- "star"/"safety guard" → "Asuransi Safety Guard Critical Cover (STAR)"
- "hoki" → "Asuransi Proteksi Wholelife Income Maksima"
- "prosper" → "Asuransi Prosper Life Guard (PROSPER)"
- "pratama" → "Asuransi Proteksi Kesehatan Ultima"

needs_clarification=true HANYA jika:
- User tanya "KPR" tanpa jenis (pembelian/sejahtera/refinancing/renovasi)
- User tanya "pinjaman"/"kredit" tanpa produk spesifik
- User tanya "asuransi" tanpa nama produk spesifik
- User tanya DOKUMEN atau PERSYARATAN suatu produk TANPA menyebut jenis nasabah (karyawan/wiraswasta/profesional)
  → berikan pilihan: ["Karyawan", "Wiraswasta", "Profesional", "Semua jenis nasabah"]

PENTING: Kembalikan HANYA satu baris JSON valid. DILARANG menambahkan penjelasan atau markdown.
{{"product_name": "...", "needs_clarification": false, "clarification_question": "", "clarification_options": []}}"""

def analyze_intent(question: str) -> dict:
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": analyze_prompt},
            {"role": "user",   "content": f"Pertanyaan: {question}"}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=150,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except:
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    return {"product_name": None, "needs_clarification": False,
            "clarification_question": "", "clarification_options": []}

def retrieve(question: str, product_name) -> str:
    docs = vectorstore.similarity_search(question, k=20)
    if not docs: return ""
    if product_name:
        pn_norm = normalize(product_name)
        target  = [d for d in docs if normalize(d.metadata.get("product_name","")) == pn_norm]
        final   = target if target else docs
    else:
        final = docs
    parts, total, seen = [], 0, set()
    for i, doc in enumerate(final, 1):
        meta   = doc.metadata
        sec_id = meta.get("section_id", f"_noid_{i}")
        if sec_id in seen: continue
        seen.add(sec_id)
        body  = meta.get("text") or doc.page_content
        chunk = (
            f"[Chunk {i}] Produk: {meta.get('product_name','-')}"
            f" | Bagian: {meta.get('section_title','-')}"
            f" | Tag: {meta.get('topic_tag','-')}\n{body}"
        )
        if total + len(chunk) > MAX_CONTEXT_CHARS:
            sisa = MAX_CONTEXT_CHARS - total
            if sisa > 200: parts.append(chunk[:sisa] + "\n[...dipotong]")
            break
        parts.append(chunk)
        total += len(chunk)

    return "\n\n---\n\n".join(parts)

answer_prompt = """\
Anda adalah "View", asisten virtual resmi BCA.
Pengembang: Fati Buulolo | LinkedIn: https://www.linkedin.com/in/fati-buulolo-7a9236391/

KONTEKS — satu-satunya sumber kebenaran:
{context}

ATURAN MUTLAK:
1. Jawab HANYA dari teks yang ada di KONTEKS. DILARANG menambahkan dari pengetahuan umum.
2. TAMPILKAN SEMUA angka, poin, sub-poin, dan detail dari KONTEKS. JANGAN lewatkan apapun.
3. Jika info tidak ada di KONTEKS: "Maaf, informasi tidak tersedia. Hubungi Halo BCA **1500 888**."
4. Gunakan nama produk PERSIS seperti di KONTEKS.
5. Format: bullet points, **bold** untuk angka dan istilah penting, URL sebagai teks biasa.
6. Nilai uang: format Rupiah (Rp50.000).
7. Tutup dengan: Info lebih lanjut: Halo BCA **1500 888**.
"""

def generate_answer(question: str, context: str, history: list) -> str:
    recent   = history[-4:] if len(history) > 4 else history
    messages = [{"role": "system", "content": answer_prompt.format(context=context)}]
    messages.extend(recent)
    messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(
        messages=messages,
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=900,
        top_p=0.9,
    )
    return resp.choices[0].message.content

developer_key_words = [
    "pengembang","siapa yang membuat","siapa yang mengembangkan",
    "dibuat oleh","developer","who made","siapa kamu","siapa pembuat",
    "tentang view","about view","siapakah kamu","siapakah yang mengembangkan",
]

def chat_fn(message: str, chat_history: list, state: dict):
    if not message.strip():
        return chat_history, state, ""
    history = state.get("history", [])
    pending = state.get("pending", {})
    if any(w in message.lower() for w in developer_key_words):
        answer = (
            "Saya adalah **View**, asisten virtual untuk informasi produk BCA.\n\n"
            "Dikembangkan oleh **Fati Buulolo**, mahasiswa Jurusan Matematika "
            "dengan fokus Machine Learning dan Data.\n\n"
            "LinkedIn: https://www.linkedin.com/in/fati-buulolo-7a9236391/\n\n"
            "Info lebih lanjut tentang produk BCA: Halo BCA **1500 888**."
        )
        chat_history = chat_history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": answer},
        ]
        history = (history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": answer},
        ])[-8:]
        return chat_history, {**state, "history": history, "pending": {}}, ""
    if pending.get("waiting"):
        original_q   = pending["original_question"]
        product_name = pending.get("product_name")
        combined_q = f"{original_q} untuk nasabah {message}"
        context = retrieve(combined_q, product_name)
        if not context.strip():
            answer = "Maaf, informasi tersebut tidak tersedia. Hubungi Halo BCA **1500 888**."
        else:
            answer = generate_answer(combined_q, context, history)
        chat_history = chat_history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": answer},
        ]
        history = (history + [
            {"role": "user",      "content": combined_q},
            {"role": "assistant", "content": answer},
        ])[-8:]
        return chat_history, {**state, "history": history, "pending": {}}, ""
    is_list, list_cat = is_list_query(message)
    if is_list and list_cat:
        answer = answer_list(list_cat)
        chat_history = chat_history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": answer},
        ]
        history = (history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": answer},
        ])[-8:]
        return chat_history, {**state, "history": history, "pending": {}}, ""
    intent = analyze_intent(message)
    if intent.get("needs_clarification") and intent.get("clarification_question"):
        clarif_q   = intent["clarification_question"]
        clarif_opt = intent.get("clarification_options", [])
        clarif_txt = clarif_q
        if clarif_opt:
            opts      = "\n".join(f"  {i+1}. **{o}**" for i, o in enumerate(clarif_opt))
            clarif_txt = f"{clarif_q}\n\n{opts}"
        chat_history = chat_history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": clarif_txt},
        ]
        new_pending = {
            "waiting":           True,
            "original_question": message,
            "product_name":      intent.get("product_name"),
        }
        return chat_history, {**state, "history": history, "pending": new_pending}, ""
    context = retrieve(message, intent.get("product_name"))
    if not context.strip():
        answer = (
            "Maaf, informasi tersebut tidak tersedia dalam data saya. "
            "Silakan hubungi Halo BCA di **1500 888**."
        )
    else:
        answer = generate_answer(message, context, history)
    chat_history = chat_history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer},
    ]
    history = (history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer},
    ])[-8:]
    return chat_history, {**state, "history": history, "pending": {}}, ""

OPENING_MESSAGE = """Halo! Saya **View** 👋

Saya asisten virtual untuk informasi produk **BCA**. Berikut beberapa hal yang bisa Anda tanyakan:

- Apa saja produk Simpanan Individu BCA?
- Apa saja produk Pinjaman Individu BCA?
- Siapakah yang mengembangkan kamu?"""

EXAMPLES = [
    "Apa persyaratan mengajukan Pinjaman Kredit Tanpa Agunan Personal?",
    "Apa saja syarat dan ketentuan dalam pengambilan KPR BCA Pembelian untuk karyawan?",
    "Berapa suku bunga promo KPR BCA Pembelian?",
    "Berapa suku bunga dan biaya yang harus dibayar jika saya mengambil Pinjaman Kredit Tanpa Agunan Personal",
    "Siapakah yang mengembangkan View?",
]

INITIAL_CHAT = [{"role": "assistant", "content": OPENING_MESSAGE}]
INITIAL_STATE = {"history": [], "pending": {}}

CSS = """
:root {
    --blue:   #0060ad;
    --blue2:  #0060ad;
    --light:  #e8f0fb;
    --border: #d0dce8;
    --gray:   #f5f6fa;
}

/* Chatbot bubbles */
.message.svelte-1s78gfg { max-width: 85%; }

.user .message-bubble-border {
    background: linear-gradient(135deg, var(--blue) 0%, var(--blue2) 100%) !important;
    color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    border: none !important;
}

.bot .message-bubble-border {
    background: var(--light) !important;
    color: #1a1a2e !important;
    border-radius: 18px 18px 18px 4px !important;
    border: 1px solid var(--border) !important;
}
/* Avatar agar Full dalam Frame Lingkaran */
.avatar-container img, 
div[class*="avatar-container"] img,
.bot .avatar-container img,
.user .avatar-container img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important; /* Memastikan gambar memenuhi lingkaran tanpa gepeng */
    border-radius: 50% !important;
    border: 2px solid var(--border) !important;
    display: block !important;
}

/* Memastikan wadahnya sendiri berbentuk lingkaran sempurna */
.avatar-container, 
div[class*="avatar-container"] {
    width: 40px !important; /* Sesuaikan ukuran yang Anda inginkan */
    height: 40px !important;
    border-radius: 50% !important;
    overflow: hidden !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Input */
#msg-input textarea {
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    background: white !important;
    transition: border-color 0.2s;
}
#msg-input textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(0,63,136,0.1) !important;
}

/* Tombol kirim */
#send-btn {
    background: linear-gradient(135deg, var(--blue) 0%, var(--blue2) 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    min-height: 48px !important;
    transition: opacity 0.2s, transform 0.1s;
}
#send-btn:hover { opacity: 0.85 !important; transform: translateY(-1px); }
#send-btn:active { transform: translateY(0); }

/* Tombol clear */
#clear-btn {
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    background: white !important;
    font-size: 0.88rem !important;
}
#clear-btn:hover { background: var(--gray) !important; }

/* Example buttons */
.gr-examples .label { display: none; }
.gr-examples button {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--blue) !important;
    font-size: 0.83rem !important;
    padding: 6px 14px !important;
    transition: all 0.15s;
}
.gr-examples button:hover {
    background: var(--light) !important;
    border-color: var(--blue) !important;
}

/* Disclaimer */
.footer-txt {
    font-size: 0.73rem;
    color: #999;
    text-align: center;
    margin-top: 6px;
}
/* Menghilangkan seluruh area metadata (processing, durasi, dll) */
div[class*="meta-text"], 
.meta-text {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Menghilangkan efek 'generating' (garis biru berkedip) di atas bubble */
div[class*="generating"],
.generating {
    display: none !important;
}

/* Menghilangkan timer di pojok kanan bawah jika masih muncul */
.show-api-button, .built-with {
    display: none !important;
}
"""

def reset_chat():
    return INITIAL_CHAT, INITIAL_STATE, ""

with gr.Blocks(title="View — BCA Product Assistant") as demo:
    state = gr.State(INITIAL_STATE)
    gr.HTML("""
    <div style="
        background: linear-gradient(135deg, #003f88 0%, #0059c2 100%);
        color: white; padding: 18px 26px 16px; border-radius: 14px;
        margin-bottom: 14px; display: flex; align-items: center; gap: 14px;
        box-shadow: 0 4px 20px rgba(0,63,136,0.25);
    ">
        <div style="
            width: 64px; height: 64px; border-radius: 50%;
            background: white;
            display: flex; align-items: center; justify-content: center;
            flex-shrink: 0; overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            padding: 4px;
        ">
            <img src="https://huggingface.co/spaces/Viewww/View-Bot-BCA-Assistant/resolve/main/avatar/logo_bca.jpg"
                 alt="BCA"
                 style="width: 100%; height: 100%; border-radius: 50%; object-fit: contain;"
            />
        </div>
            <div>
                <div style="color: #FFFFFF; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 3px;">
                View — BCA Product Assistant
            </div>
            <div style="color: rgba(255, 255, 255, 0.9); font-size: 0.79rem; opacity: 0.82;">
                Asisten virtual informasi produk BCA &nbsp;•&nbsp; Dikembangkan oleh <strong style="color: #FFFFFF !important;">Fati Buulolo</strong>
            </div>
        </div>
    </div>
    """)
    chatbot = gr.Chatbot(
        value=INITIAL_CHAT,
        elem_id="chatbot",
        height=480,
        show_label=False,
        avatar_images=(
            "avatar/user_avatar.jpg",
            "avatar/chatbot_avatar.avif",
        ),
    )
    with gr.Row(equal_height=True):
        txt = gr.Textbox(
            placeholder="Tanyakan sesuatu tentang produk BCA...",
            show_label=False,
            scale=9,
            container=False,
            elem_id="msg-input",
            autofocus=True,
        )
        btn = gr.Button("Kirim ➤", scale=1, variant="primary", elem_id="send-btn")
    clear = gr.Button("🗑️  Mulai Percakapan Baru", variant="secondary", elem_id="clear-btn")
    gr.Examples(
        examples=EXAMPLES,
        inputs=txt,
        label="Contoh Pertanyaan",
        examples_per_page=5,
    )
    gr.HTML("""
    <p class="footer-txt">
        Informasi yang diberikan View bersumber dari data produk resmi BCA.<br>
        Untuk kepastian, selalu konfirmasi ke <strong>Halo BCA 1500 888</strong>.
    </p>
    """)
    btn.click(
        fn=chat_fn,
        inputs=[txt, chatbot, state],
        outputs=[chatbot, state, txt],
    )
    txt.submit(
        fn=chat_fn,
        inputs=[txt, chatbot, state],
        outputs=[chatbot, state, txt],
    )
    clear.click(
        fn=reset_chat,
        outputs=[chatbot, state, txt],
    )


if __name__ == "__main__":
    demo.launch(
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Plus Jakarta Sans"),
        ),
    )