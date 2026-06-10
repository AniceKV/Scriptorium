import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import gdown

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scriptorium",
    page_icon="✝",
    layout="centered",
)

# ── Medieval CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root{
    --paper:#faf6ed;
    --paper-dark:#f2ead8;
    --text:#2b2b2b;
    --muted:#6b7280;
    --gold:#c59d5f;
    --gold-soft:#d8b982;
    --border:#e8dcc5;
    --accent:#7b4f2c;
}

html, body, [data-testid="stAppViewContainer"]{
    background:#f4f1ea !important;
}

[data-testid="stMain"]{
    background:transparent !important;
}

#MainMenu, header, footer{
    visibility:hidden;
}

[data-testid="stToolbar"]{
    display:none;
}

.book-wrapper{
    background:var(--paper);
    border:1px solid var(--border);
    border-radius:20px;
    top:16px; left:16px; right:16px; bottom:16px;
    max-width:1000px;
    margin:2rem auto;
    padding:3rem;
    box-shadow: 0 10px 25px rgba(0,0,0,.06), 0 2px 8px rgba(0,0,0,.03);
    position:relative;
}

.book-wrapper::before{
    content:"";
    position:absolute;
    top:16px; left:16px; right:16px; bottom:16px;
    border:1px solid rgba(197,157,95,.25);
    border-radius:14px;
    pointer-events:none;
}

.corner{ display:none; }

.title-block{ text-align:center; }

.title-main{
    display:block;
    font-family:'Cormorant Garamond', serif;
    font-size:4rem;
    font-weight:700;
    color:var(--accent);
    letter-spacing:-1px;
    line-height:1;
}

.title-sub{
    display:block;
    margin-top:.5rem;
    font-family:'Inter', sans-serif;
    font-size:.9rem;
    color:var(--muted);
    letter-spacing:.12em;
    text-transform:uppercase;
}

.divider{
    text-align:center;
    margin:1.8rem 0;
    color:var(--gold);
    font-size:1rem;
}

.section-label{
    display:block;
    margin-bottom:.5rem;
    font-family:'Inter', sans-serif;
    font-size:.8rem;
    font-weight:600;
    color:var(--accent);
    text-transform:uppercase;
    letter-spacing:.1em;
}

.stTextArea textarea{
    background:white !important;
    border:1px solid var(--border) !important;
    border-radius:12px !important;
    color:var(--text) !important;
    font-family:'Cormorant Garamond', serif !important;
    font-size:1.15rem !important;
    padding:1rem !important;
    box-shadow:none !important;
}

.stTextArea textarea:focus{
    border-color:var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(197,157,95,.15) !important;
}

[data-testid="stToggle"] label {
    color: #000000 !important;
}

.stSlider [data-baseweb="slider"] div[role="slider"]{
    background:var(--accent) !important;
}

.stSlider [data-baseweb="track"] div:first-child{
    background:var(--gold-soft) !important;
}

.stButton button{
    width:100% !important;
    border:none !important;
    border-radius:12px !important;
    background:var(--accent) !important;
    color:white !important;
    font-family:'Inter', sans-serif !important;
    font-weight:600 !important;
    padding:.8rem 1rem !important;
    transition:.2s ease !important;
}

.stButton button:hover{
    background:#5f3d23 !important;
}

.output-scroll{
    background:white;
    border:1px solid var(--border);
    border-radius:14px;
    padding:1.5rem;
    max-height:500px;
    overflow-y:auto;
    color:var(--text);
    font-family:'Cormorant Garamond', serif;
    font-size:1.2rem;
    line-height:1.9;
    white-space:pre-wrap;
}

.drop-cap::first-letter{
    font-family:'Cormorant Garamond', serif;
    float:left;
    font-size:3.5rem;
    line-height:.8;
    margin-right:.12em;
    color:var(--accent);
    font-weight:700;
}
[data-testid="stWidgetLabel"] [data-testid="stMarkdownContainer"] p {
    color: #000000 !important;
}

.footer-verse{
    margin-top:2rem;
    text-align:center;
    font-family:'Cormorant Garamond', serif;
    font-size:.95rem;
    color:var(--muted);
}
[data-testid="stToggle"] p {
    color: #000000 !important;
}

label[data-testid="stWidgetLabel"] p{
    font-family:'Inter', sans-serif !important;
    color:var(--muted) !important;
    font-size:.8rem !important;
}

.stSpinner > div {
    color: #000000 !important;
    font-weight: 500;
}

.stSpinner svg {
    stroke: #000000 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Hyperparameters (must match training) ──────────────────────────────────
block_size = 128
n_embd     = 512
n_head     = 8
n_layer    = 6
dropout    = 0.0
device     = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Model definition ───────────────────────────────────────────────────────
class MaskedSelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.cache_k = None
        self.cache_v = None
        self.use_kv_cache = False

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if self.use_kv_cache:
            if self.cache_k is not None:
                k = torch.cat([self.cache_k, k], dim=1)
                v = torch.cat([self.cache_v, v], dim=1)
            self.cache_k = k
            self.cache_v = v

        q_len = q.shape[1]
        k_len = k.shape[1]

        if q_len == k_len:
            out = F.scaled_dot_product_attention(q, k, v,
                  attn_mask=None,
                  dropout_p=dropout if self.training else 0.0,
                  is_causal=True)
        else:
            # decode step: Q is 1 token, K/V are full sequence
            out = F.scaled_dot_product_attention(q, k, v,
                  attn_mask=None,
                  dropout_p=0.0,
                  is_causal=False)
        return out

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([MaskedSelfAttention(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _all_heads(self):
        for block in self.blocks:
            for head in block.sa.heads:
                yield head

    def enable_kv_cache(self):
        for head in self._all_heads():
            head.use_kv_cache = True
            head.clear_cache()

    def disable_kv_cache(self):
        for head in self._all_heads():
            head.use_kv_cache = False
            head.clear_cache()

    def forward(self, idx, targets=None, pos=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        if pos is not None:
            pos_emb = self.position_embedding_table(pos)
        else:
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, use_kv_cache=False):
        if use_kv_cache:
            self.enable_kv_cache()
            prompt_len = idx.shape[1]
            pos = torch.arange(prompt_len, device=device)
            _, _ = self(idx, pos=pos)
            current_pos = prompt_len

        for _ in range(max_new_tokens):
            if use_kv_cache:
                idx_cond = idx[:, -1:]
                pos = torch.tensor([current_pos % block_size], device=device)
                current_pos += 1
            else:
                idx_cond = idx[:, -block_size:]
                pos = None

            logits, _ = self(idx_cond, pos=pos)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        if use_kv_cache:
            self.disable_kv_cache()

        return idx


# ── Model loading ──────────────────────────────────────────────────────────
MODEL_PATH = "model/bible_best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        file_id = "1wwI9Nmhgo0UE9LWQgvYUZOaAtNomZSIV"
        with st.spinner("Downloading model..."):
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                MODEL_PATH,
                quiet=False
            )
        st.success("Model downloaded successfully!")

    ckpt      = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    stoi      = ckpt["stoi"]
    itos      = ckpt["itos"]
    vocab_size = ckpt["vocab_size"]

    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join([itos[i] for i in l])

    model = GPTLanguageModel(vocab_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, encode, decode


# ── UI ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <span class="title-main">Scriptorium</span>
    <span class="title-sub">✦ &nbsp; A Transformer Trained from Holy Bible &nbsp; ✦</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider">— — —</div>', unsafe_allow_html=True)

st.markdown('<span class="section-label">✦ Write thy opening verse</span>', unsafe_allow_html=True)
prompt = st.text_area("", placeholder="And God said...", height=80, label_visibility="collapsed")

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature", 0.5, 1.5, 0.8, 0.05)
with col2:
    max_tokens = st.slider("Length", 10, 200, 100, 10)

use_kv_cache = st.toggle("KV Cache", value=True)

st.markdown("<br>", unsafe_allow_html=True)
generate_btn = st.button("✦ Speak the Word ✦")

st.markdown('<div class="divider">— — —</div>', unsafe_allow_html=True)

if generate_btn:
    if not prompt.strip():
        st.warning("Enter a prompt to begin.")
    else:
        with st.spinner("The scribe writes..."):
            try:
                model, encode, decode = load_model()
                encoded = encode(prompt)
                if not encoded:
                    st.error("No recognizable characters in prompt.")
                else:
                    context = torch.tensor([encoded], dtype=torch.long, device=device)
                    with torch.no_grad():
                        output_ids = model.generate(
                            context,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            use_kv_cache=use_kv_cache
                        )
                    output = decode(output_ids[0].tolist())

                    st.markdown('<span class="section-label">✦ The Scripture</span>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="output-scroll drop-cap">{output}</div>',
                        unsafe_allow_html=True
                    )
            except FileNotFoundError:
                st.error("Model file not found. Place 'bible_best.pt' in the model/ directory.")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("""
<div class="footer-verse">
    "In the beginning was the attention" — Vaswani 1:1<br>
    <span style="font-size:0.7rem; letter-spacing:0.15em; font-family:'Cinzel',serif; text-transform:uppercase;">
        19M parameters · trained from scratch · KJV Bible
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
