import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="wide")

# ---------- Style ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, .stApp, button, textarea { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(99,102,241,.35) 0%, transparent 40%),
        radial-gradient(circle at 90% 25%, rgba(20,184,166,.25) 0%, transparent 35%),
        #0b1020;
}

/* Page plus large, moins d'espace en haut et en bas */
.block-container { max-width: 1000px; padding-top: 1.5rem; padding-bottom: 1rem; }

/* Hero compact */
.hero { text-align: center; margin-bottom: 1rem; }
.pill {
    display: inline-block; padding: .2rem .8rem; border-radius: 999px;
    font-size: .7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
    color: #a5b4fc; background: rgba(99,102,241,.15); border: 1px solid rgba(99,102,241,.35);
}
.hero h1 {
    font-size: 2.2rem; font-weight: 800; margin: .6rem 0 .3rem; line-height: 1.1;
    background: linear-gradient(90deg, #fff, #a5b4fc 60%, #5eead4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #94a3b8; font-size: .95rem; margin: 0; }

/* Zone de texte */
.stTextArea textarea {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 14px !important;
    color: #f1f5f9 !important;
    padding: .8rem !important;
    font-size: 1rem;
}
.stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,.25) !important;
}

/* Boutons */
.stButton > button {
    width: 100%; border-radius: 14px; font-weight: 600; transition: all .2s ease;
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
    color: #cbd5e1;
}
.stButton > button:hover { transform: translateY(-2px); border-color: #818cf8; color: #fff; }
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    border: none; color: white;
    background: linear-gradient(90deg, #6366f1, #14b8a6);
    box-shadow: 0 10px 30px -10px rgba(99,102,241,.8);
}
.stButton > button p { font-size: .9rem; }

/* Carte résultat compacte */
.result {
    margin-top: 1rem; padding: 1.1rem; border-radius: 18px; text-align: center;
    backdrop-filter: blur(14px); animation: pop .4s ease;
}
@keyframes pop { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
.result .icon { font-size: 1.8rem; }
.result h2 { margin: .2rem 0; color: #fff; font-weight: 800; font-size: 1.4rem; }
.result .sub { color: #cbd5e1; margin: 0 0 .8rem; font-size: .9rem; }
.spam { background: rgba(239,68,68,.12); border: 1px solid rgba(239,68,68,.5); box-shadow: 0 0 40px -10px rgba(239,68,68,.6); }
.ham  { background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.5); box-shadow: 0 0 40px -10px rgba(16,185,129,.6); }

.bar { height: 10px; border-radius: 999px; background: rgba(255,255,255,.1); overflow: hidden; }
.fill { height: 100%; border-radius: 999px; }
.spam .fill { background: linear-gradient(90deg, #f97316, #ef4444); }
.ham  .fill { background: linear-gradient(90deg, #14b8a6, #22c55e); }
.split { display: flex; justify-content: space-between; margin-top: .5rem;
         font-size: .8rem; color: #94a3b8; }

.foot { text-align: center; color: #64748b; font-size: .8rem; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Modèle ----------
def clean(texts):
    return texts.str.replace(r"\d{5,}", " longnumber ", regex=True)

@st.cache_resource
def train_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(clean(df["message"]))
    model = MultinomialNB(alpha=0.1)
    model.fit(X, df["label"])
    return vectorizer, model

vectorizer, model = train_model()

# ---------- Interface ----------
if "msg" not in st.session_state:
    st.session_state.msg = ""

def set_text(value):
    st.session_state.msg = value

st.markdown("""
<div class="hero">
    <span class="pill">Machine Learning · Naive Bayes</span>
    <h1>Spam Message Detector</h1>
    <p>Paste a message. The model tells you if it looks suspicious.</p>
</div>
""", unsafe_allow_html=True)

text = st.text_area("Message", key="msg", height=80,
                    placeholder="Congratulations! You won a free prize...",
                    label_visibility="collapsed")

# Exemples + bouton d'analyse sur une seule ligne
c1, c2, c3, c4 = st.columns(4)
c1.button("🎁 Prize", on_click=set_text,
          args=("Congratulations! Call 09012345678 to claim your free prize",))
c2.button("💬 Friend", on_click=set_text,
          args=("Are you coming to dinner tonight?",))
c3.button("🤔 Tricky", on_click=set_text, args=("The winner is here",))
analyze = c4.button("Analyze message", type="primary")

if analyze:
    if text.strip() == "":
        st.warning("Please write a message first.")
    else:
        vec = vectorizer.transform(clean(pd.Series([text])))
        probs = model.predict_proba(vec)[0]
        p_ham, p_spam = probs[0] * 100, probs[1] * 100   # ordre : ['ham', 'spam']
        is_spam = p_spam > p_ham
        css, icon, title = ("spam", "🚨", "Spam probable") if is_spam else ("ham", "✅", "Message normal")
        confidence = max(p_ham, p_spam)

        st.markdown(f"""
        <div class="result {css}">
            <div class="icon">{icon}</div>
            <h2>{title}</h2>
            <p class="sub">Model confidence: {confidence:.1f}%</p>
            <div class="bar"><div class="fill" style="width:{p_spam:.1f}%"></div></div>
            <div class="split"><span>Normal {p_ham:.1f}%</span><span>Spam {p_spam:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<p class="foot">Trained on the SMS Spam Collection · A confident model can still be wrong</p>',
            unsafe_allow_html=True)