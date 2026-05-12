# app.py  — Streamlit must be configured before ANY other st.* call

import re, string
import numpy as np
import streamlit as st
import joblib

# ---------- MUST be the first Streamlit command ----------
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬", layout="centered")

# ---------- Loading artifacts (after set_page_config is fine) ----------
@st.cache_resource
def load_artifacts():
    vec = joblib.load("tfidf_vectorizer.joblib")
    clf = joblib.load("model_tfidf_logreg.joblib")
    return vec, clf

tfidf_vectorizer, model = load_artifacts()

# ---------- Same-style cleaner as training ----------
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
STOP_WORDS = set("""
a about above after again against all am an and any are as at be because been before being below
between both but by can did do does doing down during each few for from further had has have having he
her here hers herself him himself his how i if in into is it its itself just me more most my myself no
nor not of off on once only or other our ours ourselves out over own same she should so some such than
that the their theirs them themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with you your yours yourself yourselves
""".split())

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    s = text.lower()
    s = re.sub(r"<br\s*/?>", " ", s)            # HTML breaks
    s = re.sub(r"<.*?>", " ", s)                # HTML tags
    s = re.sub(r"http\S+|www\.\S+", " ", s)     # URLs
    s = s.translate(PUNCT_TABLE)                # punctuation
    tokens = re.findall(r"[a-z]+", s)           # regex tokenization
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]
    return " ".join(tokens)

# ---------- UI ----------
st.title("🎬 IMDB Review Sentiment (TF-IDF + Logistic Regression)")
st.write("Enter a movie review to get a Positive/Negative prediction.")

txt = st.text_area("Review text", height=160, placeholder="e.g., The movie was surprisingly good...")
if st.button("Predict"):
    cleaned = clean_text(txt)
    if not cleaned.strip():
        st.warning("Please enter some text.")
    else:
        X = tfidf_vectorizer.transform([cleaned])
        proba_pos = float(model.predict_proba(X)[0][1])
        label = "Positive" if proba_pos >= 0.5 else "Negative"

        st.subheader(f"Prediction: *{label}*")
        st.write(f"Confidence — Positive: *{proba_pos:.3f}, Negative: **{1 - proba_pos:.3f}*")
        with st.expander("Show cleaned text the model sees"):
            st.code(cleaned)