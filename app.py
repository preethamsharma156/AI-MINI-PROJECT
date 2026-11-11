import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="AI Password Strength Analyzer",
    page_icon="🛡️",
    layout="centered"
)

# 1. Custom Character Tokenizer for AI Pattern Detection
def char_tokenizer(password):
    return [c for c in password]

# 2. Train and Cache the AI Model
@st.cache_resource
def train_model():
    # Synthetic dataset mimicking weak, medium, and strong patterns
    data = {
        'password': [
            # Weak patterns (common words, repeated digits, keyboard walks)
            '123456', 'password', 'admin', 'qwerty', 'welcome1', 'letmein', 
            'iloveyou', 'football', 'monkey', '111111', 'sunshine', 'princess',
            # Medium patterns (dictionary words with simple numbers or single symbol)
            'Summer2024!', 'Tiger_99', 'BlueSky#1', 'CoffeeBreak8', 'Seattle99!', 
            'Student2025@', 'Developer#1', 'Password_OK', 'July1995!', 'AustinTX1',
            # Strong patterns (unpredictable, high entropy, mixed cases, symbols, no words)
            'k#9P_mZ8$qL', 'vB7*xW!2pQ', '9$mXp!Lq#8', 'wP!20_zM@1', '8*Lz#Qp&v9',
            'mK9!v2&Xp_W', 'pQ#8!vR9$mX', 'Z9*zP@3q_9x', '7$vB!8pKzW#', 'T8*wP!20zM','xhPEhv2008@%fbr'
        ] * 80,
        'strength': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
        ] * 80
    }
    df = pd.DataFrame(data)

    # Character-level TF-IDF vectorizer (slices passwords into 2 to 4-character chunks)
    vectorizer = TfidfVectorizer(tokenizer=char_tokenizer, analyzer='char', ngram_range=(2, 4))
    X = vectorizer.fit_transform(df['password'])
    y = df['strength']

    # AI Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return vectorizer, model

# 3. Web Interface
st.title("🛡️ AI Password Strength Analyzer")
st.write("An intelligent password evaluator powered by **Character-Level Machine Learning**.")

with st.spinner("Initializing AI model..."):
    vectorizer, model = train_model()

# User input
password = st.text_input("Enter a password to test:", type="password", placeholder="Type here...")

if password:
    # Transform input and get prediction
    vec_input = vectorizer.transform([password])
    prediction = model.predict(vec_input)[0]
    probabilities = model.predict_proba(vec_input)[0]

    st.write("---")
    st.subheader("Analysis Result")

    # Display status based on prediction
    if prediction == 0:
        st.error("🚨 **Strength: WEAK**")
        st.write("The AI detected highly predictable structures or common leaked character combinations.")
    elif prediction == 1:
        st.warning("⚠️ **Strength: MEDIUM**")
        st.write("Reasonably secure, but still relies on dictionary patterns or common substitution styles.")
    else:
        st.success("✅ **Strength: STRONG**")
        st.write("Excellent! The AI found this password to be highly unpredictable and complex.")

    # Breakdown table
    st.write("### AI Confidence Breakdown")
    metrics_df = pd.DataFrame({
        "Category": ["Weak Pattern Likelihood", "Medium Pattern Likelihood", "Strong Pattern Likelihood"],
        "Confidence Score": [f"{prob * 100:.1f}%" for prob in probabilities]
    })
    st.table(metrics_df)

    # Heuristic checks
    st.write("### Quick Checks")
    col1, col2, col3 = st.columns(3)
    col1.metric("Length", f"{len(password)} chars", "Good" if len(password) >= 12 else "Short")
    col2.metric("Digits", "Present" if any(c.isdigit() for c in password) else "Missing")
    col3.metric("Symbols", "Present" if any(not c.isalnum() for c in password) else "Missing")
