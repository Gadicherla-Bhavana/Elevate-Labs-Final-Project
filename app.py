
from pathlib import Path
import re
import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fake_news_model.joblib"

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")
st.write(
    "Enter a news headline and article below to get a "
    "machine-learning prediction."
)

st.warning(
    "Educational project only. Predictions may be wrong "
    "and are not proof that a news story is true or false."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Please run "
        "'py train_model.py' first."
    )
    st.stop()

model = load_model()

headline = st.text_input(
    "News headline",
    placeholder="Enter the headline (optional)"
)

article = st.text_area(
    "News article",
    placeholder="Paste the full news article here...",
    height=250
)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

if st.button("Analyze News", type="primary", use_container_width=True):
    text = clean_text(headline + " " + article)

    if len(text) < 30:
        st.info("Please enter a little more news text.")
    else:
        prediction = int(model.predict([text])[0])
        probabilities = model.predict_proba([text])[0]
        confidence = probabilities[prediction] * 100

        if prediction == 0:
            st.error("Prediction: LIKELY FAKE")
        else:
            st.success("Prediction: LIKELY REAL")

        st.metric("Model confidence", f"{confidence:.1f}%")
        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee of accuracy."
        )

st.divider()
st.caption("Built with Python, Streamlit, TF-IDF and Logistic Regression.")

