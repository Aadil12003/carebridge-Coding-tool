import streamlit as st
import requests
import json
import re
import base64
import logging
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from datetime import datetime

# -------------------- CONFIG --------------------
st.set_page_config(page_title="CareBridge PDx Tool", layout="wide")

# Logging
logging.basicConfig(level=logging.INFO)

# Secure API key
NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-70b-instruct"

# -------------------- CACHE --------------------
@st.cache_data
def call_api(prompt):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.1
    }

    try:
        logging.info("Calling API...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            raise Exception(response.text)

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


# -------------------- OCR --------------------
def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        return f"PDF Error: {str(e)}"


def extract_text_from_image(file):
    try:
        image = Image.open(file)
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"Image OCR Error: {str(e)}"


# -------------------- ANALYSIS --------------------
def clean_json(raw):
    raw = re.sub(r"```json|```", "", raw).strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    return raw[start:end]


def analyze_text(text):
    # limit input size
    if len(text) > 15000:
        text = text[:15000]

    prompt = f"""
    Extract ICD-10 PDx and return ONLY JSON:
    {{
      "patient_name": "",
      "pdx_code": "",
      "pdx_description": "",
      "pdx_rationale": "",
      "confidence_score": ""
    }}

    TEXT:
    {text}
    """

    raw = call_api(prompt)

    if not raw:
        return None

    try:
        cleaned = clean_json(raw)
        return json.loads(cleaned)
    except:
        st.error("Invalid JSON from model. Try again.")
        return None


# -------------------- MEMORY (CORRECTIONS) --------------------
def load_corrections():
    try:
        with open("corrections.json") as f:
            return json.load(f)
    except:
        return []


def save_corrections(data):
    with open("corrections.json", "w") as f:
        json.dump(data, f)


if "corrections" not in st.session_state:
    st.session_state.corrections = load_corrections()


# -------------------- UI --------------------
st.title("CareBridge PDx Tool (Fixed Version)")

uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "jpg", "jpeg", "png"])
text_input = st.text_area("Or paste clinical notes")

text = ""

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_image(uploaded_file)

if text_input:
    text += "\n" + text_input


# -------------------- ANALYZE BUTTON --------------------
if st.button("Analyze"):
    if not text.strip():
        st.warning("No input provided")
        st.stop()

    with st.spinner("Processing..."):
        result = analyze_text(text)

        if result:
            st.success("Analysis Complete")

            st.write("### Patient")
            st.write(result.get("patient_name", "Not found"))

            st.write("### PDx")
            st.write(result.get("pdx_code", ""), "-", result.get("pdx_description", ""))

            st.write("### Rationale")
            st.write(result.get("pdx_rationale", ""))

            st.write("### Confidence")
            st.write(result.get("confidence_score", ""))

            # ---------------- CORRECTION SYSTEM ----------------
            st.markdown("---")
            st.write("### Correct if wrong")

            wrong = result.get("pdx_code", "")
            correct = st.text_input("Correct Code")

            if st.button("Save Correction"):
                if correct:
                    st.session_state.corrections.append({
                        "wrong": wrong,
                        "correct": correct,
                        "date": str(datetime.now())
                    })
                    save_corrections(st.session_state.corrections)
                    st.success("Saved")
                else:
                    st.warning("Enter correct code")
