import streamlit as st
import base64
import json
import requests
from PIL import Image
import io
import fitz
import re
from datetime import datetime

st.set_page_config(
    page_title="Home Health PDx Tool",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    font-size: 28px;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 5px;
}
.sub-header {
    font-size: 14px;
    color: #666;
    margin-bottom: 20px;
}
.result-card {
    background: #f8f9fa;
    border-left: 4px solid #0066cc;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.pdx-code {
    font-size: 32px;
    font-weight: 700;
    color: #0066cc;
}
.warning-card {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.query-card {
    background: #f8d7da;
    border-left: 4px solid #dc3545;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.secondary-card {
    background: #d4edda;
    border-left: 4px solid #28a745;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

NVIDIA_API_KEY = "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"

def extract_text_from_pdf(pdf_file):
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def extract_text_from_image(image_file):
    try:
        image_bytes = image_file.read()
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        payload = {
            "model": "moonshotai/kimi-k2.5",
            "messages": [
                {
                    "role": "user",
                    "content": f"Extract all text from this medical document image. Return only the extracted text, nothing else.\n\nImage: data:image/jpeg;base64,{b64_image}"
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1,
            "stream": False
        }
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error extracting image text: {str(e)}"

def analyze_clinical_notes(clinical_text):
    prompt = f"""You are an expert Home Health ICD-10-CM Coding Specialist for OASIS and 485 coding.

Analyze this clinical documentation completely and return ONLY a valid JSON object.

Extract and analyze everything automatically including patient details, dates, physician names, and all coding information.

Return this exact JSON structure:

{{
  "patient_name": "full name from document or Not found",
  "patient_dob": "date of birth from document or Not found",
  "patient_age": "age from document or Not found",
  "patient_gender": "gender from document or Not found",
  "admission_date": "hospital admission date or Not found",
  "discharge_date": "hospital discharge date or Not found",
  "face_to_face_date": "face to face encounter date or Not found",
  "attending_physician": "attending physician full name and credentials or Not found",
  "referring_physician": "referring physician full name if different or Not found",
  "qualifying_event": "exact qualifying event with date",
  "homebound_status": "documented evidence patient is homebound with quotes from notes",
  "change_in_condition": "specific symptoms and signs that represent change in condition like increased SOB weight gain new wound fall increased pain",
  "pdx_code": "primary ICD-10-CM code with highest specificity",
  "pdx_description": "full official description of PDx code",
  "pdx_rationale": "detailed explanation of why this is correct PDx per home health guidelines",
  "pdx_alternative": "alternative code if documentation is ambiguous or None",
  "secondary_codes": [
    {{"code": "ICD-10-CM code", "description": "full description", "rationale": "why this code is included"}}
  ],
  "queries_needed": [
    "specific physician query needed"
  ],
  "wound_care": {{
    "present": "Yes or No",
    "details": "wound location size depth stage if documented",
    "skilled_need": "specific skilled nursing need for wound"
  }},
  "lab_draw": {{
    "present": "Yes or No",
    "details": "specific labs ordered and frequency"
  }},
  "skilled_need": {{
    "service": "SN or PT or OT or ST or combination",
    "rationale": "specific skilled need justification"
  }},
  "medications_high_risk": "list any high risk medications requiring skilled monitoring",
  "coding_warnings": [
    "specific compliance warning"
  ],
  "pdgm_considerations": "relevant PDGM comorbidity adjustment codes and clinical grouping notes"
}}

CRITICAL ICD-10-CM HOME HEALTH CODING RULES:
- PDx must be definitive diagnosis never a symptom R-code never Z-code never history-of code
- Never code possible suspected rule-out diagnoses
- Never code resolved conditions
- Do not code symptoms integral to PDx like edema with HF or dyspnea with COPD
- Use highest specificity with all required characters
- HFpEF preserved EF = I50.3x diastolic. Systolic reduced EF = I50.2x. Combined = I50.4x
- 5th character: 1=acute 2=chronic 3=acute on chronic
- Use I11.0 only if provider explicitly documents HTN caused HF
- Troponin elevation do not code if provider says demand mismatch from CHF
- Paroxysmal AFib = I48.0 not I48.11 or I48.19
- Pulmonary hypertension group 2 = I27.22
- Tricuspid regurgitation = I07.1
- Morbid obesity = E66.01 plus Z68.4x for BMI
- Sleep apnea do not code if only suspected until confirmed by sleep study
- Splenic infarct do not code if documented as possible or suspected
- Change in condition must be symptoms signs not the diagnosis itself

Clinical documentation to analyze:
{clinical_text}

Return ONLY valid JSON. No markdown fences. No explanation. Just the JSON object starting with {{ and ending with }}"""

    payload = {
        "model": "moonshotai/kimi-k2.5",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 2500,
        "temperature": 0.1,
        "top_p": 0.9,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers=headers,
        json=payload
    )

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"]
    cleaned = re.sub(r'```json|```', '', raw_text).strip()
    return json.loads(cleaned)

st.markdown('<div class="main-header">🏥 Home Health PDx Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload discharge summary or clinical notes — everything fills automatically</div>', unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Document")
    
    upload_type = st.radio(
        "Document type",
        ["PDF", "Image (photo of document)", "Paste text directly"],
        horizontal=True
    )
    
    clinical_text = ""
    
    if upload_type == "PDF":
        uploaded_file = st.file_uploader(
            "Upload discharge summary PDF",
            type=["pdf"],
            help="Upload hospital discharge summary or clinical notes"
        )
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):
                clinical_text = extract_text_from_pdf(uploaded_file)
            st.success(f"PDF extracted successfully — {len(clinical_text)} characters")
            with st.expander("View extracted text"):
                st.text(clinical_text[:2000] + "..." if len(clinical_text) > 2000 else clinical_text)
    
    elif upload_type == "Image (photo of document)":
        uploaded_file = st.file_uploader(
            "Upload photo of clinical document",
            type=["jpg", "jpeg", "png"],
            help="Take a photo of the discharge summary and upload here"
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded document", use_column_width=True)
            uploaded_file.seek(0)
            with st.spinner("Extracting text from image..."):
                clinical_text = extract_text_from_image(uploaded_file)
            st.success("Image text extracted successfully")
    
    else:
        clinical_text = st.text_area(
            "Paste clinical notes here",
            height=300,
            placeholder="Paste discharge summary, progress notes, or referral information here..."
        )
    
    analyze_button = st.button(
        "Analyze and Generate PDx Codes",
        type="primary",
        use_container_width=True,
        disabled=not bool(clinical_text)
    )

with col2:
    st.subheader("Analysis Results")
    
    if analyze_button and clinical_text:
        with st.spinner("Analyzing clinical documentation... this takes 15-20 seconds..."):
            try:
                result = analyze_clinical_notes(clinical_text)
                
                st.markdown("### Patient Information")
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.markdown(f"**Patient:** {result.get('patient_name', 'Not found')}")
                    st.markdown(f"**DOB:** {result.get('patient_dob', 'Not found')}")
                    st.markdown(f"**Age/Gender:** {result.get('patient_age', '')} {result.get('patient_gender', '')}")
                with info_col2:
                    st.markdown(f"**Admission:** {result.get('admission_date', 'Not found')}")
                    st.markdown(f"**Discharge:** {result.get('discharge_date', 'Not found')}")
                    st.markdown(f"**F2F Date:** {result.get('face_to_face_date', 'Not found')}")
                
                st.markdown(f"**Attending Physician:** {result.get('attending_physician', 'Not found')}")
                st.markdown(f"**Referring Physician:** {result.get('referring_physician', 'Not found')}")
                
                st.markdown("---")
                
                st.markdown("### Primary Diagnosis (PDx)")
                st.markdown(f'<div class="result-card"><span class="pdx-code">{result.get("pdx_code", "")}</span><br><strong>{result.get("pdx_description", "")}</strong><br><br>{result.get("pdx_rationale", "")}</div>', unsafe_allow_html=True)
                
                if result.get("pdx_alternative") and result.get("pdx_alternative") != "None":
                    st.markdown(f'<div class="warning-card">⚠️ <strong>Alternative PDx:</strong> {result.get("pdx_alternative")}</div>', unsafe_allow_html=True)
                
                st.markdown("### Change in Condition")
                st.markdown(f'<div class="result-card">{result.get("change_in_condition", "Not documented")}</div>', unsafe_allow_html=True)
                
                st.markdown("### Secondary Diagnoses")
                secondary = result.get("secondary_codes", [])
                if secondary:
                    for code in secondary:
                        st.markdown(f'<div class="secondary-card"><strong>{code.get("code", "")}</strong> — {code.get("description", "")}<br><small>{code.get("rationale", "")}</small></div>', unsafe_allow_html=True)
                
                queries = result.get("queries_needed", [])
                if queries:
                    st.markdown("### Physician Queries Needed")
                    for query in queries:
                        st.markdown(f'<div class="query-card">❓ {query}</div>', unsafe_allow_html=True)
                
                wound = result.get("wound_care", {})
                if wound.get("present") == "Yes":
                    st.markdown("### Wound Care")
                    st.markdown(f'<div class="result-card">🩹 <strong>Wound Present:</strong> {wound.get("details", "")}<br><strong>Skilled Need:</strong> {wound.get("skilled_need", "")}</div>', unsafe_allow_html=True)
                
                lab = result.get("lab_draw", {})
                if lab.get("present") == "Yes":
                    st.markdown("### Lab Draw")
                    st.markdown(f'<div class="result-card">🔬 {lab.get("details", "")}</div>', unsafe_allow_html=True)
                
                skilled = result.get("skilled_need", {})
                st.markdown("### Skilled Need")
                st.markdown(f'<div class="result-card"><strong>{skilled.get("service", "")}</strong><br>{skilled.get("rationale", "")}</div>', unsafe_allow_html=True)
                
                warnings = result.get("coding_warnings", [])
                if warnings:
                    st.markdown("### Coding Warnings")
                    for warning in warnings:
                        st.markdown(f'<div class="warning-card">⚠️ {warning}</div>', unsafe_allow_html=True)
                
                st.markdown("### PDGM Considerations")
                st.markdown(f'<div class="result-card">{result.get("pdgm_considerations", "None noted")}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                result_json = json.dumps(result, indent=2)
                st.download_button(
                    label="Download Full Analysis as JSON",
                    data=result_json,
                    file_name=f"pdx_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("Please check your document and try again")
    
    elif not clinical_text:
        st.info("Upload a document or paste clinical notes on the left to begin analysis")
```

---

**STEP 4 — Create requirements file:**

1. Back in your GitHub repository
2. Click **Add file** → **Create new file**
3. Name it exactly: **requirements.txt**
4. Paste this:
```
streamlit
requests
Pillow
PyMuPDF
