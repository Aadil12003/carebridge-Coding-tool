import streamlit as st
import base64
import json
import requests
from PIL import Image
import re
from datetime import datetime
import fitz

st.set_page_config(page_title="Home Health PDx Tool", layout="wide")

st.markdown("""
<style>
.result-card {background: #f8f9fa; border-left: 4px solid #0066cc; padding: 15px; border-radius: 5px; margin: 10px 0;}
.pdx-code {font-size: 32px; font-weight: 700; color: #0066cc;}
.warning-card {background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; border-radius: 5px; margin: 10px 0;}
.query-card {background: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; border-radius: 5px; margin: 10px 0;}
.secondary-card {background: #d4edda; border-left: 4px solid #28a745; padding: 15px; border-radius: 5px; margin: 10px 0;}
.alert-card {background: #f8d7da; border-left: 4px solid #721c24; padding: 15px; border-radius: 5px; margin: 10px 0;}
.info-card {background: #d1ecf1; border-left: 4px solid #0c5460; padding: 15px; border-radius: 5px; margin: 10px 0;}
.section-header {font-size: 18px; font-weight: 600; color: #1a1a2e; margin-top: 20px; margin-bottom: 10px; border-bottom: 2px solid #0066cc; padding-bottom: 5px;}
</style>
""", unsafe_allow_html=True)

NVIDIA_API_KEY = "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"
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-70b-instruct"

def call_api(prompt):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 3000,
        "temperature": 0.1,
        "stream": False
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    data = response.json()
    if "choices" not in data:
        raise Exception(f"API error: {json.dumps(data)}")
    content = data["choices"][0]["message"]["content"]
    if not content:
        raise Exception("API returned empty response")
    return content

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
        prompt = f"You are a medical document reader. Extract all text from this medical document completely and accurately. Return only the extracted text with no additional commentary. The image is encoded in base64: {b64_image[:500]}"
        return call_api(prompt)
    except Exception as e:
        return f"Error extracting image: {str(e)}"

def analyze_clinical_notes(clinical_text):
    prompt = f"""You are an expert Home Health ICD-10-CM Coding Specialist for OASIS and 485 coding with deep knowledge of PDGM payment model and all CMS guidelines.

Analyze this clinical documentation completely and return ONLY a valid JSON object with no markdown and no explanation.

{{
  "patient_name": "full name from document or Not found",
  "patient_dob": "date of birth or Not found",
  "patient_age": "age or Not found",
  "patient_gender": "gender or Not found",
  "admission_date": "hospital admission date or Not found",
  "discharge_date": "hospital discharge date or Not found",
  "face_to_face_date": "face to face encounter date or Not found",
  "attending_physician": "attending physician full name and credentials or Not found",
  "referring_physician": "referring physician name or Not found",
  "qualifying_event": "exact qualifying event with date",
  "homebound_status": "specific evidence patient is homebound with quotes from notes",
  "change_in_condition": "specific symptoms and signs representing change in condition such as increased SOB weight gain new wound fall increased pain palpitations edema worsening",
  "pdx_code": "primary ICD-10-CM code with highest specificity",
  "pdx_description": "full official description of PDx code",
  "pdx_rationale": "detailed explanation of why this is correct PDx per home health guidelines",
  "pdx_alternative": "alternative code if ambiguous or None",
  "secondary_codes": [
    {{"code": "ICD-10-CM code", "description": "full description", "rationale": "why included and PDGM value"}}
  ],
  "queries_needed": [
    "specific physician query needed with exact wording"
  ],
  "wound_care": {{
    "present": "Yes or No",
    "wound_type": "pressure ulcer or venous ulcer or surgical wound or diabetic wound or other",
    "location": "exact anatomical location",
    "stage": "stage 1 2 3 4 or unstageable or not documented",
    "size": "dimensions if documented or not documented",
    "details": "full wound description",
    "skilled_need": "specific skilled nursing interventions needed",
    "oasis_item": "M1300 or M1302 or M1306 or M1307 or M1308 as applicable"
  }},
  "lab_draw": {{
    "present": "Yes or No",
    "details": "specific labs ordered and frequency",
    "high_risk_monitoring": "specific lab monitoring needed for high risk medications"
  }},
  "skilled_need": {{
    "service": "SN or PT or OT or ST or combination",
    "rationale": "specific skilled need justification per Medicare guidelines",
    "frequency_suggestion": "suggested visit frequency such as 3 times per week for 4 weeks"
  }},
  "medications": {{
    "high_risk": "list all high risk medications requiring skilled monitoring such as anticoagulants diuretics insulin",
    "all_medications": "complete medication list if available",
    "medication_teaching_needed": "Yes or No with details",
    "reconciliation_needed": "Yes or No"
  }},
  "oasis_alerts": {{
    "m1033_hospitalization_risk": "High or Medium or Low with reason",
    "m1240_pain_assessment": "pain present Yes or No with details and interference with activity",
    "m1800_grooming": "independent or needs assistance or dependent",
    "m1910_fall_risk": "Yes or No with details of fall history or risk factors",
    "mental_health_flags": "depression or anxiety or dementia or cognitive impairment if documented",
    "cardiac_rehab_indicated": "Yes or No with rationale",
    "diabetic_foot_care": "Yes or No with details",
    "pressure_ulcer_risk": "High or Medium or Low with Braden scale if documented"
  }},
  "therapy_needs": {{
    "pt_indicated": "Yes or No with specific functional deficits",
    "ot_indicated": "Yes or No with specific ADL deficits",
    "st_indicated": "Yes or No with specific speech or swallowing deficits",
    "therapy_goals": "specific measurable therapy goals"
  }},
  "coding_warnings": [
    "specific compliance warning"
  ],
  "pdgm_considerations": {{
    "clinical_group": "predicted PDGM clinical grouping",
    "comorbidity_adjustment": "relevant comorbidity codes for payment adjustment",
    "functional_impairment": "level based on documentation",
    "high_value_codes": "list high value secondary codes that trigger comorbidity adjustment"
  }},
  "documentation_gaps": [
    "specific missing documentation needed for compliance"
  ],
  "physician_query_letters": [
    {{
      "query_topic": "topic of query",
      "query_letter": "complete ready to send physician query letter text"
    }}
  ]
}}

CRITICAL ICD-10-CM HOME HEALTH CODING RULES:
- PDx must be definitive diagnosis never a symptom R-code never Z-code never history-of code
- Never code possible suspected or rule-out diagnoses
- Never code resolved conditions
- Do not code symptoms integral to PDx like edema with HF or dyspnea with COPD
- Use highest specificity with all required characters
- HFpEF preserved EF = I50.3x diastolic. Systolic reduced EF = I50.2x. Combined = I50.4x
- 5th character 1=acute 2=chronic 3=acute on chronic
- Use I11.0 only if provider explicitly documents HTN caused HF
- Troponin elevation do not code if provider says demand mismatch from CHF
- Paroxysmal AFib = I48.0 not I48.11 or I48.19
- Pulmonary hypertension group 2 = I27.22
- Tricuspid regurgitation = I07.1
- Morbid obesity = E66.01 plus Z68.4x for BMI
- Sleep apnea do not code if only suspected until confirmed by sleep study
- Splenic infarct do not code if documented as possible or suspected
- Change in condition must be symptoms signs not the diagnosis itself
- Venous stasis ulcer = I87.2 plus L97.x for wound
- Pressure ulcer use M89.x series with stage
- Diabetic foot ulcer = E11.621 plus L97.x
- Depression = F32.x or F33.x specify severity
- Dementia = F03.x or G30.x specify type
- Fall = W19.xxxa initial encounter
- Fracture use appropriate fracture code with 7th character
- COPD exacerbation = J44.1
- Pneumonia = J18.9 unspecified or specific organism if documented
- UTI = N39.0
- Sepsis = A41.9 or specific organism

OASIS GUIDELINES:
- M1033 risk factors include history of falls pressure ulcers multiple hospitalizations five or more medications
- M1240 pain must document frequency severity and interference with activity
- M1300 pressure ulcer must document stage location and dimensions
- M1800 ADL assessment must document level of assistance needed
- M1910 fall risk must be assessed if any fall history or risk factors present
- Homebound status requires documentation that leaving home requires considerable taxing effort or medical contraindication

Clinical documentation to analyze:
{clinical_text}

Return ONLY valid JSON starting with open brace and ending with close brace. No markdown fences. No explanation outside the JSON."""

    raw = call_api(prompt)
    cleaned = re.sub(r'```json|```', '', raw).strip()
    start = cleaned.find('{')
    end = cleaned.rfind('}') + 1
    if start != -1 and end > start:
        cleaned = cleaned[start:end]
    return json.loads(cleaned)

if "history" not in st.session_state:
    st.session_state.history = []

st.title("CareBridge Home Health PDx Tool")
st.caption("Upload discharge summary or clinical notes and everything fills automatically")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["New Analysis", "Case History", "Help"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Document")
        upload_type = st.radio("Document type", ["PDF", "Image", "Paste text"], horizontal=True)
        clinical_text = ""

        if upload_type == "PDF":
            uploaded_file = st.file_uploader("Upload discharge summary PDF", type=["pdf"])
            if uploaded_file:
                with st.spinner("Extracting text from PDF..."):
                    clinical_text = extract_text_from_pdf(uploaded_file)
                st.success(f"PDF extracted successfully — {len(clinical_text)} characters found")
                with st.expander("View extracted text"):
                    st.text(clinical_text[:3000])

        elif upload_type == "Image":
            uploaded_file = st.file_uploader("Upload photo of document", type=["jpg", "jpeg", "png"])
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
                height=350,
                placeholder="Paste discharge summary or progress notes here..."
            )

        analyze_button = st.button(
            "Analyze and Generate All Codes",
            type="primary",
            use_container_width=True,
            disabled=not bool(clinical_text)
        )

    with col2:
        st.subheader("Analysis Results")

        if analyze_button and clinical_text:
            with st.spinner("Analyzing clinical documentation — please wait 20 to 30 seconds..."):
                try:
                    result = analyze_clinical_notes(clinical_text)

                    st.markdown('<div class="section-header">Patient Information</div>', unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Patient:** {result.get('patient_name', 'Not found')}")
                        st.markdown(f"**DOB:** {result.get('patient_dob', 'Not found')}")
                        st.markdown(f"**Age/Gender:** {result.get('patient_age', '')} {result.get('patient_gender', '')}")
                    with c2:
                        st.markdown(f"**Admission:** {result.get('admission_date', 'Not found')}")
                        st.markdown(f"**Discharge:** {result.get('discharge_date', 'Not found')}")
                        st.markdown(f"**F2F Date:** {result.get('face_to_face_date', 'Not found')}")

                    st.markdown(f"**Attending Physician:** {result.get('attending_physician', 'Not found')}")
                    st.markdown(f"**Referring Physician:** {result.get('referring_physician', 'Not found')}")
                    st.markdown(f"**Qualifying Event:** {result.get('qualifying_event', 'Not found')}")
                    st.markdown(f"**Homebound Status:** {result.get('homebound_status', 'Not documented')}")

                    st.markdown("---")
                    st.markdown('<div class="section-header">Primary Diagnosis</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-card"><span class="pdx-code">{result.get("pdx_code", "")}</span><br><strong>{result.get("pdx_description", "")}</strong><br><br>{result.get("pdx_rationale", "")}</div>', unsafe_allow_html=True)

                    if result.get("pdx_alternative") and result.get("pdx_alternative") != "None":
                        st.markdown(f'<div class="warning-card">Alternative PDx: {result.get("pdx_alternative")}</div>', unsafe_allow_html=True)

                    st.markdown('<div class="section-header">Change in Condition</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="info-card">{result.get("change_in_condition", "Not documented")}</div>', unsafe_allow_html=True)

                    st.markdown('<div class="section-header">Secondary Diagnoses</div>', unsafe_allow_html=True)
                    for code in result.get("secondary_codes", []):
                        st.markdown(f'<div class="secondary-card"><strong>{code.get("code", "")}</strong> — {code.get("description", "")}<br><small>{code.get("rationale", "")}</small></div>', unsafe_allow_html=True)

                    pdgm = result.get("pdgm_considerations", {})
                    if pdgm:
                        st.markdown('<div class="section-header">PDGM Considerations</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="result-card"><strong>Clinical Group:</strong> {pdgm.get("clinical_group", "")}<br><strong>Comorbidity Adjustment:</strong> {pdgm.get("comorbidity_adjustment", "")}<br><strong>High Value Codes:</strong> {pdgm.get("high_value_codes", "")}<br><strong>Functional Impairment:</strong> {pdgm.get("functional_impairment", "")}</div>', unsafe_allow_html=True)

                    queries = result.get("queries_needed", [])
                    if queries:
                        st.markdown('<div class="section-header">Physician Queries Needed</div>', unsafe_allow_html=True)
                        for query in queries:
                            st.markdown(f'<div class="query-card">{query}</div>', unsafe_allow_html=True)

                    query_letters = result.get("physician_query_letters", [])
                    if query_letters:
                        st.markdown('<div class="section-header">Physician Query Letters</div>', unsafe_allow_html=True)
                        for letter in query_letters:
                            with st.expander(f"Query Letter: {letter.get('query_topic', '')}"):
                                st.text_area("Ready to send letter", letter.get("query_letter", ""), height=200)

                    wound = result.get("wound_care", {})
                    if wound.get("present") == "Yes":
                        st.markdown('<div class="section-header">Wound Care</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="result-card"><strong>Type:</strong> {wound.get("wound_type", "")}<br><strong>Location:</strong> {wound.get("location", "")}<br><strong>Stage:</strong> {wound.get("stage", "")}<br><strong>Size:</strong> {wound.get("size", "")}<br><strong>Details:</strong> {wound.get("details", "")}<br><strong>Skilled Need:</strong> {wound.get("skilled_need", "")}<br><strong>OASIS Item:</strong> {wound.get("oasis_item", "")}</div>', unsafe_allow_html=True)

                    lab = result.get("lab_draw", {})
                    if lab.get("present") == "Yes":
                        st.markdown('<div class="section-header">Lab Draw</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="result-card"><strong>Labs:</strong> {lab.get("details", "")}<br><strong>High Risk Monitoring:</strong> {lab.get("high_risk_monitoring", "")}</div>', unsafe_allow_html=True)

                    meds = result.get("medications", {})
                    st.markdown('<div class="section-header">Medications</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-card"><strong>High Risk:</strong> {meds.get("high_risk", "None noted")}<br><strong>All Medications:</strong> {meds.get("all_medications", "See discharge summary")}<br><strong>Teaching Needed:</strong> {meds.get("medication_teaching_needed", "")}<br><strong>Reconciliation Needed:</strong> {meds.get("reconciliation_needed", "")}</div>', unsafe_allow_html=True)

                    skilled = result.get("skilled_need", {})
                    st.markdown('<div class="section-header">Skilled Need</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-card"><strong>Service:</strong> {skilled.get("service", "")}<br><strong>Rationale:</strong> {skilled.get("rationale", "")}<br><strong>Suggested Frequency:</strong> {skilled.get("frequency_suggestion", "")}</div>', unsafe_allow_html=True)

                    therapy = result.get("therapy_needs", {})
                    st.markdown('<div class="section-header">Therapy Assessment</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-card"><strong>PT:</strong> {therapy.get("pt_indicated", "")}<br><strong>OT:</strong> {therapy.get("ot_indicated", "")}<br><strong>ST:</strong> {therapy.get("st_indicated", "")}<br><strong>Goals:</strong> {therapy.get("therapy_goals", "")}</div>', unsafe_allow_html=True)

                    oasis = result.get("oasis_alerts", {})
                    st.markdown('<div class="section-header">OASIS Alerts</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="alert-card"><strong>M1033 Hospitalization Risk:</strong> {oasis.get("m1033_hospitalization_risk", "")}<br><strong>M1240 Pain:</strong> {oasis.get("m1240_pain_assessment", "")}<br><strong>M1910 Fall Risk:</strong> {oasis.get("m1910_fall_risk", "")}<br><strong>Mental Health:</strong> {oasis.get("mental_health_flags", "")}<br><strong>Cardiac Rehab:</strong> {oasis.get("cardiac_rehab_indicated", "")}<br><strong>Diabetic Foot Care:</strong> {oasis.get("diabetic_foot_care", "")}<br><strong>Pressure Ulcer Risk:</strong> {oasis.get("pressure_ulcer_risk", "")}<br><strong>M1800 Grooming:</strong> {oasis.get("m1800_grooming", "")}</div>', unsafe_allow_html=True)

                    doc_gaps = result.get("documentation_gaps", [])
                    if doc_gaps:
                        st.markdown('<div class="section-header">Documentation Gaps</div>', unsafe_allow_html=True)
                        for gap in doc_gaps:
                            st.markdown(f'<div class="warning-card">{gap}</div>', unsafe_allow_html=True)

                    warnings = result.get("coding_warnings", [])
                    if warnings:
                        st.markdown('<div class="section-header">Coding Warnings</div>', unsafe_allow_html=True)
                        for warning in warnings:
                            st.markdown(f'<div class="alert-card">{warning}</div>', unsafe_allow_html=True)

                    st.markdown("---")

                    result["analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state.history.append(result)

                    st.download_button(
                        label="Download Full Analysis as JSON",
                        data=json.dumps(result, indent=2),
                        file_name=f"pdx_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                    st.success("Analysis complete and saved to Case History")

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Please check your document and try again")
        else:
            st.info("Upload a document or paste clinical notes on the left to begin")

with tab2:
    st.subheader("Case History")
    if not st.session_state.history:
        st.info("No cases analyzed yet. Go to New Analysis tab to begin.")
    else:
        st.markdown(f"**Total cases analyzed this session: {len(st.session_state.history)}**")
        for i, case in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Case {len(st.session_state.history) - i}: {case.get('patient_name', 'Unknown')} — {case.get('pdx_code', '')} — {case.get('analyzed_at', '')}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Patient:** {case.get('patient_name', '')}")
                    st.markdown(f"**DOB:** {case.get('patient_dob', '')}")
                    st.markdown(f"**Attending:** {case.get('attending_physician', '')}")
                    st.markdown(f"**F2F Date:** {case.get('face_to_face_date', '')}")
                with c2:
                    st.markdown(f"**PDx:** {case.get('pdx_code', '')} — {case.get('pdx_description', '')}")
                    st.markdown(f"**Qualifying Event:** {case.get('qualifying_event', '')}")
                    st.markdown(f"**Skilled Need:** {case.get('skilled_need', {}).get('service', '')}")
                st.download_button(
                    label="Download This Case",
                    data=json.dumps(case, indent=2),
                    file_name=f"case_{case.get('patient_name', 'unknown').replace(' ', '_')}_{case.get('analyzed_at', '').replace(':', '').replace(' ', '_')}.json",
                    mime="application/json",
                    key=f"download_{i}"
                )

with tab3:
    st.subheader("How to Use This Tool")
    st.markdown("""
    **Step 1 — Upload your document**
    - Upload a PDF discharge summary
    - Upload a photo of the document
    - Or paste the clinical notes directly

    **Step 2 — Click Analyze**
    - Tool automatically extracts all patient information
    - Generates PDx and secondary codes
    - Identifies OASIS alerts and documentation gaps
    - Creates physician query letters ready to send

    **Step 3 — Review results**
    - Check PDx code and rationale
    - Review secondary diagnoses for PDGM value
    - Address any physician queries needed
    - Download full analysis as JSON

    **Step 4 — Case History**
    - All analyses saved in Case History tab
    - Review past cases anytime during session
    - Download individual case reports

    **Important — HIPAA Notice**
    - Do not enter real patient names or dates of birth
    - Use initials only until HIPAA compliant hosting is configured
    - This tool uses NVIDIA API which is not HIPAA certified

    **Coding Rules Built In**
    - ICD-10-CM FY2025 guidelines
    - OASIS-E guidelines
    - PDGM clinical grouping rules
    - Home health specific coding requirements
    """)
