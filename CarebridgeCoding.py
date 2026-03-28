import streamlit as st
import pytesseract
import base64
import json
import requests
from PIL import Image
import re
from datetime import datetime
import fitz
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from io import BytesIO
import pandas as pd

# Remove Windows-specific path - Streamlit Cloud has tesseract in PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Common Files\DESIGNER\tesseract.exe"

st.set_page_config(page_title="CareBridge PDx Tool", layout="wide")

st.markdown("""
<style>
.result-card {background: #f8f9fa; border-left: 4px solid #0066cc; padding: 15px; border-radius: 5px; margin: 10px 0;}
.pdx-code {font-size: 32px; font-weight: 700; color: #0066cc;}
.warning-card {background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; border-radius: 5px; margin: 10px 0;}
.query-card {background: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; border-radius: 5px; margin: 10px 0;}
.secondary-card {background: #d4edda; border-left: 4px solid #28a745; padding: 15px; border-radius: 5px; margin: 10px 0;}
.alert-card {background: #f8d7da; border-left: 4px solid #721c24; padding: 15px; border-radius: 5px; margin: 10px 0;}
.info-card {background: #d1ecf1; border-left: 4px solid #0c5460; padding: 15px; border-radius: 5px; margin: 10px 0;}
.confidence-high {background: #d4edda; border-left: 4px solid #28a745; padding: 10px; border-radius: 5px; margin: 5px 0; font-weight: 700;}
.confidence-medium {background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; border-radius: 5px; margin: 5px 0; font-weight: 700;}
.confidence-low {background: #f8d7da; border-left: 4px solid #dc3545; padding: 10px; border-radius: 5px; margin: 5px 0; font-weight: 700;}
.section-header {font-size: 18px; font-weight: 600; color: #1a1a2e; margin-top: 20px; margin-bottom: 10px; border-bottom: 2px solid #0066cc; padding-bottom: 5px;}
.correction-card {background: #e8f4fd; border-left: 4px solid #0066cc; padding: 15px; border-radius: 5px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

NVIDIA_API_KEY = "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"
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-70b-instruct"

if "history" not in st.session_state:
    st.session_state.history = []
if "corrections" not in st.session_state:
    st.session_state.corrections = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "comparison_cases" not in st.session_state:
    st.session_state.comparison_cases = []

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
        image = Image.open(image_file)
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"Error extracting image: {str(e)}"

def build_corrections_context():
    if not st.session_state.corrections:
        return ""
    context = "\n\nLEARNED CORRECTIONS FROM PREVIOUS CASES - APPLY THESE:\n"
    for c in st.session_state.corrections:
        context += f"- When documentation shows: {c['context']}, the correct PDx is {c['correct_code']} not {c['wrong_code']}. Reason: {c['reason']}\n"
    return context

def analyze_clinical_notes(clinical_text):
    corrections_context = build_corrections_context()

    prompt = f"""You are an expert Home Health ICD-10-CM Coding Specialist for OASIS and 485 coding with deep knowledge of PDGM payment model and all CMS guidelines.

Analyze this clinical documentation completely and return ONLY a valid JSON object with no markdown and no explanation.
{corrections_context}

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
  "change_in_condition": "specific symptoms and signs representing change in condition",
  "pdx_code": "primary ICD-10-CM code with highest specificity",
  "pdx_description": "full official description of PDx code",
  "pdx_rationale": "detailed explanation of why this is correct PDx",
  "pdx_alternative": "alternative code if ambiguous or None",
  "confidence_score": "High or Medium or Low",
  "confidence_reason": "why this confidence level was assigned",
  "secondary_codes": [
    {{"code": "ICD-10-CM code", "description": "full description", "rationale": "why included and PDGM value"}}
  ],
  "queries_needed": [
    "specific physician query needed"
  ],
  "physician_query_letters": [
    {{
      "query_topic": "topic of query",
      "query_letter": "complete ready to send physician query letter text"
    }}
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
    "frequency_suggestion": "suggested visit frequency"
  }},
  "medications": {{
    "high_risk": "list all high risk medications",
    "all_medications": "complete medication list if available",
    "medication_teaching_needed": "Yes or No with details",
    "reconciliation_needed": "Yes or No"
  }},
  "oasis_alerts": {{
    "m1033_hospitalization_risk": "High or Medium or Low with reason",
    "m1240_pain_assessment": "pain present Yes or No with details",
    "m1800_grooming": "independent or needs assistance or dependent",
    "m1910_fall_risk": "Yes or No with details",
    "mental_health_flags": "depression or anxiety or dementia if documented or None",
    "cardiac_rehab_indicated": "Yes or No with rationale",
    "diabetic_foot_care": "Yes or No with details",
    "pressure_ulcer_risk": "High or Medium or Low",
    "face_to_face_missing": "Yes or No",
    "homebound_documentation_sufficient": "Yes or No with reason"
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
- Paroxysmal AFib = I48.0
- Pulmonary hypertension group 2 = I27.22
- Tricuspid regurgitation = I07.1
- Morbid obesity = E66.01 plus Z68.4x for BMI
- Sleep apnea do not code if only suspected
- Change in condition must be symptoms signs not the diagnosis itself

Clinical documentation:
{clinical_text}

Return ONLY valid JSON starting with open brace and ending with close brace."""

    raw = call_api(prompt)
    cleaned = re.sub(r'```json|```', '', raw).strip()
    start = cleaned.find('{')
    end = cleaned.rfind('}') + 1
    if start != -1 and end > start:
        cleaned = cleaned[start:end]
    return json.loads(cleaned)

def generate_pdf_report(result):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('title', parent=styles['Title'], fontSize=18, textColor=colors.HexColor('#0066cc'))
    header_style = ParagraphStyle('header', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#1a1a2e'))
    body_style = ParagraphStyle('body', parent=styles['Normal'], fontSize=10, spaceAfter=6)

    story.append(Paragraph("CareBridge Home Health PDx Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Patient Information", header_style))
    patient_data = [
        ["Patient Name", result.get('patient_name', 'Not found')],
        ["Date of Birth", result.get('patient_dob', 'Not found')],
        ["Age / Gender", f"{result.get('patient_age', '')} {result.get('patient_gender', '')}"],
        ["Admission Date", result.get('admission_date', 'Not found')],
        ["Discharge Date", result.get('discharge_date', 'Not found')],
        ["Face to Face Date", result.get('face_to_face_date', 'Not found')],
        ["Attending Physician", result.get('attending_physician', 'Not found')],
        ["Referring Physician", result.get('referring_physician', 'Not found')],
        ["Qualifying Event", result.get('qualifying_event', 'Not found')],
    ]
    t = Table(patient_data, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4fd')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Primary Diagnosis (PDx)", header_style))
    story.append(Paragraph(f"<b>{result.get('pdx_code', '')} — {result.get('pdx_description', '')}</b>", body_style))
    story.append(Paragraph(f"Rationale: {result.get('pdx_rationale', '')}", body_style))
    story.append(Paragraph(f"Confidence: {result.get('confidence_score', '')} — {result.get('confidence_reason', '')}", body_style))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Change in Condition", header_style))
    story.append(Paragraph(result.get('change_in_condition', 'Not documented'), body_style))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Secondary Diagnoses", header_style))
    secondary = result.get('secondary_codes', [])
    if secondary:
        sec_data = [["Code", "Description", "Rationale"]]
        for code in secondary:
            sec_data.append([code.get('code', ''), code.get('description', ''), code.get('rationale', '')])
        t2 = Table(sec_data, colWidths=[1*inch, 2.5*inch, 2.5*inch])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066cc')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(t2)
    story.append(Spacer(1, 0.2*inch))

    queries = result.get('queries_needed', [])
    if queries:
        story.append(Paragraph("Physician Queries Needed", header_style))
        for q in queries:
            story.append(Paragraph(f"- {q}", body_style))
        story.append(Spacer(1, 0.2*inch))

    skilled = result.get('skilled_need', {})
    story.append(Paragraph("Skilled Need", header_style))
    story.append(Paragraph(f"Service: {skilled.get('service', '')}", body_style))
    story.append(Paragraph(f"Rationale: {skilled.get('rationale', '')}", body_style))
    story.append(Paragraph(f"Frequency: {skilled.get('frequency_suggestion', '')}", body_style))
    story.append(Spacer(1, 0.2*inch))

    oasis = result.get('oasis_alerts', {})
    story.append(Paragraph("OASIS Alerts", header_style))
    oasis_data = [
        ["M1033 Hospitalization Risk", oasis.get('m1033_hospitalization_risk', '')],
        ["M1240 Pain Assessment", oasis.get('m1240_pain_assessment', '')],
        ["M1910 Fall Risk", oasis.get('m1910_fall_risk', '')],
        ["Mental Health Flags", oasis.get('mental_health_flags', '')],
        ["Face to Face Missing", oasis.get('face_to_face_missing', '')],
        ["Homebound Documentation", oasis.get('homebound_documentation_sufficient', '')],
        ["Pressure Ulcer Risk", oasis.get('pressure_ulcer_risk', '')],
    ]
    t3 = Table(oasis_data, colWidths=[2.5*inch, 3.5*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fff3cd')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 0.2*inch))

    warnings = result.get('coding_warnings', [])
    if warnings:
        story.append(Paragraph("Coding Warnings", header_style))
        for w in warnings:
            story.append(Paragraph(f"- {w}", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

def render_results(result):
    conf = result.get('confidence_score', 'Medium')
    conf_class = 'confidence-high' if conf == 'High' else 'confidence-low' if conf == 'Low' else 'confidence-medium'
    st.markdown(f'<div class="{conf_class}">Confidence: {conf} — {result.get("confidence_reason", "")}</div>', unsafe_allow_html=True)

    oasis = result.get('oasis_alerts', {})
    missing_flags = []
    if oasis.get('face_to_face_missing') == 'Yes':
        missing_flags.append("Face to face encounter date is missing")
    if oasis.get('homebound_documentation_sufficient') == 'No':
        missing_flags.append("Homebound status not sufficiently documented")
    if result.get('face_to_face_date') == 'Not found':
        missing_flags.append("Face to face date not found in document")
    if missing_flags:
        st.markdown('<div class="section-header">OASIS Compliance Alerts</div>', unsafe_allow_html=True)
        for flag in missing_flags:
            st.markdown(f'<div class="alert-card">ALERT: {flag}</div>', unsafe_allow_html=True)

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
    st.markdown(f"**Attending:** {result.get('attending_physician', 'Not found')}")
    st.markdown(f"**Referring:** {result.get('referring_physician', 'Not found')}")
    st.markdown(f"**Qualifying Event:** {result.get('qualifying_event', 'Not found')}")
    st.markdown(f"**Homebound:** {result.get('homebound_status', 'Not documented')}")

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
            with st.expander(f"Query: {letter.get('query_topic', '')}"):
                st.text_area("Ready to send", letter.get("query_letter", ""), height=200, key=f"letter_{letter.get('query_topic', '')}")

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
    st.markdown(f'<div class="result-card"><strong>High Risk:</strong> {meds.get("high_risk", "None noted")}<br><strong>All Medications:</strong> {meds.get("all_medications", "See discharge summary")}<br><strong>Teaching Needed:</strong> {meds.get("medication_teaching_needed", "")}<br><strong>Reconciliation:</strong> {meds.get("reconciliation_needed", "")}</div>', unsafe_allow_html=True)

    skilled = result.get("skilled_need", {})
    st.markdown('<div class="section-header">Skilled Need</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-card"><strong>Service:</strong> {skilled.get("service", "")}<br><strong>Rationale:</strong> {skilled.get("rationale", "")}<br><strong>Frequency:</strong> {skilled.get("frequency_suggestion", "")}</div>', unsafe_allow_html=True)

    therapy = result.get("therapy_needs", {})
    st.markdown('<div class="section-header">Therapy Assessment</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-card"><strong>PT:</strong> {therapy.get("pt_indicated", "")}<br><strong>OT:</strong> {therapy.get("ot_indicated", "")}<br><strong>ST:</strong> {therapy.get("st_indicated", "")}<br><strong>Goals:</strong> {therapy.get("therapy_goals", "")}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">OASIS Alerts</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="alert-card"><strong>M1033 Risk:</strong> {oasis.get("m1033_hospitalization_risk", "")}<br><strong>M1240 Pain:</strong> {oasis.get("m1240_pain_assessment", "")}<br><strong>M1910 Fall Risk:</strong> {oasis.get("m1910_fall_risk", "")}<br><strong>Mental Health:</strong> {oasis.get("mental_health_flags", "")}<br><strong>Cardiac Rehab:</strong> {oasis.get("cardiac_rehab_indicated", "")}<br><strong>Diabetic Foot:</strong> {oasis.get("diabetic_foot_care", "")}<br><strong>Pressure Ulcer Risk:</strong> {oasis.get("pressure_ulcer_risk", "")}<br><strong>M1800 Grooming:</strong> {oasis.get("m1800_grooming", "")}</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-header">Feedback and Correction</div>', unsafe_allow_html=True)
    st.markdown('<div class="correction-card">If any code above is wrong please correct it here so the tool learns for next time.</div>', unsafe_allow_html=True)

    with st.form(key="correction_form"):
        wrong_code = st.text_input("Wrong code that was suggested", value=result.get('pdx_code', ''))
        correct_code = st.text_input("Correct code it should be")
        correction_context = st.text_area("Brief description of why (optional)", height=80)
        submitted = st.form_submit_button("Submit Correction")
        if submitted and correct_code:
            correction = {
                "wrong_code": wrong_code,
                "correct_code": correct_code,
                "context": correction_context or f"Patient with {result.get('pdx_description', '')}",
                "reason": correction_context or "Coder correction",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "patient": result.get('patient_name', 'Unknown')
            }
            st.session_state.corrections.append(correction)
            st.success(f"Correction saved. Tool will now use {correct_code} in similar cases.")

    st.markdown("---")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            label="Download JSON",
            data=json.dumps(result, indent=2),
            file_name=f"pdx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    with col_dl2:
        try:
            pdf_buffer = generate_pdf_report(result)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name=f"pdx_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.warning(f"PDF generation error: {str(e)}")
    with col_dl3:
        if st.button("Add to Comparison"):
            st.session_state.comparison_cases.append(result)
            st.success("Added to comparison")

st.title("CareBridge Home Health PDx Tool")
st.caption("Upload discharge summary or clinical notes and everything fills automatically")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["New Analysis", "Case History", "Compare Cases", "Monthly Report", "Help"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Documents")

        st.markdown("**You can upload multiple documents**")
        uploaded_pdfs = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
        uploaded_images = st.file_uploader("Upload images of documents", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        pasted_text = st.text_area("Or paste clinical notes here", height=200, placeholder="Paste discharge summary or progress notes here...")

        if st.button("Try Voice Input"):
            st.info("Voice input requires microphone access. Please type your notes or upload a document instead. Voice feature coming soon.")

        all_text = ""
        if uploaded_pdfs:
            for pdf in uploaded_pdfs:
                with st.spinner(f"Extracting {pdf.name}..."):
                    extracted = extract_text_from_pdf(pdf)
                    all_text += f"\n\n--- Document: {pdf.name} ---\n{extracted}"
            st.success(f"{len(uploaded_pdfs)} PDF files extracted")

        if uploaded_images:
            for img_file in uploaded_images:
                image = Image.open(img_file)
                st.image(image, caption=img_file.name, use_column_width=True)
                img_file.seek(0)
                with st.spinner(f"Extracting text from {img_file.name}..."):
                    extracted = extract_text_from_image(img_file)
                    all_text += f"\n\n--- Image: {img_file.name} ---\n{extracted}"
            st.success(f"{len(uploaded_images)} images extracted")

        if pasted_text:
            all_text += f"\n\n--- Pasted Notes ---\n{pasted_text}"

        if all_text:
            with st.expander("View all extracted text"):
                st.text(all_text[:3000])

        analyze_button = st.button(
            "Analyze All Documents and Generate Codes",
            type="primary",
            use_container_width=True,
            disabled=not bool(all_text)
        )

    with col2:
        st.subheader("Analysis Results")

        if analyze_button and all_text:
            with st.spinner("Analyzing all documents — please wait 20 to 30 seconds..."):
                try:
                    result = analyze_clinical_notes(all_text)
                    result["analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state.current_result = result
                    st.session_state.history.append(result)
                    render_results(result)
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Please check your document and try again")

        elif st.session_state.current_result:
            render_results(st.session_state.current_result)
        else:
            st.info("Upload documents or paste clinical notes on the left to begin")

with tab2:
    st.subheader("Case History")

    if not st.session_state.history:
        st.info("No cases analyzed yet.")
    else:
        search_term = st.text_input("Search by patient name, diagnosis, or date", placeholder="Type to search...")

        filtered = st.session_state.history
        if search_term:
            filtered = [c for c in st.session_state.history if
                search_term.lower() in c.get('patient_name', '').lower() or
                search_term.lower() in c.get('pdx_code', '').lower() or
                search_term.lower() in c.get('pdx_description', '').lower() or
                search_term.lower() in c.get('analyzed_at', '').lower()]

        st.markdown(f"**Showing {len(filtered)} of {len(st.session_state.history)} cases**")

        for i, case in enumerate(reversed(filtered)):
            with st.expander(f"{case.get('patient_name', 'Unknown')} — {case.get('pdx_code', '')} — {case.get('analyzed_at', '')}"):
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
                    st.markdown(f"**Confidence:** {case.get('confidence_score', '')}")

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(case, indent=2),
                        file_name=f"case_{i}.json",
                        mime="application/json",
                        key=f"json_{i}"
                    )
                with btn_col2:
                    try:
                        pdf_buf = generate_pdf_report(case)
                        st.download_button(
                            label="Download PDF",
                            data=pdf_buf,
                            file_name=f"case_{i}.pdf",
                            mime="application/pdf",
                            key=f"pdf_{i}"
                        )
                    except Exception:
                        pass

        if st.session_state.corrections:
            st.markdown("---")
            st.markdown("**Correction History**")
            for corr in st.session_state.corrections:
                st.markdown(f"- {corr['date']}: Changed {corr['wrong_code']} to {corr['correct_code']} for {corr['patient']} — {corr['reason']}")

with tab3:
    st.subheader("Side by Side Case Comparison")

    if len(st.session_state.comparison_cases) < 2:
        st.info("Add at least 2 cases to comparison using the Add to Comparison button in the analysis results.")
        if st.session_state.comparison_cases:
            st.markdown(f"**{len(st.session_state.comparison_cases)} case added so far**")
    else:
        case_a = st.session_state.comparison_cases[-2]
        case_b = st.session_state.comparison_cases[-1]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Case A: {case_a.get('patient_name', 'Unknown')}**")
            st.markdown(f"PDx: {case_a.get('pdx_code', '')} — {case_a.get('pdx_description', '')}")
            st.markdown(f"Analyzed: {case_a.get('analyzed_at', '')}")
            st.markdown(f"Confidence: {case_a.get('confidence_score', '')}")
            st.markdown(f"Qualifying Event: {case_a.get('qualifying_event', '')}")
            st.markdown(f"Skilled Need: {case_a.get('skilled_need', {}).get('service', '')}")
            st.markdown("**Secondary Codes:**")
            for code in case_a.get('secondary_codes', []):
                st.markdown(f"- {code.get('code', '')} {code.get('description', '')}")
        with col_b:
            st.markdown(f"**Case B: {case_b.get('patient_name', 'Unknown')}**")
            st.markdown(f"PDx: {case_b.get('pdx_code', '')} — {case_b.get('pdx_description', '')}")
            st.markdown(f"Analyzed: {case_b.get('analyzed_at', '')}")
            st.markdown(f"Confidence: {case_b.get('confidence_score', '')}")
            st.markdown(f"Qualifying Event: {case_b.get('qualifying_event', '')}")
            st.markdown(f"Skilled Need: {case_b.get('skilled_need', {}).get('service', '')}")
            st.markdown("**Secondary Codes:**")
            for code in case_b.get('secondary_codes', []):
                st.markdown(f"- {code.get('code', '')} {code.get('description', '')}")

        if st.button("Clear Comparison"):
            st.session_state.comparison_cases = []
            st.rerun()

with tab4:
    st.subheader("Monthly Report")

    if not st.session_state.history:
        st.info("No cases analyzed yet. Analyze some patients first.")
    else:
        st.markdown(f"**Total cases this session: {len(st.session_state.history)}**")

        pdx_counts = {}
        confidence_counts = {"High": 0, "Medium": 0, "Low": 0}
        skilled_counts = {}
        physicians = {}

        for case in st.session_state.history:
            pdx = case.get('pdx_code', 'Unknown')
            pdx_counts[pdx] = pdx_counts.get(pdx, 0) + 1

            conf = case.get('confidence_score', 'Medium')
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

            skilled = case.get('skilled_need', {}).get('service', 'Unknown')
            skilled_counts[skilled] = skilled_counts.get(skilled, 0) + 1

            physician = case.get('attending_physician', 'Unknown')
            physicians[physician] = physicians.get(physician, 0) + 1

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cases", len(st.session_state.history))
        with col2:
            st.metric("High Confidence", confidence_counts.get("High", 0))
        with col3:
            st.metric("Corrections Made", len(st.session_state.corrections))

        st.markdown("**Most Common PDx Codes**")
        sorted_pdx = sorted(pdx_counts.items(), key=lambda x: x[1], reverse=True)
        for code, count in sorted_pdx[:10]:
            st.markdown(f"- {code}: {count} cases")

        st.markdown("**Skilled Service Distribution**")
        for service, count in skilled_counts.items():
            st.markdown(f"- {service}: {count} cases")

        st.markdown("**Confidence Score Distribution**")
        for level, count in confidence_counts.items():
            st.markdown(f"- {level}: {count} cases")

        st.markdown("**Cases by Physician**")
        for physician, count in sorted(physicians.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"- {physician}: {count} cases")

        report_data = {
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "total_cases": len(st.session_state.history),
            "pdx_distribution": pdx_counts,
            "confidence_distribution": confidence_counts,
            "skilled_distribution": skilled_counts,
            "corrections": st.session_state.corrections,
            "cases": st.session_state.history
        }

        st.download_button(
            label="Download Monthly Report JSON",
            data=json.dumps(report_data, indent=2),
            file_name=f"monthly_report_{datetime.now().strftime('%Y%m')}.json",
            mime="application/json"
        )

with tab5:
    st.subheader("How to Use CareBridge PDx Tool")
    st.markdown("""
    **Step 1 — Upload your documents**
    - Upload one or multiple PDF discharge summaries at once
    - Upload photos of documents
    - Or paste clinical notes directly
    - Tool combines all documents automatically

    **Step 2 — Click Analyze**
    - Tool automatically extracts all patient information
    - Generates PDx and all secondary codes with PDGM value
    - Shows confidence score for each analysis
    - Identifies OASIS alerts and documentation gaps
    - Creates physician query letters ready to send
    - Identifies wound care, lab draw, therapy needs

    **Step 3 — Review and correct**
    - Check confidence score — red means review carefully
    - Submit corrections using the feedback form
    - Tool learns from your corrections for future cases

    **Step 4 — Download reports**
    - Download PDF report ready to print or send
    - Download JSON for records
    - Add cases to comparison for auditing

    **Step 5 — Monthly Report**
    - See all codes used this session
    - Track most common diagnoses
    - Monitor correction history
    - Download full audit trail

    **HIPAA Notice**
    - Do not enter real patient names until HIPAA compliant hosting is configured
    - Use initials only for now
    - This tool uses NVIDIA API which is not HIPAA certified
    - Contact us to upgrade to HIPAA compliant version

    **Built in coding rules**
    - ICD-10-CM FY2025 guidelines
    - OASIS-E guidelines
    - PDGM clinical grouping rules
    - Home health specific coding requirements
    - Learns from your corrections over time
    """)
