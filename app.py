import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai

st.set_page_config(page_title="SupplyChainGPT", page_icon="📦", layout="wide")
st.title("🚢 SupplyChainGPT: Incoterms & Compliance Auditor")
st.markdown("**Open-source AI tool for SMEs** — Upload any Bill of Lading PDF and get instant checks for Incoterms consistency, multilingual accuracy, and regulatory red flags.")

# File uploader
uploaded_file = st.file_uploader("Upload your Bill of Lading or shipping document (PDF)", type="pdf")
text_input = st.text_area("OR paste text here (if you don't have PDF)", height=200)

if uploaded_file or text_input:
    if uploaded_file:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    else:
        text = text_input

    if st.button("🔍 Analyze with AI (Gemini)", type="primary"):
        with st.spinner("Analyzing document... This takes ~10 seconds"):
            # Use secret key (we add it in Streamlit Cloud next)
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel("gemini-2.5-flash")  # free & fast in 2026

            prompt = f"""You are a world-class supply chain compliance auditor specializing in Incoterms 2020, multilingual documentation errors, and regulatory compliance (US/EU customs, CMMC 2.0 implications).

Document text:
{text[:15000]}

Perform these exact tasks and output ONLY in clean Markdown:
1. Extract every Incoterm mentioned.
2. Check consistency (responsibilities, transport mode, risk transfer, cost allocation). Flag any contradictions.
3. If the document is not in English, translate key sections and check if the translation changes legal meaning (consensus-driven accuracy check).
4. Flag any regulatory risks (customs declarations, sanctions language, etc.).
5. Give a severity rating (Low/Medium/High) and exact suggested fixes.

Use tables where helpful. Be extremely precise — this could prevent a $50,000+ shipment delay."""

            response = model.generate_content(prompt)
            st.markdown("### 📋 AI Audit Report")
            st.markdown(response.text)

            st.success("Analysis complete! Share this link with logistics teams.")

# Footer for visa evidence
st.markdown("---")
st.caption("SupplyChainGPT-Audit • Open-source project by Sid Vithal • Launched 2026 • https://github.com/Siddeshvk/SupplyChainGPT-Audit")
