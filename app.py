import os
import streamlit as st
import base64
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from paddleocr import PPStructureV3
from openai import OpenAI
import json

# --- 1. SETUP AND AUTHENTICATION (Phase 0) ---

# Load environment variables from .env file
load_dotenv()

# Retrieve AK/SK from environment (secure method)
QIANFAN_AK = os.getenv("QIANFAN_AK")
QIANFAN_SK = os.getenv("QIANFAN_SK")

if not QIANFAN_AK or not QIANFAN_SK:
    st.error("Authentication Error: QIANFAN_AK and QIANFAN_SK environment variables must be set in the .env file.")
    st.stop()

# Baidu AI Studio/Qianfan uses the OpenAI-compatible client structure [cite: 2.4, 3.5]
# Note: You may need to generate an Access Token manually if using AK/SK directly
# The correct endpoint for the AI Studio LLM API is:
BAIDU_BASE_URL = "https://aistudio.baidu.com/llm/lmapi/v3" 
ERNIE_MODEL = "ernie-4.5-vl-thinking" # Multimodal model for complex reasoning

# Initialize the Streamlit session state keys (used to cache objects)
if 'ocr_pipeline' not in st.session_state:
    @st.cache_resource
    def initialize_ocr_pipeline():
        """Initializes the PaddleOCR PPStructureV3 pipeline (slow, so we cache it)."""
        st.info("Initializing PaddleOCR PPStructureV3 pipeline. This may take a minute on first run...")
        # PPStructureV3 is excellent for tables, layout, and outputting Markdown [cite: 1.1, 1.4]
        return PPStructureV3(use_doc_orientation_classify=False, use_doc_unwarping=False)

    st.session_state.ocr_pipeline = initialize_ocr_pipeline()
    st.success("PaddleOCR Pipeline ready.")

if 'ernie_client' not in st.session_state:
    @st.cache_resource
    def initialize_ernie_client(ak, sk):
        """Initializes the ERNIE API client using the Access Key (AK) and Secret Key (SK)."""
        try:
            # We use the official 'qianfan' SDK to handle AK/SK and token refresh securely
            from qianfan import Completion
            client = Completion(ak=ak, sk=sk, endpoint=BAIDU_BASE_URL)
            return client
        except Exception as e:
             st.error(f"Error initializing ERNIE Client: {e}")
             return None
             
    st.session_state.ernie_client = initialize_ernie_client(QIANFAN_AK, QIANFAN_SK)


# --- 2. PADDLEOCR FUNCTION (Phase 1.3) ---

def process_document_to_markdown(pdf_bytes: BytesIO) -> str:
    """Uses PPStructureV3 to convert PDF bytes to structured Markdown."""
    
    # PPStructureV3 works best with a file path, so we save the uploaded bytes temporarily
    temp_pdf_path = Path("temp_srs.pdf")
    try:
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_bytes.read())

        # Predict using the cached pipeline
        output = st.session_state.ocr_pipeline.predict(input=str(temp_pdf_path))
        
        full_markdown_content = ""
        
        # Iterate through pages and combine Markdown outputs
        for i, res in enumerate(output):
            temp_md_path = f"temp_page_{i}.md"
            # PPStructureV3 can directly save the structured result as Markdown [cite: 1.3]
            res.save_to_markdown(save_path=temp_md_path) 
            
            with open(temp_md_path, 'r', encoding='utf-8') as f:
                markdown_chunk = f.read()
                full_markdown_content += f"\n\n### Requirements Page {i+1} ---\n"
                full_markdown_content += markdown_chunk
            
            os.remove(temp_md_path)
            
        return full_markdown_content
        
    except Exception as e:
        st.error(f"Error during OCR processing: {e}")
        return "ERROR: Failed to process document via PaddleOCR."
    finally:
        if temp_pdf_path.exists():
            os.remove(temp_pdf_path)


# --- 3. ERNIE MULTIMODAL AUDIT FUNCTION (Phase 1.4 & 2) ---

def run_compliance_audit(srs_markdown: str, image_bytes: BytesIO) -> str:
    """Sends the SRS Markdown and UI Screenshot to ERNIE for compliance check."""
    
    # 1. Encode the image into Base64 for API transmission
    image_base64 = base64.b64encode(image_bytes.read()).decode("utf-8")
    
    # 2. Define the Auditor Persona and Instructions (Phase 2.1)
    # The VLM is instructed to output structured JSON for easy parsing.
    system_prompt = (
        "You are a Senior QA Engineer specializing in software requirements compliance. "
        "Your task is to compare the provided Software Requirements Specification (SRS) against the provided screenshot of the system's UI. "
        "Analyze the UI to find missing features, incorrect labels, or incomplete workflows based on the requirements text. "
        "Output your findings ONLY as a JSON array of compliance issues."
    )
    
    user_prompt = (
        f"SRS Document (Markdown):\n---\n{srs_markdown}\n---\n\n"
        "COMPLIANCE CHECK: Analyze the attached screenshot against the requirements above. "
        "Identify every instance where the visual implementation DOES NOT match the SRS text. "
        "Output a JSON array named 'compliance_issues' detailing the failures. "
        "Strictly use the following JSON Schema for each item:"
        "\n\n```json\n"
        "[\n"
        "  {\n"
        "    \"requirement_id\": \"(e.g., R1.1)\",\n"
        "    \"discrepancy_type\": \"(e.g., Missing Element, Incorrect Label, Mismatched Flow)\",\n"
        "    \"srs_requirement\": \"(The exact requirement text from the document)\",\n"
        "    \"ui_observed\": \"(What is visible in the screenshot)\",\n"
        "    \"severity\": \"(High, Medium, Low)\",\n"
        "    \"justification\": \"(Explain the reasoning for failure)\"\n"
        "  }\n"
        "]\n```"
    )

    # 3. Construct the Multimodal Payload (Phase 1.4)
    # The payload is an array of content parts (text and image)
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": user_prompt},
                # Image payload is passed as Base64 data [cite: 3.2]
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
            ]
        }
    ]

    # 4. Call ERNIE API
    try:
        with st.spinner("ERNIE is analyzing requirements and UI screenshot..."):
            response = st.session_state.ernie_client.chat.completions.create(
                model=ERNIE_MODEL, 
                messages=messages,
                temperature=0.1, # Keep reasoning deterministic
            )
        
        # The response is the string containing the JSON array
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error calling ERNIE API: {e}")
        return "API_ERROR: Check your network connection and ERNIE model service status."


# --- 4. STREAMLIT UI (Phase 3) ---

st.set_page_config(layout="wide", page_title="ERNIE Multimodal Compliance Checker")

st.title("👨‍💻 ERNIE Software Compliance Checker")
st.markdown("Automate QA: Compare a PDF specification (via PaddleOCR) against a UI screenshot (via ERNIE VLM).")

# Split the layout for inputs
col1, col2 = st.columns(2)

with col1:
    srs_file = st.file_uploader(
        "1. Upload Software Requirements Specification (SRS) PDF",
        type=["pdf"],
        help="Upload the document containing tables, lists, and text requirements."
    )

with col2:
    ui_screenshot = st.file_uploader(
        "2. Upload System UI Screenshot/Flowchart (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        help="Upload the visual proof to be audited against the SRS."
    )

audit_button = st.button("▶️ Run Multimodal Compliance Audit", type="primary", use_container_width=True)

# --- EXECUTION LOGIC ---

if audit_button and srs_file and ui_screenshot:
    st.markdown("---")
    st.header("Audit Results")
    
    # A. Display Uploaded Images
    st.subheader("Uploaded Assets")
    c1, c2 = st.columns(2)
    with c1:
        st.text("SRS Document (PDF uploaded)")
    with c2:
        st.text("Prototype Screenshot")
        st.image(ui_screenshot, use_column_width=True)
    
    # B. Phase 1.3: Run OCR Extraction
    with st.spinner("Phase 1: Extracting Structured Requirements from PDF..."):
        srs_markdown = process_document_to_markdown(srs_file)

    if "ERROR" in srs_markdown:
        st.error("Failed to extract requirements. Check PDF quality or file type.")
        st.stop()
        
    st.subheader("Raw OCR Output (For Reference)")
    st.code(srs_markdown[:1000] + "...", language="markdown")

    # C. Phase 1.4 & 2: Run Multimodal Audit
    with st.spinner("Phase 2: Running ERNIE VLM Multimodal Reasoning..."):
        audit_report_json_str = run_compliance_audit(srs_markdown, ui_screenshot)

    # D. Display Final Structured Result
    st.subheader("Final Compliance Report (Structured JSON)")
    
    if "API_ERROR" in audit_report_json_str:
        st.error(f"API Error: {audit_report_json_str}")
    else:
        try:
            # ERNIE should return a clean JSON array
            audit_data = json.loads(audit_report_json_str)
            st.json(audit_data)
            
            st.success(f"Audit Complete: Found {len(audit_data)} compliance issue(s).")
            
        except json.JSONDecodeError:
            st.warning("ERNIE did not return clean JSON. Displaying raw output:")
            st.code(audit_report_json_str)

elif audit_button:
    st.warning("Please upload both the SRS PDF and the UI Screenshot to run the audit.")