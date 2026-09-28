import io
import os
import time
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# Import system prompts module
import prompts

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
GEMINI_MODEL = "gemini-3.8-flash"
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2

# ==============================================================================
# PAGE CONFIGURATION & METADATA
# ==============================================================================
st.set_page_config(
    page_title="Deadline Tracker — AI Academic Deadline Assistant",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM CSS STYLING (Glassmorphism & Academic Dark Theme)
# ==============================================================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Background Accent */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    /* Header Container Card */
    .header-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Custom Card Style */
    .custom-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Analysis Output Card */
    .analysis-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    /* Footer Text */
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 32px;
        padding-top: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_gemini_api_key():
    """Retrieve Gemini API Key securely from Streamlit secrets or environment variables."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")

def process_uploaded_image(uploaded_file):
    """
    Reads uploaded image (PNG, JPG, JPEG, WEBP) using PIL and converts it 
    in-memory to standard PNG bytes to guarantee full Gemini API compatibility.
    """
    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        
        # Convert RGBA/Paletted images to standard RGB for PNG export
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
            
        byte_arr = io.BytesIO()
        img.save(byte_arr, format="PNG")
        return byte_arr.getvalue(), "image/png"
    except Exception:
        uploaded_file.seek(0)
        mime = uploaded_file.type or "image/png"
        return uploaded_file.getvalue(), mime

def call_gemini_with_retry(client, image_part, system_prompt):
    """
    Executes Gemini Vision call with up to 3 automatic retries (2s, 4s, 8s backoff)
    for 503 temporary service unavailable / high demand errors.
    Does NOT retry authentication errors, invalid models, or malformed inputs.
    """
    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 1. Try Interactions API if available in installed SDK
            if hasattr(client, "interactions") and callable(getattr(client.interactions, "create", None)):
                try:
                    interaction_response = client.interactions.create(
                        model=GEMINI_MODEL,
                        input=[system_prompt, image_part],
                    )
                    if hasattr(interaction_response, "text") and interaction_response.text:
                        return True, interaction_response.text.strip(), None
                    elif hasattr(interaction_response, "output") and interaction_response.output:
                        return True, str(interaction_response.output).strip(), None
                except Exception:
                    pass

            # 2. Standard Google GenAI models.generate_content API call
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    system_prompt,
                    image_part,
                ],
            )

            if response and response.text:
                return True, response.text.strip(), None
            else:
                return False, "", "empty_response"
        except Exception as err:
            err_str = str(err).lower()
            err_code = getattr(err, "code", None) or getattr(err, "status_code", None)

            # Check if error is a 503 temporary server error or high demand
            is_503_error = ("503" in err_str or "unavailable" in err_str or "high demand" in err_str or err_code == 503)

            if is_503_error:
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    return False, "", "503_UNAVAILABLE"
            else:
                # Immediate return for non-503 errors (e.g. 401, 403, 404, 429)
                return False, "", err

    return False, "", "503_UNAVAILABLE"

# ==============================================================================
# INITIALIZE SESSION STATE
# ==============================================================================
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "extracted_deadlines" not in st.session_state:
    st.session_state.extracted_deadlines = []

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("⚙️ Settings")

    st.session_state.user_name = st.text_input(
        "Your Name",
        value=st.session_state.user_name,
        placeholder="e.g. Alex Smith",
        help="Enter your name for personalized notifications.",
    )

    st.session_state.user_email = st.text_input(
        "Email Address",
        value=st.session_state.user_email,
        placeholder="student@university.edu",
        help="Enter your email to receive deadline summary reports.",
    )

    st.markdown("---")
    st.info(
        "💡 **About Deadline Tracker**:\n"
        "Upload a syllabus, timetable, assignment sheet, or notice to track deadlines effortlessly with Gemini AI."
    )

# ==============================================================================
# 1. HEADER SECTION
# ==============================================================================
st.markdown(
    """
<div class="header-card">
    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #818cf8;">📅 Deadline Tracker</h1>
    <h3 style="margin-top: 4px; margin-bottom: 12px; font-size: 1.2rem; font-weight: 500; color: #cbd5e1;">
        Your AI Academic Deadline Assistant
    </h3>
    <p style="margin: 0; color: #94a3b8; font-size: 1rem; line-height: 1.5;">
        Upload your syllabus, assignment schedule, exam timetable, or academic notice and organize your important deadlines with AI.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 3. IMAGE UPLOAD SECTION
# ==============================================================================
st.subheader("📤 Upload Academic Document")

uploaded_file = st.file_uploader(
    "Upload a photo or screenshot of your syllabus, timetable, assignment sheet, exam schedule, or academic notice.",
    type=["png", "jpg", "jpeg", "webp"],
    help="Supported file types: PNG, JPG, JPEG, WEBP",
)

if uploaded_file is not None:
    # Reset analysis if a new file is uploaded
    if st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.analysis_complete = False
        st.session_state.analysis_result = ""

    st.success(f"File uploaded successfully: `{uploaded_file.name}`")

    # Display image preview
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Document: {uploaded_file.name}", use_container_width=True)
    except Exception:
        st.error("Error rendering image preview.")

st.markdown("---")

# ==============================================================================
# 4 & 5. ANALYSIS & GEMINI 3.8 FLASH VISION INTEGRATION WITH RETRY LOGIC
# ==============================================================================
st.subheader("📋 Extracted Deadlines")

analyze_clicked = st.button("🔍 Analyze Document", type="primary")

if analyze_clicked:
    if st.session_state.uploaded_file is None:
        st.warning("Please upload an academic document first.")
    else:
        api_key = get_gemini_api_key()
        if not api_key:
            st.error("⚠️ Gemini API Key is missing. Please add `GEMINI_API_KEY` to `.streamlit/secrets.toml` or set it in your environment variables.")
        else:
            with st.spinner("🔍 Analyzing your document with Gemini 3.8 Flash..."):
                # Process image bytes & mime type for multimodal input
                image_bytes, mime_type = process_uploaded_image(st.session_state.uploaded_file)

                # Construct multimodal Part using google-genai SDK
                image_part = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                )

                # Initialize Google GenAI client
                client = genai.Client(api_key=api_key)

                # Execute request with exponential backoff retry for 503 high-demand errors
                success, result_text, error_detail = call_gemini_with_retry(
                    client, image_part, prompts.SYSTEM_PROMPT
                )

                if success and result_text:
                    st.session_state.analysis_result = result_text.strip()
                    st.session_state.analysis_complete = True
                    st.success("Document analyzed successfully with Gemini 3.8 Flash!")
                else:
                    # Classify error types cleanly
                    if error_detail == "503_UNAVAILABLE":
                        st.warning("Gemini is temporarily busy. Please try again in a few moments.")
                    elif isinstance(error_detail, Exception):
                        err_str = str(error_detail).lower()
                        if "401" in err_str or "403" in err_str or "invalid api key" in err_str or "unauthorized" in err_str:
                            st.error("⚠️ Gemini API Key authentication error. Please verify your API key in `.streamlit/secrets.toml`.")
                        elif "404" in err_str or "not_found" in err_str:
                            st.error("⚠️ Configured Gemini model was not found. Please verify model availability.")
                        elif "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                            st.error("⚠️ Gemini API rate limit or quota exceeded. Please wait a moment before trying again.")
                        else:
                            st.error("⚠️ Gemini analysis error. Please try again.")
                    else:
                        st.warning("Gemini returned an empty response. Please check if the document image is clear and contains readable text.")

# Display Extracted Deadlines Result
if st.session_state.analysis_complete and st.session_state.analysis_result:
    st.markdown(
        f"""
        <div class="analysis-card">
            {st.session_state.analysis_result}
        </div>
        """,
        unsafe_allow_html=True,
    )
elif not st.session_state.analysis_complete:
    st.markdown(
        """
        <div class="custom-card" style="text-align: center; padding: 32px; color: #94a3b8;">
            <p style="font-size: 1.1rem; margin: 0;">Upload a document and click <b>Analyze Document</b> to extract your academic deadlines.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ==============================================================================
# 6. CHAT PLACEHOLDER (For future step)
# ==============================================================================
st.subheader("💬 Ask About Your Deadlines")

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask something about your deadlines...")

if user_question:
    st.session_state.chat_messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    placeholder_reply = "AI chat will be connected in the next development step."
    with st.chat_message("assistant"):
        st.info(placeholder_reply)

    st.session_state.chat_messages.append({"role": "assistant", "content": placeholder_reply})

st.markdown("---")

# ==============================================================================
# 7. SUMMARY SECTION (For future step)
# ==============================================================================
st.subheader("📨 Deadline Summary")

send_summary_clicked = st.button("📧 Send Deadline Summary")

if send_summary_clicked:
    st.info("Email delivery will be connected after deadline extraction and AI chat are implemented.")

st.markdown("---")

# ==============================================================================
# 8. FOOTER
# ==============================================================================
st.markdown(
    '<div class="footer-text">Built with Python • Streamlit • Google Gemini</div>',
    unsafe_allow_html=True,
)
