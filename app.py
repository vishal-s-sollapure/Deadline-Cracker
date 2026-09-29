import datetime
import io
import json
import os
import re
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
from PIL import Image
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Literal

import importlib
import prompts

# Force reload of system prompts module to ensure fresh prompt attributes
importlib.reload(prompts)

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
GEMINI_MODEL = "gemini-3.8-flash"
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2

# ==============================================================================
# PYDANTIC SCHEMA FOR STRUCTURED GEMINI EXTRACTION
# ==============================================================================
class DeadlineItem(BaseModel):
    type: Literal["Assignment", "Exam", "Project", "Submission", "Quiz", "Presentation", "Other"] = Field(
        description="Category of the academic deadline."
    )
    subject: str = Field(
        default="",
        description="Subject or course name/code (e.g. DBMS). Return empty string if missing."
    )
    title: str = Field(
        default="",
        description="Task title or description (e.g. Database Design and Normalization). Return empty string if missing."
    )
    date: str = Field(
        default="",
        description="Deadline date in YYYY-MM-DD format if clear, or original date text. Return empty string if missing."
    )
    time: str = Field(
        default="",
        description="Deadline time if specified (e.g. 11:59 PM). Return empty string if missing."
    )
    venue: str = Field(
        default="",
        description="Location, room number, or submission portal. Return empty string if missing."
    )
    remarks: str = Field(
        default="",
        description="Additional instructions, group details, weightage, or notes. Return empty string if missing."
    )

class DeadlineExtractionSchema(BaseModel):
    document_type: Literal["syllabus", "timetable", "assignment_sheet", "exam_schedule", "academic_notice", "mixed"] = Field(
        description="Primary type of the analyzed document."
    )
    deadlines: List[DeadlineItem] = Field(
        default_factory=list,
        description="List of extracted academic deadlines."
    )

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

    /* Summary Bar Card */
    .summary-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(129, 140, 248, 0.2);
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
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
# HELPER FUNCTIONS, SECRETS & DATE SORTING
# ==============================================================================
def get_gemini_api_key():
    """Retrieve Gemini API Key securely from Streamlit secrets or environment variables."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")

def get_email_secrets():
    """Retrieve Gmail sender credentials securely from Streamlit secrets."""
    try:
        sender_email = st.secrets.get("EMAIL_ADDRESS", "")
        app_password = st.secrets.get("EMAIL_APP_PASSWORD", "")
        return sender_email, app_password
    except Exception:
        return "", ""

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

def parse_and_validate_deadlines_json(raw_json_str):
    """
    Safely parses Gemini's raw JSON response, validates fields against schema expectations,
    and returns (document_type, list_of_deadline_dicts, error_message).
    """
    if not raw_json_str:
        return "mixed", [], "Empty response received from Gemini."

    # Clean potential Markdown code fences
    clean_str = raw_json_str.strip()
    if clean_str.startswith("```json"):
        clean_str = clean_str[7:]
    elif clean_str.startswith("```"):
        clean_str = clean_str[3:]
    if clean_str.endswith("```"):
        clean_str = clean_str[:-3]
    clean_str = clean_str.strip()

    try:
        data = json.loads(clean_str)
    except json.JSONDecodeError as err:
        return "mixed", [], f"JSON syntax error: {str(err)}"

    if not isinstance(data, dict):
        return "mixed", [], "Invalid JSON format: Top-level structure must be an object."

    doc_type = data.get("document_type", "mixed")
    allowed_doc_types = {"syllabus", "timetable", "assignment_sheet", "exam_schedule", "academic_notice", "mixed"}
    if doc_type not in allowed_doc_types:
        doc_type = "mixed"

    raw_deadlines = data.get("deadlines")
    if not isinstance(raw_deadlines, list):
        return doc_type, [], "Invalid JSON structure: 'deadlines' field must be a list."

    allowed_types = {"Assignment", "Exam", "Project", "Submission", "Quiz", "Presentation", "Other"}
    validated_deadlines = []

    for item in raw_deadlines:
        if not isinstance(item, dict):
            continue

        raw_type = str(item.get("type", "Other")).strip()
        matching_type = next((t for t in allowed_types if t.lower() == raw_type.lower()), "Other")

        clean_item = {
            "type": matching_type,
            "subject": str(item.get("subject", "")).strip(),
            "title": str(item.get("title") or item.get("task_name") or item.get("task") or "").strip(),
            "date": str(item.get("date") or item.get("deadline") or "").strip(),
            "time": str(item.get("time", "")).strip(),
            "venue": str(item.get("venue") or item.get("location") or "").strip(),
            "remarks": str(item.get("remarks") or item.get("additional_info") or "").strip(),
        }
        validated_deadlines.append(clean_item)

    return doc_type, validated_deadlines, None

def parse_date_for_sorting(date_str: str):
    """
    Parses a date string for chronological sorting.
    Returns (0, date_object) for valid dates (sorted chronologically).
    Returns (1, datetime.date.max) for invalid/empty/ambiguous dates (sorted to bottom).
    """
    if not date_str or not isinstance(date_str, str):
        return (1, datetime.date.max)

    clean_str = date_str.strip()
    if not clean_str or clean_str.lower() in ("not specified", "unknown", "n/a", "none", "tbd"):
        return (1, datetime.date.max)

    # Try ISO YYYY-MM-DD
    try:
        dt = datetime.datetime.strptime(clean_str[:10], "%Y-%m-%d").date()
        return (0, dt)
    except Exception:
        pass

    # Try standard date formats
    formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%d %b %Y", "%b %d, %Y", "%d %B %Y", "%B %d, %Y"
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(clean_str, fmt).date()
            return (0, dt)
        except Exception:
            pass

    # Regex search for YYYY-MM-DD anywhere in string
    iso_match = re.search(r"\b(\d{4})-\d{2}-\d{2}\b", clean_str)
    if iso_match:
        try:
            dt = datetime.datetime.strptime(iso_match.group(0), "%Y-%m-%d").date()
            return (0, dt)
        except Exception:
            pass

    return (1, datetime.date.max)

def generate_deadline_summary(deadlines: list, student_name: str = "") -> str:
    """
    Deterministically generates a clean, readable Markdown academic deadline summary
    from extracted deadline dictionaries and student name without API calls.
    """
    if not deadlines:
        return "No academic deadlines were found to generate a summary."

    name_display = student_name.strip() if student_name and student_name.strip() else "Student"

    # Sort deadlines chronologically by date
    sorted_deadlines = sorted(
        deadlines,
        key=lambda item: parse_date_for_sorting(item.get("date", ""))
    )

    total_count = len(sorted_deadlines)

    assignments = []
    projects = []
    exams = []
    others = []
    upcoming_dated = []

    for d in sorted_deadlines:
        t_lower = (d.get("type") or "").strip().lower()

        # Dated items for chronological overview
        is_valid_date, _ = parse_date_for_sorting(d.get("date", ""))
        if is_valid_date == 0:
            upcoming_dated.append(d)

        if "exam" in t_lower or "quiz" in t_lower:
            exams.append(d)
        elif "project" in t_lower:
            projects.append(d)
        elif "assignment" in t_lower or "submission" in t_lower or "presentation" in t_lower:
            assignments.append(d)
        else:
            others.append(d)

    lines = []
    lines.append("### 📅 Academic Deadline Summary")
    lines.append(f"**Student**: {name_display}\n")
    lines.append(f"📌 **Overview**: You currently have **{total_count}** academic deadline{'s' if total_count != 1 else ''}.\n")

    # Upcoming Deadlines section (top dated items)
    if upcoming_dated:
        lines.append("#### ⏳ Upcoming Chronological Deadlines")
        for idx, d in enumerate(upcoming_dated[:5], start=1):
            title = d.get("title") or "Task"
            subj = d.get("subject")
            date_val = d.get("date") or "Date not specified"
            subj_str = f" ({subj})" if subj else ""
            lines.append(f"{idx}. **{title}**{subj_str}")
            lines.append(f"   - **Due**: `{date_val}`")
            if d.get("time"):
                lines.append(f"   - **Time**: {d.get('time')}")
            if d.get("venue"):
                lines.append(f"   - **Venue/Portal**: {d.get('venue')}")
        lines.append("")

    # Assignments Section
    if assignments:
        lines.append("#### 📝 Assignments & Submissions")
        for d in assignments:
            title = d.get("title") or "Assignment"
            subj = d.get("subject")
            date_val = d.get("date") or "Not specified"
            subj_str = f" — *{subj}*" if subj else ""
            details = []
            if d.get("time"):
                details.append(f"Time: {d['time']}")
            if d.get("venue"):
                details.append(f"Venue/Portal: {d['venue']}")
            if d.get("remarks"):
                details.append(f"Notes: {d['remarks']}")

            detail_str = f" ({', '.join(details)})" if details else ""
            lines.append(f"• **{title}**{subj_str} | **Due**: `{date_val}`{detail_str}")
        lines.append("")

    # Projects Section
    if projects:
        lines.append("#### 🚀 Projects")
        for d in projects:
            title = d.get("title") or "Project"
            subj = d.get("subject")
            date_val = d.get("date") or "Not specified"
            subj_str = f" — *{subj}*" if subj else ""
            details = []
            if d.get("time"):
                details.append(f"Time: {d['time']}")
            if d.get("venue"):
                details.append(f"Venue/Portal: {d['venue']}")
            if d.get("remarks"):
                details.append(f"Notes: {d['remarks']}")

            detail_str = f" ({', '.join(details)})" if details else ""
            lines.append(f"• **{title}**{subj_str} | **Due**: `{date_val}`{detail_str}")
        lines.append("")

    # Exams Section
    if exams:
        lines.append("#### 🧪 Exams & Quizzes")
        for d in exams:
            title = d.get("title") or "Exam"
            subj = d.get("subject")
            date_val = d.get("date") or "Not specified"
            subj_str = f" — *{subj}*" if subj else ""
            lines.append(f"• **{title}**{subj_str} | **Date**: `{date_val}`")
            if d.get("time"):
                lines.append(f"  - **Time**: {d.get('time')}")
            if d.get("venue"):
                lines.append(f"  - **Venue/Room**: {d.get('venue')}")
            if d.get("remarks"):
                lines.append(f"  - **Notes**: {d.get('remarks')}")
        lines.append("")

    # Other Section
    if others:
        lines.append("#### 📌 Other Deadlines")
        for d in others:
            title = d.get("title") or "Event/Task"
            subj = d.get("subject")
            date_val = d.get("date") or "Not specified"
            subj_str = f" — *{subj}*" if subj else ""
            lines.append(f"• **{title}**{subj_str} | **Due**: `{date_val}`")
            if d.get("remarks"):
                lines.append(f"  - **Notes**: {d.get('remarks')}")
        lines.append("")

    # Important Notes Section
    lines.append("#### ⚠️ Important Notes")
    lines.append("• Submission times and venues are listed as specified in the uploaded document.")
    lines.append("• Always refer to your official university portal or syllabus for final instructions.")

    return "\n".join(lines)

def generate_email_text_body(deadlines: list, student_name: str = "") -> str:
    """
    Generates a clean plain-text body for email delivery via Gmail SMTP.
    """
    name_display = student_name.strip() if student_name and student_name.strip() else "Student"

    sorted_deadlines = sorted(
        deadlines,
        key=lambda item: parse_date_for_sorting(item.get("date", ""))
    )

    total_count = len(sorted_deadlines)

    valid_dated = [d for d in sorted_deadlines if parse_date_for_sorting(d.get("date", ""))[0] == 0]
    next_deadline_str = "None"
    if valid_dated:
        first = valid_dated[0]
        f_title = first.get("title") or first.get("subject") or "Upcoming Task"
        f_date = first.get("date") or ""
        next_deadline_str = f"{f_title} — {f_date}"

    body_lines = [
        f"Hello {name_display},\n",
        "Here is your Academic Deadline Summary from Deadline Tracker.\n",
        f"📌 Total Deadlines: {total_count}",
        f"⚡ Next Deadline: {next_deadline_str}\n",
        "--------------------------------------------------\n",
    ]

    assignments = [d for d in sorted_deadlines if any(k in (d.get("type") or "").lower() for k in ["assignment", "submission", "presentation"])]
    projects = [d for d in sorted_deadlines if "project" in (d.get("type") or "").lower()]
    exams = [d for d in sorted_deadlines if any(k in (d.get("type") or "").lower() for k in ["exam", "quiz"])]
    others = [d for d in sorted_deadlines if d not in assignments and d not in projects and d not in exams]

    if assignments or projects:
        body_lines.append("📚 ASSIGNMENTS & PROJECTS\n")
        idx = 1
        for d in (assignments + projects):
            title = d.get("title") or "Task"
            subj = d.get("subject") or "Not specified"
            date_val = d.get("date") or "Not specified"
            t_type = d.get("type") or "Assignment"
            time_val = d.get("time") or "Not specified"
            venue_val = d.get("venue") or "Not specified"

            body_lines.append(f"{idx}. {title}")
            body_lines.append(f"   Subject: {subj}")
            body_lines.append(f"   Date: {date_val}")
            body_lines.append(f"   Type: {t_type}")
            if time_val != "Not specified":
                body_lines.append(f"   Time: {time_val}")
            if venue_val != "Not specified":
                body_lines.append(f"   Venue: {venue_val}")
            if d.get("remarks"):
                body_lines.append(f"   Notes: {d.get('remarks')}")
            body_lines.append("")
            idx += 1

    if exams:
        body_lines.append("📝 EXAMS & QUIZZES\n")
        idx = 1
        for d in exams:
            title = d.get("title") or "Exam"
            subj = d.get("subject") or "Not specified"
            date_val = d.get("date") or "Not specified"
            time_val = d.get("time") or "Not specified"
            venue_val = d.get("venue") or "Not specified"

            body_lines.append(f"{idx}. {title}")
            body_lines.append(f"   Subject: {subj}")
            body_lines.append(f"   Date: {date_val}")
            body_lines.append(f"   Time: {time_val}")
            body_lines.append(f"   Venue: {venue_val}")
            if d.get("remarks"):
                body_lines.append(f"   Notes: {d.get('remarks')}")
            body_lines.append("")
            idx += 1

    if others:
        body_lines.append("📌 OTHER DEADLINES\n")
        idx = 1
        for d in others:
            title = d.get("title") or "Task"
            subj = d.get("subject") or "Not specified"
            date_val = d.get("date") or "Not specified"

            body_lines.append(f"{idx}. {title}")
            body_lines.append(f"   Subject: {subj}")
            body_lines.append(f"   Date: {date_val}")
            if d.get("remarks"):
                body_lines.append(f"   Notes: {d.get('remarks')}")
            body_lines.append("")
            idx += 1

    body_lines.append("--------------------------------------------------")
    body_lines.append("⚠️ Important:")
    body_lines.append("Only information extracted from your uploaded document is included.")
    body_lines.append("Missing dates, times, or venues are listed as 'Not specified'.\n")
    body_lines.append("Generated by Deadline Tracker — AI Academic Deadline Assistant.")

    return "\n".join(body_lines)

def send_deadline_email(sender_email: str, app_password: str, recipient_email: str, subject: str, body: str):
    """
    Sends a plain-text email using Gmail SMTP with STARTTLS on port 587.
    Returns (success: bool, error_message: str or None).
    """
    if not sender_email or not app_password:
        return False, "Sender credentials missing in Streamlit secrets."

    if not recipient_email or "@" not in recipient_email:
        return False, "Invalid recipient email address."

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain", "utf-8"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
        return True, None
    except smtplib.SMTPAuthenticationError:
        return False, "Email authentication failed. Please verify your Gmail App Password in Streamlit secrets."
    except Exception as err:
        return False, f"SMTP delivery error: {str(err)}"

def call_gemini_with_retry(client, image_part, system_prompt):
    """
    Executes Gemini Vision call with structured output schema and retry logic for 503 errors.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=DeadlineExtractionSchema,
        temperature=0.1,
    )
    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if hasattr(client, "interactions") and callable(getattr(client.interactions, "create", None)):
                try:
                    interaction_response = client.interactions.create(
                        model=GEMINI_MODEL,
                        input=[image_part],
                        config=config,
                    )
                    if hasattr(interaction_response, "text") and interaction_response.text:
                        return True, interaction_response.text.strip(), None
                    elif hasattr(interaction_response, "output") and interaction_response.output:
                        return True, str(interaction_response.output).strip(), None
                except Exception:
                    pass

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[image_part],
                config=config,
            )

            if response and response.text:
                return True, response.text.strip(), None
            else:
                return False, "", "empty_response"
        except Exception as err:
            err_str = str(err).lower()
            err_code = getattr(err, "code", None) or getattr(err, "status_code", None)

            is_503_error = ("503" in err_str or "unavailable" in err_str or "high demand" in err_str or err_code == 503)

            if is_503_error:
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    return False, "", "503_UNAVAILABLE"
            else:
                return False, "", err

    return False, "", "503_UNAVAILABLE"

def call_gemini_chat_with_retry(client, deadlines_json_str, chat_history, user_query):
    """
    Executes Gemini text call for deadline chat Q&A using extracted deadlines JSON context.
    Includes current date context and recent chat history for reasoning.
    """
    current_date_str = datetime.date.today().strftime("%Y-%m-%d")

    history_text = ""
    if chat_history:
        history_lines = []
        for msg in chat_history[-6:]:  # Keep recent history turns
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role_label}: {msg['content']}")
        history_text = "\n".join(history_lines)

    prompt_content = f"""TODAY'S DATE: {current_date_str}

EXTRACTED ACADEMIC DEADLINES DATA (Source of Truth):
{deadlines_json_str}

RELEVANT CONVERSATION HISTORY:
{history_text if history_text else "None"}

USER QUESTION:
{user_query}
"""

    chat_system_instruction = getattr(prompts, "DEADLINE_CHAT_PROMPT", prompts.SYSTEM_PROMPT)

    config = types.GenerateContentConfig(
        system_instruction=chat_system_instruction,
        temperature=0.2,
    )

    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt_content],
                config=config,
            )
            if response and response.text:
                return True, response.text.strip(), None
            else:
                return False, "", "empty_response"
        except Exception as err:
            err_str = str(err).lower()
            err_code = getattr(err, "code", None) or getattr(err, "status_code", None)
            is_503_error = ("503" in err_str or "unavailable" in err_str or "high demand" in err_str or err_code == 503)

            if is_503_error:
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    return False, "", "503_UNAVAILABLE"
            else:
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

if "document_type" not in st.session_state:
    st.session_state.document_type = ""

if "json_parse_error" not in st.session_state:
    st.session_state.json_parse_error = None

if "deadline_summary" not in st.session_state:
    st.session_state.deadline_summary = ""

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
        st.session_state.extracted_deadlines = []
        st.session_state.analysis_complete = False
        st.session_state.analysis_result = ""
        st.session_state.document_type = ""
        st.session_state.json_parse_error = None
        st.session_state.deadline_summary = ""

    st.success(f"File uploaded successfully: `{uploaded_file.name}`")

    # Display image preview
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Document: {uploaded_file.name}", use_container_width=True)
    except Exception:
        st.error("Error rendering image preview.")

st.markdown("---")

# ==============================================================================
# 4 & 5. ANALYSIS & GEMINI VISION STRUCTURED EXTRACTION
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
                image_bytes, mime_type = process_uploaded_image(st.session_state.uploaded_file)

                image_part = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                )

                client = genai.Client(api_key=api_key)

                success, result_text, error_detail = call_gemini_with_retry(
                    client, image_part, prompts.SYSTEM_PROMPT_EXTRACT_DEADLINES
                )

                if success and result_text:
                    st.session_state.analysis_result = result_text.strip()
                    doc_type, validated_deadlines, parse_err = parse_and_validate_deadlines_json(result_text)

                    st.session_state.document_type = doc_type
                    st.session_state.extracted_deadlines = validated_deadlines
                    st.session_state.json_parse_error = parse_err
                    st.session_state.analysis_complete = True

                    # Generate deterministic deadline summary
                    st.session_state.deadline_summary = generate_deadline_summary(
                        validated_deadlines,
                        st.session_state.user_name
                    )

                    st.success("Document analyzed successfully with Gemini 3.8 Flash!")
                else:
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

# Render Results / Extracted Deadlines Table
if st.session_state.analysis_complete:
    if st.session_state.json_parse_error:
        st.error(f"⚠️ {st.session_state.json_parse_error}")
        if st.session_state.analysis_result:
            with st.expander("🔍 View Raw AI Output for Debugging"):
                st.code(st.session_state.analysis_result, language="json")
    else:
        deadlines = st.session_state.extracted_deadlines

        if not deadlines:
            st.markdown(
                """
                <div class="custom-card" style="text-align: center; padding: 28px; color: #94a3b8;">
                    <p style="font-size: 1.1rem; margin: 0; font-weight: 500; color: #cbd5e1;">ℹ️ No academic deadlines were found in this document.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Sort deadlines chronologically by date, undated items at bottom
            sorted_deadlines = sorted(
                deadlines,
                key=lambda item: parse_date_for_sorting(item.get("date", ""))
            )

            # Map document types to human-readable labels
            doc_type_map = {
                "syllabus": "Syllabus",
                "timetable": "Timetable",
                "assignment_sheet": "Assignment Sheet",
                "exam_schedule": "Exam Schedule",
                "academic_notice": "Academic Notice",
                "mixed": "Mixed Document",
            }
            doc_type_label = doc_type_map.get(st.session_state.document_type, "Academic Document")

            # Calculate Next Deadline summary info if a valid date exists
            valid_dated = [d for d in sorted_deadlines if parse_date_for_sorting(d.get("date", ""))[0] == 0]
            next_deadline_summary = ""
            if valid_dated:
                first = valid_dated[0]
                first_title = first.get("title") or first.get("subject") or "Upcoming Task"
                first_date = first.get("date")
                next_deadline_summary = f"⚡ <b>Next Deadline</b>: {first_title} — <code style='color:#818cf8;'>{first_date}</code>"

            # Display Summary Bar
            st.markdown(
                f"""
                <div class="summary-card">
                    <div>
                        <span style="font-size: 1.1rem; font-weight: 600; color: #818cf8;">📌 {len(sorted_deadlines)} deadline{"s" if len(sorted_deadlines) != 1 else ""} found</span>
                        <span style="color: #94a3b8; margin-left: 12px; font-size: 0.95rem;">• Document Type: <b style="color: #e2e8f0;">{doc_type_label}</b></span>
                    </div>
                    <div style="font-size: 0.95rem; color: #cbd5e1;">
                        {next_deadline_summary}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Construct DataFrame with required display column names
            df_rows = []
            for d in sorted_deadlines:
                df_rows.append({
                    "Type": d.get("type") or "Other",
                    "Subject": d.get("subject") or "—",
                    "Task / Title": d.get("title") or "—",
                    "Date": d.get("date") or "—",
                    "Time": d.get("time") or "—",
                    "Venue": d.get("venue") or "—",
                    "Remarks": d.get("remarks") or "—",
                })

            df = pd.DataFrame(df_rows)

            # Render clean, responsive Streamlit table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Type": st.column_config.TextColumn("Type", help="Category of deadline"),
                    "Subject": st.column_config.TextColumn("Subject", help="Course / Subject"),
                    "Task / Title": st.column_config.TextColumn("Task / Title", help="Deadline title or task"),
                    "Date": st.column_config.TextColumn("Date", help="Due date (YYYY-MM-DD or original)"),
                    "Time": st.column_config.TextColumn("Time", help="Due time"),
                    "Venue": st.column_config.TextColumn("Venue", help="Venue or submission portal"),
                    "Remarks": st.column_config.TextColumn("Remarks", help="Additional notes / instructions"),
                },
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
# 6. AI DEADLINE ASSISTANT CHAT
# ==============================================================================
st.subheader("💬 Ask About Your Deadlines")

# Display previous conversation history
for message in st.session_state.chat_messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

user_question = st.chat_input("Ask something about your deadlines (e.g. 'What is my next deadline?')...")

if user_question:
    # 1. Store user question and render in chat
    st.session_state.chat_messages.append({"role": "user", "content": user_question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_question)

    # 2. Check empty state (if document not analyzed yet)
    if not st.session_state.analysis_complete or not st.session_state.extracted_deadlines:
        notice_reply = "Please upload and analyze an academic document first so I can answer questions about your deadlines."
        with st.chat_message("assistant", avatar="🤖"):
            st.info(f"⚠️ {notice_reply}")
        st.session_state.chat_messages.append({"role": "assistant", "content": notice_reply})
    else:
        api_key = get_gemini_api_key()
        if not api_key:
            err_reply = "⚠️ Gemini API Key is missing. Please configure GEMINI_API_KEY in secrets.toml."
            with st.chat_message("assistant", avatar="🤖"):
                st.error(err_reply)
            st.session_state.chat_messages.append({"role": "assistant", "content": err_reply})
        else:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    deadlines_json_str = json.dumps(st.session_state.extracted_deadlines, indent=2)
                    client = genai.Client(api_key=api_key)

                    # Exclude current question from history to avoid duplicate input
                    past_history = st.session_state.chat_messages[:-1]

                    success, reply_text, err_detail = call_gemini_chat_with_retry(
                        client=client,
                        deadlines_json_str=deadlines_json_str,
                        chat_history=past_history,
                        user_query=user_question,
                    )

                    if success and reply_text:
                        st.markdown(reply_text)
                        st.session_state.chat_messages.append({"role": "assistant", "content": reply_text})
                    else:
                        if err_detail == "503_UNAVAILABLE":
                            fail_msg = "Gemini is temporarily busy. Please try again in a moment."
                        else:
                            fail_msg = "⚠️ Gemini analysis error. Please try again."
                        st.warning(fail_msg)
                        st.session_state.chat_messages.append({"role": "assistant", "content": fail_msg})

st.markdown("---")

# ==============================================================================
# 7. DEADLINE SUMMARY & GMAIL SMTP SECTION
# ==============================================================================
st.subheader("📨 Deadline Summary")

if not st.session_state.analysis_complete or not st.session_state.extracted_deadlines:
    st.markdown(
        """
        <div class="custom-card" style="text-align: center; padding: 24px; color: #94a3b8;">
            <p style="font-size: 1rem; margin: 0;">Analyze an academic document first to generate your deadline summary.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif st.session_state.analysis_complete and len(st.session_state.extracted_deadlines) == 0:
    st.markdown(
        """
        <div class="custom-card" style="text-align: center; padding: 24px; color: #94a3b8;">
            <p style="font-size: 1rem; margin: 0;">No academic deadlines were found, so a summary cannot be generated.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Generate fresh summary (dynamic to student_name entered in sidebar)
    st.session_state.deadline_summary = generate_deadline_summary(
        st.session_state.extracted_deadlines,
        st.session_state.user_name
    )

    st.markdown(
        f"""
        <div class="analysis-card" style="margin-bottom: 20px;">
        """,
        unsafe_allow_html=True,
    )
    st.markdown(st.session_state.deadline_summary)
    st.markdown("</div>", unsafe_allow_html=True)

# Email Delivery Trigger Button
send_summary_clicked = st.button("📧 Send Deadline Summary")

if send_summary_clicked:
    if not st.session_state.analysis_complete or not st.session_state.extracted_deadlines:
        st.warning("⚠️ Please upload and analyze an academic document first before sending a summary.")
    elif not st.session_state.user_email or "@" not in st.session_state.user_email:
        st.warning("⚠️ Please enter a valid Email Address in the sidebar settings first.")
    else:
        sender_email, app_password = get_email_secrets()
        if not sender_email or not app_password:
            st.error("⚠️ Gmail SMTP credentials missing. Please add `EMAIL_ADDRESS` and `EMAIL_APP_PASSWORD` to `.streamlit/secrets.toml`.")
        else:
            with st.spinner("📧 Sending deadline summary email via Gmail SMTP..."):
                recipient = st.session_state.user_email.strip()
                subject = "📅 Deadline Tracker — Your Academic Deadline Summary"
                email_body = generate_email_text_body(
                    st.session_state.extracted_deadlines,
                    st.session_state.user_name
                )

                sent_success, send_err = send_deadline_email(
                    sender_email=sender_email,
                    app_password=app_password,
                    recipient_email=recipient,
                    subject=subject,
                    body=email_body
                )

                if sent_success:
                    st.success(f"✅ Deadline summary sent successfully to `{recipient}`")
                else:
                    st.error(f"❌ {send_err}")

st.markdown("---")

# ==============================================================================
# 8. FOOTER
# ==============================================================================
st.markdown(
    '<div class="footer-text">Built with Python • Streamlit • Google Gemini</div>',
    unsafe_allow_html=True,
)
