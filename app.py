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
# MODEL CONFIGURATION & RETRY CONSTANTS
# ==============================================================================
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"]
MAX_RETRIES = 2
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

    /* Next Deadline Highlight Card */
    .next-deadline-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(30, 27, 75, 0.8) 100%);
        border: 1px solid rgba(129, 140, 248, 0.35);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
    }

    /* Insights Box Card */
    .insight-box {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
    }

    /* Footer Text */
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.88rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# HELPER FUNCTIONS, SECRETS & DATE CALCULATIONS
# ==============================================================================
def get_gemini_api_key():
    """Retrieve Gemini API Key securely from sidebar input, secrets, or environment."""
    if "user_gemini_key" in st.session_state and st.session_state.user_gemini_key.strip():
        return st.session_state.user_gemini_key.strip()
    try:
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"].strip():
            return st.secrets["GEMINI_API_KEY"].strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip()

def get_email_secrets():
    """Retrieve Gmail sender credentials securely from Streamlit secrets."""
    try:
        sender_email = (st.secrets.get("EMAIL_ADDRESS") or st.secrets.get("SENDER_EMAIL", "")).strip()
        app_password = (st.secrets.get("EMAIL_APP_PASSWORD") or st.secrets.get("SENDER_PASSWORD", "")).strip()
        return sender_email, app_password
    except Exception:
        return "", ""

def process_uploaded_document(uploaded_file):
    """
    Reads uploaded academic document (PNG, JPG, JPEG, WEBP, PDF).
    Returns (bytes, mime_type).
    - For PDF: returns PDF bytes directly with mime_type 'application/pdf'.
    - For images: opens image via PIL and returns standard PNG bytes with mime_type 'image/png'.
    """
    if uploaded_file is None:
        return None, ""

    filename = getattr(uploaded_file, "name", "").lower()
    file_type = getattr(uploaded_file, "type", "").lower()

    if filename.endswith(".pdf") or file_type == "application/pdf":
        uploaded_file.seek(0)
        return uploaded_file.getvalue(), "application/pdf"

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
    deduplicates identical records, sorts deadlines chronologically, and returns
    (document_type, list_of_deadline_dicts, error_message).
    """
    if not raw_json_str:
        return "mixed", [], "⚠️ Gemini returned an empty response. Please try uploading a clearer document."

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
    except Exception:
        return "mixed", [], "⚠️ I couldn't reliably structure all the deadlines from this document. Please try uploading a clearer document."

    if not isinstance(data, dict):
        return "mixed", [], "⚠️ I couldn't reliably structure all the deadlines from this document. Please try uploading a clearer document."

    doc_type = str(data.get("document_type", "mixed")).strip().lower()
    allowed_doc_types = {"syllabus", "timetable", "assignment_sheet", "exam_schedule", "academic_notice", "mixed"}
    if doc_type not in allowed_doc_types:
        doc_type = "mixed"

    raw_deadlines = data.get("deadlines")
    if not isinstance(raw_deadlines, list):
        return doc_type, [], "⚠️ I couldn't reliably structure all the deadlines from this document. Please try uploading a clearer document."

    allowed_types = {"Assignment", "Exam", "Project", "Submission", "Quiz", "Presentation", "Other"}
    seen_keys = set()
    validated_deadlines = []

    for item in raw_deadlines:
        if not isinstance(item, dict):
            continue

        raw_type = str(item.get("type", "Other")).strip()
        matching_type = next((t for t in allowed_types if t.lower() == raw_type.lower()), "Other")

        def clean_field(val):
            if val is None:
                return ""
            s = str(val).strip()
            if s.lower() in ("null", "none", "unknown", "tbd", "n/a", "not specified", "unclear", "—"):
                return ""
            return s

        subject_val = clean_field(item.get("subject"))
        title_val = clean_field(item.get("title") or item.get("task_name") or item.get("task"))
        date_val = clean_field(item.get("date") or item.get("deadline"))
        time_val = clean_field(item.get("time"))
        venue_val = clean_field(item.get("venue") or item.get("location"))
        remarks_val = clean_field(item.get("remarks") or item.get("additional_info"))

        # Deduplication key based on (type, subject, title, date)
        dedup_key = (matching_type.lower(), subject_val.lower(), title_val.lower(), date_val.lower())
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        clean_item = {
            "type": matching_type,
            "subject": subject_val,
            "title": title_val,
            "date": date_val,
            "time": time_val,
            "venue": venue_val,
            "remarks": remarks_val,
        }
        validated_deadlines.append(clean_item)

    # Sort deadlines chronologically: Dated deadlines first (sorted by date), undated deadlines after
    def deadline_sort_key(d):
        d_str = d.get("date", "")
        _, _, sort_tuple, _ = compute_deadline_status_and_days(d_str)
        return sort_tuple

    validated_deadlines.sort(key=deadline_sort_key)

    return doc_type, validated_deadlines, None

def compute_deadline_status_and_days(date_str: str):
    """
    Parses a date string relative to current system date (today).
    Returns (status_label, days_left_str, sort_key, parsed_date_obj)
    
    Statuses:
    🔴 Overdue: date is before today
    🟠 Due Today: date is today
    🟡 Due Soon: date is 1-3 days away
    🟢 Upcoming: date is >3 days away
    ⚪ Date unclear: unparseable or ambiguous date
    """
    if not date_str or not isinstance(date_str, str):
        return "⚪ Date unclear", "—", (1, datetime.date.max), None

    clean_str = date_str.strip()
    if not clean_str or clean_str.lower() in ("not specified", "unknown", "n/a", "none", "tbd", "—"):
        return "⚪ Date unclear", "—", (1, datetime.date.max), None

    today = datetime.date.today()
    parsed_dt = None

    # Try ISO YYYY-MM-DD
    try:
        parsed_dt = datetime.datetime.strptime(clean_str[:10], "%Y-%m-%d").date()
    except Exception:
        pass

    if parsed_dt is None:
        formats = [
            "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
            "%d %b %Y", "%b %d, %Y", "%d %B %Y", "%B %d, %Y"
        ]
        for fmt in formats:
            try:
                parsed_dt = datetime.datetime.strptime(clean_str, fmt).date()
                break
            except Exception:
                pass

    if parsed_dt is None:
        iso_match = re.search(r"\b(\d{4})-\d{2}-\d{2}\b", clean_str)
        if iso_match:
            try:
                parsed_dt = datetime.datetime.strptime(iso_match.group(0), "%Y-%m-%d").date()
            except Exception:
                pass

    if parsed_dt is None:
        return "⚪ Date unclear", "—", (1, datetime.date.max), None

    diff_days = (parsed_dt - today).days

    if diff_days < 0:
        status = "🔴 Overdue"
        abs_days = abs(diff_days)
        days_str = f"Overdue by {abs_days} day{'s' if abs_days != 1 else ''}"
        sort_key = (0, parsed_dt)
    elif diff_days == 0:
        status = "🟠 Due Today"
        days_str = "Due today"
        sort_key = (0, parsed_dt)
    elif diff_days == 1:
        status = "🟡 Due Soon"
        days_str = "1 day remaining"
        sort_key = (0, parsed_dt)
    elif diff_days <= 3:
        status = "🟡 Due Soon"
        days_str = f"{diff_days} days remaining"
        sort_key = (0, parsed_dt)
    else:
        status = "🟢 Upcoming"
        days_str = f"{diff_days} days remaining"
        sort_key = (0, parsed_dt)

    return status, days_str, sort_key, parsed_dt

def normalize_deadline_item(item: dict) -> dict:
    """Helper to ensure clean, consistent dictionary fields."""
    return {
        "type": str(item.get("type") or "Other").strip(),
        "subject": str(item.get("subject") or "").strip(),
        "title": str(item.get("title") or item.get("task_name") or item.get("task") or "").strip(),
        "date": str(item.get("date") or item.get("deadline") or "").strip(),
        "time": str(item.get("time") or "").strip(),
        "venue": str(item.get("venue") or item.get("location") or "").strip(),
        "remarks": str(item.get("remarks") or item.get("additional_info") or "").strip(),
    }

def is_api_key_error(err) -> bool:
    """Check if an exception indicates an invalid API key error."""
    if not err:
        return False
    err_str = str(err).lower()
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    return (
        "api_key_invalid" in err_str
        or "invalid api key" in err_str
        or "api key not valid" in err_str
        or "unauthorized" in err_str
        or code in (401, 403)
        or ("400" in err_str and ("key" in err_str or "invalid" in err_str))
    )

def is_rate_limit_error(err) -> bool:
    """Check if an exception or error detail indicates a genuine 429 rate limit or quota failure."""
    if not err:
        return False
    if is_api_key_error(err):
        return False
    err_str = str(err).lower()
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    return ("429" in err_str or "quota" in err_str or "rate limit" in err_str or "resource_exhausted" in err_str or code == 429)

def parse_retry_seconds(err) -> float:
    """Extracts retry wait time in seconds from Gemini API error message if present."""
    if not err:
        return 0.0
    err_str = str(err)
    match = re.search(r"retry in ([\d\.]+)s", err_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.0

# ==============================================================================
# INSIGHTS RENDER HELPERS
# ==============================================================================
def render_upcoming_focus(raw_deadlines: list):
    """Renders 🎯 Upcoming Focus card with the top 3 upcoming deadlines."""
    today = datetime.date.today()
    processed = []
    for d in raw_deadlines:
        status, days_str, sort_key, dt_obj = compute_deadline_status_and_days(d.get("date", ""))
        if dt_obj and dt_obj >= today:
            processed.append((dt_obj, d, status, days_str))

    processed.sort(key=lambda x: x[0])
    top_focus = processed[:3]

    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.9rem; font-weight: 700; color: #818cf8; text-transform: uppercase; margin-bottom: 8px;">🎯 Upcoming Focus</div>', unsafe_allow_html=True)
    if not top_focus:
        st.markdown('<p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">🎉 No upcoming deadlines scheduled!</p>', unsafe_allow_html=True)
    else:
        for idx, (dt_obj, d, status, days_str) in enumerate(top_focus, start=1):
            title = d.get("title") or "Task"
            subj = d.get("subject") or "General"
            date_fmt = dt_obj.strftime("%b %d, %Y")
            st.markdown(
                f"""
                <div style="margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc;">{idx}. {title}</div>
                    <div style="font-size: 0.85rem; color: #cbd5e1;">📚 {subj} • 📅 <code style="color:#818cf8;">{date_fmt}</code> ({days_str})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

def render_weekly_insights(raw_deadlines: list):
    """Renders 📅 This Week insights card."""
    today = datetime.date.today()
    this_week = []
    for d in raw_deadlines:
        status, days_str, sort_key, dt_obj = compute_deadline_status_and_days(d.get("date", ""))
        if dt_obj and 0 <= (dt_obj - today).days <= 7:
            this_week.append((dt_obj, d, status, days_str))

    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.9rem; font-weight: 700; color: #818cf8; text-transform: uppercase; margin-bottom: 8px;">📅 This Week</div>', unsafe_allow_html=True)
    if not this_week:
        st.markdown('<p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">🎉 No deadlines are due this week.</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color: #a5b4fc; font-size: 0.9rem; font-weight: 600; margin-bottom: 8px;">You have {len(this_week)} deadline{"s" if len(this_week) != 1 else ""} this week:</p>', unsafe_allow_html=True)
        for dt_obj, d, status, days_str in this_week[:3]:
            title = d.get("title") or "Task"
            subj = d.get("subject") or "General"
            st.markdown(f'• **{title}** (*{subj}*) — `{days_str}`')
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# LOCAL DEADLINE QUERY ENGINE (DEFINITIVE LOCAL-FIRST SOLUTION)
# ==============================================================================
def answer_local_deadline_query(user_query: str, raw_deadlines: list):
    """
    LOCAL DEADLINE QUERY ENGINE
    Guarantees local processing for all deadline lookups without calling Gemini API.
    Returns (handled: bool, response_text: str).
    """
    if not user_query or not user_query.strip():
        return False, ""

    if not raw_deadlines:
        return True, "📄 Please upload and analyze an academic document first so I can answer questions about your deadlines."

    q = user_query.strip().lower()
    deadlines = [normalize_deadline_item(d) for d in raw_deadlines]
    today = datetime.date.today()

    # Pre-process dates & status for all deadlines
    processed = []
    for d in deadlines:
        status, days_str, sort_key, dt_obj = compute_deadline_status_and_days(d.get("date", ""))
        processed.append({
            "data": d,
            "status": status,
            "days_str": days_str,
            "sort_key": sort_key,
            "dt_obj": dt_obj,
        })

    # Sort processed items chronologically
    processed.sort(key=lambda x: x["sort_key"])

    # Collect unique subjects from extracted deadlines
    extracted_subjects = {}
    for d in deadlines:
        subj = d["subject"].strip()
        if subj:
            extracted_subjects[subj.lower()] = subj

    # -------------------------------------------------------------------------
    # 0. DEADLINE SUMMARY QUERY (LOCAL SUMMARY INSIGHT)
    # -------------------------------------------------------------------------
    if any(k in q for k in ["summary of my deadlines", "deadline summary", "give me a summary", "overall summary", "summarize my deadlines"]):
        total_cnt = len(processed)
        assign_proj_cnt = sum(1 for p in processed if any(k in p["data"]["type"].lower() for k in ["assignment", "project", "submission", "presentation"]))
        exams_cnt = sum(1 for p in processed if any(k in p["data"]["type"].lower() for k in ["exam", "quiz"]))
        overdue_cnt = sum(1 for p in processed if "overdue" in p["status"].lower())
        due_soon_cnt = sum(1 for p in processed if "due soon" in p["status"].lower() or "due today" in p["status"].lower())
        upcoming_cnt = sum(1 for p in processed if "upcoming" in p["status"].lower())

        upcoming_items = [p for p in processed if p["dt_obj"] and p["dt_obj"] >= today]
        next_str = "None"
        if upcoming_items:
            first = upcoming_items[0]
            next_str = f"{first['data']['title'] or 'Task'} — `{first['data']['date']}` ({first['days_str']})"

        summary_msg = (
            f"📊 **Deadline Summary**\n\n"
            f"📌 **Total:** {total_cnt}\n"
            f"📝 **Assignments & Projects:** {assign_proj_cnt}\n"
            f"🧾 **Exams:** {exams_cnt}\n"
            f"🔴 **Overdue:** {overdue_cnt}\n"
            f"🟡 **Due Soon:** {due_soon_cnt}\n"
            f"🟢 **Upcoming:** {upcoming_cnt}\n\n"
            f"⚡ **Next:** {next_str}"
        )
        return True, summary_msg

    # -------------------------------------------------------------------------
    # 1. COUNT QUESTIONS
    # -------------------------------------------------------------------------
    if "how many" in q or "count" in q or "number of" in q:
        if "exam" in q or "quiz" in q:
            count = sum(1 for p in processed if any(k in p["data"]["type"].lower() for k in ["exam", "quiz"]))
            return True, f"📝 You have **{count}** exam{'s' if count != 1 else ''} in your uploaded document."
        elif "assignment" in q or "submission" in q:
            count = sum(1 for p in processed if any(k in p["data"]["type"].lower() for k in ["assignment", "submission"]))
            return True, f"📚 You have **{count}** assignment{'s' if count != 1 else ''} in your uploaded document."
        elif "project" in q:
            count = sum(1 for p in processed if "project" in p["data"]["type"].lower())
            return True, f"🚀 You have **{count}** project{'s' if count != 1 else ''} in your uploaded document."
        elif any(k in q for k in ["deadline", "total", "task", "item"]):
            count = len(processed)
            return True, f"📌 You currently have **{count}** total academic deadline{'s' if count != 1 else ''} extracted from your uploaded document."

    # -------------------------------------------------------------------------
    # 2. NEXT DEADLINE / DUE FIRST
    # -------------------------------------------------------------------------
    if any(k in q for k in ["next deadline", "due first", "what is due first", "which assignment is due first", "which project is due first", "what should i complete first", "what's next", "next task"]):
        if "assignment" in q:
            assigns = [p for p in processed if any(k in p["data"]["type"].lower() for k in ["assignment", "submission"])]
            upcoming_a = [p for p in assigns if p["dt_obj"] and p["dt_obj"] >= today]
            target = upcoming_a[0] if upcoming_a else (assigns[0] if assigns else None)
        elif "project" in q:
            projs = [p for p in processed if "project" in p["data"]["type"].lower()]
            upcoming_p = [p for p in projs if p["dt_obj"] and p["dt_obj"] >= today]
            target = upcoming_p[0] if upcoming_p else (projs[0] if projs else None)
        elif "exam" in q:
            exams = [p for p in processed if any(k in p["data"]["type"].lower() for k in ["exam", "quiz"])]
            upcoming_e = [p for p in exams if p["dt_obj"] and p["dt_obj"] >= today]
            target = upcoming_e[0] if upcoming_e else (exams[0] if exams else None)
        else:
            upcoming = [p for p in processed if p["dt_obj"] and p["dt_obj"] >= today]
            target = upcoming[0] if upcoming else (processed[0] if processed else None)

        if not target:
            return True, "I couldn't find any valid upcoming deadlines in your uploaded document."

        d = target["data"]
        title = d["title"] or "Upcoming Task"
        subj = d["subject"] or "General"
        date_val = d["date"] or "Not specified"
        days_str = target["days_str"]
        status = target["status"]

        return True, (
            f"⚡ **Your Next Deadline**\n\n"
            f"📌 **{title}**\n"
            f"📚 **Subject:** {subj}\n"
            f"📅 **Date:** `{date_val}`\n"
            f"⏳ **Status:** {days_str} ({status})"
        )

    # -------------------------------------------------------------------------
    # 3. OVERDUE DEADLINES
    # -------------------------------------------------------------------------
    if any(k in q for k in ["overdue", "which deadlines are overdue", "what is overdue", "show my overdue deadlines", "past due"]):
        overdue_items = [p for p in processed if "overdue" in p["status"].lower()]
        if not overdue_items:
            return True, "🎉 Great! You have no overdue deadlines."

        lines = ["🔴 **Overdue Deadlines**\n"]
        for idx, p in enumerate(overdue_items, start=1):
            d = p["data"]
            title = d["title"] or "Task"
            subj = f" ({d['subject']})" if d['subject'] else ""
            date_val = d["date"] or "Not specified"
            days_str = p["days_str"]
            lines.append(f"{idx}. **{title}**{subj}")
            lines.append(f"   - **Due Date:** `{date_val}`")
            lines.append(f"   - **Status:** `{days_str}`")
        return True, "\n".join(lines)

    # -------------------------------------------------------------------------
    # 4. CATEGORY FILTERING (EXAMS, ASSIGNMENTS, PROJECTS)
    # -------------------------------------------------------------------------
    if any(k in q for k in ["show all my exams", "show my exams", "list exams", "all exams", "show exams", "my exams", "exams"]) and not any(s in q for s in extracted_subjects.keys()):
        exams = [p for p in processed if any(k in p["data"]["type"].lower() for k in ["exam", "quiz"])]
        if not exams:
            return True, "I couldn't find any exams in your uploaded document."

        lines = ["📝 **Your Exams**\n"]
        for idx, p in enumerate(exams, start=1):
            d = p["data"]
            title = d["title"] or "Exam"
            subj = f" ({d['subject']})" if d['subject'] else ""
            date_val = d["date"] or "Not specified"
            time_val = f"\n   🕐 {d['time']}" if d["time"] else ""
            venue_val = f"\n   📍 {d['venue']}" if d["venue"] else ""
            lines.append(f"{idx}. **{title}**{subj} — `{date_val}` ({p['days_str']}){time_val}{venue_val}\n")
        return True, "\n".join(lines)

    if any(k in q for k in ["show all my assignments", "show my assignments", "list assignments", "all assignments", "show assignments", "my assignments", "assignments"]) and not any(s in q for s in extracted_subjects.keys()):
        assigns = [p for p in processed if any(k in p["data"]["type"].lower() for k in ["assignment", "submission"])]
        if not assigns:
            return True, "I couldn't find any assignments in your uploaded document."

        lines = ["📚 **Your Assignments**\n"]
        for idx, p in enumerate(assigns, start=1):
            d = p["data"]
            title = d["title"] or "Assignment"
            subj = f" ({d['subject']})" if d['subject'] else ""
            date_val = d["date"] or "Not specified"
            lines.append(f"{idx}. **{title}**{subj} | Due: `{date_val}` ({p['days_str']})")
        return True, "\n".join(lines)

    if any(k in q for k in ["show all my projects", "show my projects", "list projects", "all projects", "show projects", "my projects", "projects"]) and not any(s in q for s in extracted_subjects.keys()):
        projs = [p for p in processed if "project" in p["data"]["type"].lower()]
        if not projs:
            return True, "I couldn't find any projects in your uploaded document."

        lines = ["🚀 **Your Projects**\n"]
        for idx, p in enumerate(projs, start=1):
            d = p["data"]
            title = d["title"] or "Project"
            subj = f" ({d['subject']})" if d['subject'] else ""
            date_val = d["date"] or "Not specified"
            lines.append(f"{idx}. **{title}**{subj} | Due: `{date_val}` ({p['days_str']})")
        return True, "\n".join(lines)

    # -------------------------------------------------------------------------
    # 5. SUBJECT MATCHING & LOOKUP (ZERO HALLUCINATION & ZERO API LOOKUP)
    # -------------------------------------------------------------------------
    matched_extracted = []
    for s_lower, s_orig in extracted_subjects.items():
        if s_lower in q:
            matched_extracted.append((len(s_lower), s_lower, s_orig))

    if matched_extracted:
        matched_extracted.sort(key=lambda x: x[0], reverse=True)
        _, target_s_lower, target_s_orig = matched_extracted[0]

        matching = [p for p in processed if p["data"]["subject"].lower() == target_s_lower or target_s_lower in p["data"]["subject"].lower()]
        
        if "exam" in q or "quiz" in q:
            filtered = [p for p in matching if any(k in p["data"]["type"].lower() for k in ["exam", "quiz"])]
            if filtered:
                matching = filtered
        elif "assignment" in q:
            filtered = [p for p in matching if any(k in p["data"]["type"].lower() for k in ["assignment", "submission"])]
            if filtered:
                matching = filtered

        if matching:
            lines = [f"📚 **Deadlines for {target_s_orig}:**\n"]
            for idx, p in enumerate(matching, start=1):
                d = p["data"]
                title = d["title"] or d["type"]
                date_val = d["date"] or "Not specified"
                time_val = f" | 🕐 {d['time']}" if d["time"] else ""
                venue_val = f" | 📍 {d['venue']}" if d["venue"] else ""
                lines.append(f"{idx}. **{title}** (`{d['type']}`)")
                lines.append(f"   - **Date:** `{date_val}` ({p['days_str']}){time_val}{venue_val}")
            return True, "\n".join(lines)

    subj_regex = re.search(r"(?:when is my|deadline for|deadlines for|schedule for|about|search for|show)\s+([a-zA-Z0-9\s&\-\.]+?)(?:\s+exam|\s+assignment|\s+project|\s+deadline|\s+deadlines|\?|$)", q)
    if subj_regex:
        searched_term = subj_regex.group(1).strip()
        stop_words = {"my", "the", "all", "upcoming", "next", "overdue", "this", "a", "an", "particular"}
        if searched_term and searched_term not in stop_words and len(searched_term) > 1:
            matching = [p for p in processed if searched_term in p["data"]["subject"].lower() or searched_term in p["data"]["title"].lower() or p["data"]["subject"].lower() in searched_term]
            if not matching:
                return True, f"I couldn't find any deadline for **{searched_term.title()}** in your uploaded document."
            else:
                lines = [f"📚 **Deadlines for {searched_term.title()}:**\n"]
                for idx, p in enumerate(matching, start=1):
                    d = p["data"]
                    title = d["title"] or d["type"]
                    date_val = d["date"] or "Not specified"
                    lines.append(f"{idx}. **{title}** ({d['subject'] or 'General'}) | Due: `{date_val}` ({p['days_str']})")
                return True, "\n".join(lines)

    # -------------------------------------------------------------------------
    # 6. DATE & MONTH WINDOW FILTERING
    # -------------------------------------------------------------------------
    if "this week" in q:
        items = [p for p in processed if p["dt_obj"] and 0 <= (p["dt_obj"] - today).days <= 7]
        if not items:
            return True, "🎉 No deadlines are due this week."
        lines = ["📅 **Deadlines Due This Week:**\n"]
        for idx, p in enumerate(items, start=1):
            d = p["data"]
            lines.append(f"{idx}. **{d['title'] or 'Task'}** ({d['subject'] or 'General'}) — `{d['date']}` ({p['days_str']})")
        return True, "\n".join(lines)

    if "next week" in q:
        items = [p for p in processed if p["dt_obj"] and 7 < (p["dt_obj"] - today).days <= 14]
        if not items:
            return True, "📅 You have no deadlines due next week."
        lines = ["📅 **Deadlines Due Next Week:**\n"]
        for idx, p in enumerate(items, start=1):
            d = p["data"]
            lines.append(f"{idx}. **{d['title'] or 'Task'}** ({d['subject'] or 'General'}) — `{d['date']}` ({p['days_str']})")
        return True, "\n".join(lines)

    if "this month" in q:
        items = [p for p in processed if p["dt_obj"] and p["dt_obj"].month == today.month and p["dt_obj"].year == today.year]
        if not items:
            return True, "📅 You have no deadlines due this month."
        lines = ["📅 **Deadlines Due This Month:**\n"]
        for idx, p in enumerate(items, start=1):
            d = p["data"]
            lines.append(f"{idx}. **{d['title'] or 'Task'}** ({d['subject'] or 'General'}) — `{d['date']}` ({p['days_str']})")
        return True, "\n".join(lines)

    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    for m_idx, m_name in enumerate(months, start=1):
        if m_name in q:
            items = [p for p in processed if (p["dt_obj"] and p["dt_obj"].month == m_idx) or (m_name[:3] in p["data"]["date"].lower())]
            if not items:
                return True, f"I couldn't find any deadlines due in **{m_name.title()}** in your uploaded document."
            lines = [f"📅 **Deadlines in {m_name.title()}:**\n"]
            for idx, p in enumerate(items, start=1):
                d = p["data"]
                lines.append(f"{idx}. **{d['title'] or 'Task'}** ({d['subject'] or 'General'}) — `{d['date']}` ({p['days_str']})")
            return True, "\n".join(lines)

    if "upcoming" in q or "chronological" in q:
        items = [p for p in processed if p["dt_obj"] and p["dt_obj"] >= today]
        if not items:
            return True, "📌 No upcoming deadlines were found in your uploaded document."
        lines = ["⏳ **Upcoming Deadlines:**\n"]
        for idx, p in enumerate(items, start=1):
            d = p["data"]
            lines.append(f"{idx}. **{d['title'] or 'Task'}** ({d['subject'] or 'General'}) — `{d['date']}` ({p['days_str']})")
        return True, "\n".join(lines)

    # -------------------------------------------------------------------------
    # 7. GENERAL DEADLINE CATCH-ALL LOCAL SEARCH
    # -------------------------------------------------------------------------
    deadline_keywords = [
        "deadline", "deadlines", "due", "exam", "exams", "assignment", "assignments",
        "project", "projects", "submission", "submissions", "subject", "subjects",
        "overdue", "upcoming", "today", "tomorrow", "week", "month", "show", "list",
        "schedule", "table", "notes", "venue", "time", "date", "summary", "overview",
        "dbms", "operating systems", "data structures", "ai", "ml", "web", "when", "what"
    ]

    if any(k in q for k in deadline_keywords) or any(s in q for s in extracted_subjects.keys()):
        lines = ["📋 **Your Extracted Academic Deadlines:**\n"]
        for idx, p in enumerate(processed, start=1):
            d = p["data"]
            title = d["title"] or d["type"]
            subj = f" ({d['subject']})" if d['subject'] else ""
            date_val = d["date"] or "Not specified"
            lines.append(f"{idx}. **{title}**{subj} — `{d['type']}` | `{date_val}` ({p['days_str']})")
        return True, "\n".join(lines)

    return False, ""

def format_deadlines_for_context(deadlines: list) -> str:
    """
    Formats extracted deadline data into clean, structured text context for Gemini AI Chat.
    """
    if not deadlines:
        return "No extracted deadlines available."

    context_lines = []
    for idx, d in enumerate(deadlines, start=1):
        status, days_str, _, _ = compute_deadline_status_and_days(d.get("date", ""))
        context_lines.append(f"Deadline {idx}:")
        context_lines.append(f"  Type: {d.get('type') or 'Other'}")
        context_lines.append(f"  Subject: {d.get('subject') or 'Not specified'}")
        context_lines.append(f"  Task: {d.get('title') or 'Not specified'}")
        context_lines.append(f"  Date: {d.get('date') or 'Not specified'} (Status: {status}, {days_str})")
        context_lines.append(f"  Time: {d.get('time') or 'Not specified'}")
        context_lines.append(f"  Venue: {d.get('venue') or 'Not specified'}")
        context_lines.append(f"  Remarks: {d.get('remarks') or 'None'}")
        context_lines.append("")

    return "\n".join(context_lines)

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
        key=lambda item: compute_deadline_status_and_days(item.get("date", ""))[2]
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
        status, days_str, _, dt_obj = compute_deadline_status_and_days(d.get("date", ""))
        if dt_obj is not None:
            upcoming_dated.append((d, status, days_str))

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
        for idx, (d, status, days_str) in enumerate(upcoming_dated[:5], start=1):
            title = d.get("title") or "Task"
            subj = d.get("subject")
            date_val = d.get("date") or "Date not specified"
            subj_str = f" ({subj})" if subj else ""
            lines.append(f"{idx}. **{title}**{subj_str} — `{status}` ({days_str})")
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
    extracted_remarks = [d.get("remarks").strip() for d in sorted_deadlines if d.get("remarks", "").strip()]
    if extracted_remarks:
        for rem in set(extracted_remarks):
            lines.append(f"• {rem}")
    else:
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
        key=lambda item: compute_deadline_status_and_days(item.get("date", ""))[2]
    )

    total_count = len(sorted_deadlines)

    valid_dated = [d for d in sorted_deadlines if compute_deadline_status_and_days(d.get("date", ""))[3] is not None]
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
            status, days_str, _, _ = compute_deadline_status_and_days(d.get("date", ""))

            body_lines.append(f"{idx}. {title}")
            body_lines.append(f"   Subject: {subj}")
            body_lines.append(f"   Date: {date_val} ({status} — {days_str})")
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
            status, days_str, _, _ = compute_deadline_status_and_days(d.get("date", ""))

            body_lines.append(f"{idx}. {title}")
            body_lines.append(f"   Subject: {subj}")
            body_lines.append(f"   Date: {date_val} ({status} — {days_str})")
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
            status, days_str, _, _ = compute_deadline_status_and_days(d.get("date", ""))

            body_lines.append(f"{idx}. {title}")
            body_lines.append(f"   Subject: {subj}")
            body_lines.append(f"   Date: {date_val} ({status} — {days_str})")
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
    sender_email = (sender_email or "").strip()
    app_password = (app_password or "").strip()
    recipient_email = (recipient_email or "").strip()

    if not sender_email or not app_password:
        return False, "Sender email is not configured in Streamlit secrets."

    if not recipient_email or "@" not in recipient_email:
        return False, "Please enter your email address first."

    server = None
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain", "utf-8"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        return True, None
    except smtplib.SMTPAuthenticationError:
        return False, (
            "Gmail authentication failed. Please check that:\n"
            "1. 2-Step Verification is enabled on the sender Google account.\n"
            "2. A valid Gmail App Password was created.\n"
            "3. EMAIL_ADDRESS matches the Gmail account that created the App Password.\n"
            "4. EMAIL_APP_PASSWORD in Streamlit secrets contains the correct App Password."
        )
    except smtplib.SMTPException as err:
        return False, f"Gmail SMTP error: {str(err)}"
    except Exception as err:
        return False, f"Email delivery error: {str(err)}"
    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass

def call_gemini_with_retry(client, image_part, system_prompt):
    """
    Executes Gemini Vision call with structured output schema, model fallbacks, and retry logic.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=DeadlineExtractionSchema,
        temperature=0.1,
    )

    models_to_try = GEMINI_MODELS
    last_err = None
    rate_limit_err = None

    for model_name in models_to_try:
        backoff = INITIAL_BACKOFF_SECONDS
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[image_part],
                    config=config,
                )
                if response and response.text:
                    return True, response.text.strip(), None
                else:
                    last_err = "empty_response"
            except Exception as err:
                err_str = str(err).lower()
                last_err = err

                if is_api_key_error(err):
                    return False, "", err  # Return API key error immediately without retrying invalid key

                if "404" in err_str or "not found" in err_str or "no longer available" in err_str:
                    break  # Skip non-existent model immediately

                if is_rate_limit_error(err):
                    rate_limit_err = err
                    wait_sec = parse_retry_seconds(err)
                    if wait_sec > 0 and wait_sec <= 6:
                        time.sleep(wait_sec + 0.5)
                        continue
                    elif attempt < MAX_RETRIES:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    else:
                        break  # Try next fallback model
                else:
                    if "503" in err_str or "unavailable" in err_str:
                        if attempt < MAX_RETRIES:
                            time.sleep(backoff)
                            backoff *= 2
                            continue

    final_err = rate_limit_err if rate_limit_err else last_err
    return False, "", final_err

def call_gemini_chat_with_retry(client, deadlines_text_context, user_query, chat_history=None):
    """
    Executes Gemini text call for complex reasoning questions with model fallback, retry backoff and 429 rate limit detection.
    Max 2 retries with exponential backoff (2s, 4s).
    """
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    history_text = ""
    if chat_history:
        history_lines = []
        for msg in chat_history[-6:]:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role_label}: {msg['content']}")
        history_text = "\n".join(history_lines)

    chat_system_instruction = f"""You are Deadline Tracker, an AI academic deadline assistant.

CURRENT TODAY'S DATE: {today_str}

You have access ONLY to the following extracted academic deadline data:

{deadlines_text_context}

Answer the user's question using ONLY this data.

Rules:
1. Never invent information, dates, assignments, subjects, exam times, venues, remarks, or deadlines.
2. Never guess missing dates or details.
3. If the requested information is not present in the extracted data, respond clearly: "I couldn't find that information in your uploaded document."
4. Keep answers concise, student-friendly, and useful.
5. When listing deadlines, use clear bullets or a small table.
6. For date-related questions, use the actual dates from the data.
7. For relative dates such as today, tomorrow, this week, and overdue, calculate them using TODAY'S DATE ({today_str}).
8. Stay focused on academic deadlines. If the user asks an unrelated question (e.g. general programming, entertainment, politics, general trivia), respond: "I'm your Academic Deadline Assistant. I can help you with your assignments, projects, exams, and academic deadlines." """

    prompt_content = f"""Recent Conversation History:
{history_text if history_text else "None"}

USER QUESTION:
{user_query}
"""

    config = types.GenerateContentConfig(
        system_instruction=chat_system_instruction,
        temperature=0.2,
    )

    models_to_try = GEMINI_MODELS
    last_err = None
    rate_limit_err = None

    for model_name in models_to_try:
        backoff = INITIAL_BACKOFF_SECONDS
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt_content],
                    config=config,
                )
                if response and response.text:
                    return True, response.text.strip(), None
                else:
                    last_err = "empty_response"
            except Exception as err:
                err_str = str(err).lower()
                last_err = err

                if is_api_key_error(err):
                    return False, "", err

                if "404" in err_str or "not found" in err_str or "no longer available" in err_str:
                    break

                if is_rate_limit_error(err):
                    rate_limit_err = err
                    wait_sec = parse_retry_seconds(err)
                    if wait_sec > 0 and wait_sec <= 6:
                        time.sleep(wait_sec + 0.5)
                        continue
                    elif attempt < MAX_RETRIES:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    else:
                        break
                else:
                    if "503" in err_str or "unavailable" in err_str:
                        if attempt < MAX_RETRIES:
                            time.sleep(backoff)
                            backoff *= 2
                            continue

    final_err = rate_limit_err if rate_limit_err else last_err
    return False, "", final_err

# ==============================================================================
# INITIALIZE SESSION STATE
# ==============================================================================
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "user_gemini_key" not in st.session_state:
    st.session_state.user_gemini_key = ""

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
    st.subheader("🔑 API Key Override")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.user_gemini_key,
        type="password",
        placeholder="AIzaSy...",
        help="Paste your Gemini API key from https://aistudio.google.com/ if needed.",
    )
    if api_key_input != st.session_state.user_gemini_key:
        st.session_state.user_gemini_key = api_key_input.strip()

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.9rem; color: #94a3b8; line-height: 1.5;">
            💡 <b>About Deadline Tracker</b><br>
            Your AI-powered academic deadline assistant for organizing assignments, projects, exams, and submissions.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 1. HERO SECTION
# ==============================================================================
api_key_check = get_gemini_api_key()
ai_status_pill = (
    '<span style="background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3); padding: 5px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">🟢 AI Assistant Ready</span>'
    if api_key_check
    else '<span style="background: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.3); padding: 5px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">🟡 Gemini API Key Needed</span>'
)

st.markdown(
    f"""
<div class="header-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #818cf8;">📅 Deadline Tracker</h1>
            <h3 style="margin-top: 4px; margin-bottom: 0; font-size: 1.15rem; font-weight: 500; color: #cbd5e1;">
                AI Academic Deadline Assistant
            </h3>
        </div>
        <div>
            {ai_status_pill}
        </div>
    </div>
    <p style="margin-top: 12px; margin-bottom: 0; color: #94a3b8; font-size: 0.98rem; line-height: 1.5;">
        Upload your academic schedule, assignments, exams, or notices and let AI organize your important deadlines.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. DOCUMENT UPLOAD SECTION
# ==============================================================================
st.subheader("📄 Upload Academic Document")
st.caption("Supported: **PNG • JPG • JPEG • WEBP • PDF**")

uploaded_file = st.file_uploader(
    "Upload a syllabus, timetable, assignment sheet, exam schedule, or academic notice.",
    type=["png", "jpg", "jpeg", "webp", "pdf"],
    help="Supported formats: PNG, JPG, JPEG, WEBP, PDF",
)

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_ext = file_name.split(".")[-1].lower() if "." in file_name else ""
    allowed_exts = {"png", "jpg", "jpeg", "webp", "pdf"}

    if file_ext not in allowed_exts:
        st.error("⚠️ Unsupported file format. Please upload a valid PNG, JPG, JPEG, WEBP, or PDF document.")
    else:
        if st.session_state.uploaded_file != uploaded_file:
            st.session_state.uploaded_file = uploaded_file

        st.success(f"File uploaded successfully: `{file_name}`")

        # Display document preview
        if file_ext == "pdf" or uploaded_file.type == "application/pdf":
            file_size_kb = round(len(uploaded_file.getvalue()) / 1024, 1)
            st.info(f"📄 **PDF Document Ready for Analysis:** `{file_name}` ({file_size_kb} KB)")
        else:
            try:
                uploaded_file.seek(0)
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded Document: {file_name}", use_container_width=True)
            except Exception:
                st.warning("⚠️ Image preview unavailable, but the document can still be analyzed.")

st.markdown("---")

# ==============================================================================
# 3. ANALYSIS & SMART DEADLINE DASHBOARD
# ==============================================================================
st.subheader("📋 Smart Deadline Dashboard")

analyze_clicked = st.button("🔍 Analyze Document", type="primary")

if analyze_clicked:
    if st.session_state.uploaded_file is None:
        st.warning("Please upload an academic document first.")
    else:
        api_key = get_gemini_api_key()
        if not api_key:
            st.error("⚠️ Gemini API Key is missing. Please add `GEMINI_API_KEY` in `.streamlit/secrets.toml` or paste your key in Sidebar Settings.")
        else:
            with st.spinner("🔍 Gemini is analyzing your academic document..."):
                doc_bytes, mime_type = process_uploaded_document(st.session_state.uploaded_file)

                doc_part = types.Part.from_bytes(
                    data=doc_bytes,
                    mime_type=mime_type,
                )

                client = genai.Client(api_key=api_key)

                success, result_text, error_detail = call_gemini_with_retry(
                    client, doc_part, prompts.SYSTEM_PROMPT_EXTRACT_DEADLINES
                )

                if success and result_text:
                    doc_type, validated_deadlines, parse_err = parse_and_validate_deadlines_json(result_text)

                    if parse_err:
                        st.error(f"⚠️ {parse_err}")
                        # Keep previous valid dataset safe (Requirement 12)
                    else:
                        st.session_state.analysis_result = result_text.strip()
                        st.session_state.document_type = doc_type
                        st.session_state.extracted_deadlines = validated_deadlines
                        st.session_state.json_parse_error = None
                        st.session_state.analysis_complete = True

                        st.session_state.deadline_summary = generate_deadline_summary(
                            validated_deadlines,
                            st.session_state.user_name
                        )
                else:
                    if is_api_key_error(error_detail):
                        st.error(
                            f"⚠️ **Gemini API Key Authentication Error:**\n\n"
                            f"`{error_detail}`\n\n"
                            f"👉 **How to fix:** Please get a free valid Gemini API key from "
                            f"[Google AI Studio](https://aistudio.google.com/) and paste it into **Sidebar Settings ⚙️ -> Gemini API Key** or update `.streamlit/secrets.toml`."
                        )
                    elif is_rate_limit_error(error_detail) or error_detail == "RATE_LIMIT_EXCEEDED":
                        wait_sec = parse_retry_seconds(error_detail)
                        if wait_sec > 0:
                            st.warning(f"⚠️ **Gemini API Rate Limit Reached:** Google Gemini free tier limit reached (20 requests/min). Please wait **{int(wait_sec + 1)} seconds** and click **Analyze Document** again.")
                        else:
                            st.warning("⚠️ **Gemini API Rate Limit Reached:** Google Gemini free tier rate limit exceeded (20 requests/min). Please wait a few moments and click **Analyze Document** again.")
                    elif error_detail == "503_UNAVAILABLE":
                        st.warning("Gemini is temporarily busy. Please try again in a few moments.")
                    elif isinstance(error_detail, Exception):
                        st.error(f"⚠️ Gemini analysis error: {str(error_detail)}")
                    else:
                        st.warning("Gemini returned an empty response. Please check if the document is clear and contains readable text.")

# Render Smart Dashboard Results
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
            doc_type_map = {
                "syllabus": "Academic Syllabus",
                "timetable": "Course Timetable",
                "assignment_sheet": "Assignment Sheet",
                "exam_schedule": "Exam Schedule",
                "academic_notice": "Academic Notice",
                "mixed": "Mixed Academic Schedule"
            }
            doc_label = doc_type_map.get(st.session_state.document_type, "Mixed Academic Schedule")
            total_cnt = len(deadlines)
            assign_cnt = sum(1 for d in deadlines if d.get("type") in ("Assignment", "Project", "Submission"))
            exam_cnt = sum(1 for d in deadlines if d.get("type") in ("Exam", "Quiz"))

            st.markdown(
                f"""
                <div class="summary-card" style="margin-bottom: 20px; border-left: 4px solid #4ade80;">
                    <div>
                        <span style="font-size: 1.05rem; font-weight: 700; color: #4ade80;">📄 Document analyzed successfully</span>
                        <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 4px;">
                            <strong>Document Type:</strong> {doc_label}
                        </div>
                    </div>
                    <div style="display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.95rem;">
                        <span style="color: #f8fafc;">📌 <strong>Deadlines Found:</strong> {total_cnt}</span>
                        <span style="color: #38bdf8;">📚 <strong>Assignments & Projects:</strong> {assign_cnt}</span>
                        <span style="color: #f43f5e;">📝 <strong>Exams:</strong> {exam_cnt}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Prepare deadline metadata with Python calculations
            processed_deadlines = []
            valid_dated_items = []

            for d in deadlines:
                status, days_str, sort_key, dt_obj = compute_deadline_status_and_days(d.get("date", ""))
                processed_deadlines.append((sort_key, status, days_str, dt_obj, d))
                if dt_obj is not None:
                    valid_dated_items.append((dt_obj, d, status, days_str))

            # Sort valid items by date chronologically
            valid_dated_items.sort(key=lambda x: x[0])

            # Determine Next Deadline: earliest upcoming (>= today), or nearest valid if all overdue
            today = datetime.date.today()
            next_deadline_item = None
            for item in valid_dated_items:
                if item[0] >= today:
                    next_deadline_item = item
                    break
            if not next_deadline_item and valid_dated_items:
                next_deadline_item = valid_dated_items[-1]

            # ------------------------------------------------------------------
            # 1. SMART DEADLINE SUMMARY CARDS (5 Dashboard Cards)
            # ------------------------------------------------------------------
            total_count = len(deadlines)
            overdue_count = sum(1 for _, status, _, _, _ in processed_deadlines if "overdue" in status.lower())
            due_soon_count = sum(1 for _, status, _, _, _ in processed_deadlines if "due soon" in status.lower() or "due today" in status.lower())
            assign_proj_count = sum(
                1 for d in deadlines 
                if any(k in (d.get("type") or "").lower() for k in ["assignment", "project", "submission", "presentation"])
            )
            exams_count = sum(
                1 for d in deadlines 
                if any(k in (d.get("type") or "").lower() for k in ["exam", "quiz"])
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric(label="📌 Total Deadlines", value=total_count)
            with c2:
                st.metric(label="🔴 Overdue", value=overdue_count)
            with c3:
                st.metric(label="🟠 Due Soon", value=due_soon_count)
            with c4:
                st.metric(label="📚 Assignments & Projects", value=assign_proj_count)
            with c5:
                st.metric(label="📝 Exams", value=exams_count)

            # ------------------------------------------------------------------
            # 7 & 8. NEXT DEADLINE HIGHLIGHT CARD WITH SMART STATUS
            # ------------------------------------------------------------------
            if next_deadline_item:
                dt_obj, nd_data, nd_status, nd_days = next_deadline_item
                nd_title = nd_data.get("title") or "Upcoming Task"
                nd_subj = nd_data.get("subject") or "General"
                nd_date_raw = nd_data.get("date") or "Not specified"
                nd_date_fmt = dt_obj.strftime("%B %d, %Y") if dt_obj else nd_date_raw

                is_overdue = "overdue" in nd_status.lower()
                card_title = "🔴 OVERDUE DEADLINE" if is_overdue else "⚡ NEXT DEADLINE"

                st.markdown(
                    f"""
                    <div class="next-deadline-card">
                        <div style="font-size: 0.85rem; font-weight: 700; color: {'#f87171' if is_overdue else '#818cf8'}; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">
                            {card_title}
                        </div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">
                            {nd_title}
                        </div>
                        <div style="font-size: 0.95rem; color: #cbd5e1; margin-bottom: 10px;">
                            📚 <b>Subject:</b> {nd_subj} &nbsp;•&nbsp; 📅 <b>Date:</b> <code style="color:#818cf8;">{nd_date_fmt}</code>
                        </div>
                        <div style="display: inline-block; background: rgba(129, 140, 248, 0.18); border: 1px solid rgba(129, 140, 248, 0.35); border-radius: 6px; padding: 4px 14px; font-size: 0.9rem; font-weight: 600; color: #a5b4fc;">
                            ⏳ {nd_days} &nbsp;({nd_status})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ------------------------------------------------------------------
            # 4, 5 & 6. DEADLINE INSIGHTS (UPCOMING FOCUS & THIS WEEK)
            # ------------------------------------------------------------------
            col_focus, col_week = st.columns(2)
            with col_focus:
                render_upcoming_focus(deadlines)
            with col_week:
                render_weekly_insights(deadlines)

            st.markdown("<br>", unsafe_allow_html=True)

            # ------------------------------------------------------------------
            # FILTERS AND SEARCH SECTION
            # ------------------------------------------------------------------
            search_col, filter_type_col, filter_subj_col, filter_status_col = st.columns([2, 1, 1, 1])

            with search_col:
                search_query = st.text_input("🔎 Search deadlines", placeholder="Search title, subject, venue, remarks...", key="search_query")

            with filter_type_col:
                available_types = ["All"] + sorted(list(set(d.get("type", "Other") for d in deadlines if d.get("type"))))
                selected_type = st.selectbox("Type", options=available_types, key="filter_type")

            with filter_subj_col:
                raw_subjs = sorted(list(set(d.get("subject", "").strip() for d in deadlines if d.get("subject", "").strip())))
                available_subjs = ["All"] + raw_subjs
                selected_subj = st.selectbox("Subject", options=available_subjs, key="filter_subj")

            with filter_status_col:
                status_options = ["All", "🔴 Overdue", "🟠 Due Today", "🟡 Due Soon", "🟢 Upcoming", "⚪ Date unclear"]
                selected_status = st.selectbox("Status", options=status_options, key="filter_status")

            # ------------------------------------------------------------------
            # FILTERING, SEARCHING & CHRONOLOGICAL SORTING
            # ------------------------------------------------------------------
            filtered_deadlines = []

            for sort_key, status, days_str, dt_obj, d in processed_deadlines:
                # Filter by Type
                d_type = d.get("type") or "Other"
                if selected_type != "All" and d_type.lower() != selected_type.lower():
                    continue

                # Filter by Subject
                d_subj = (d.get("subject") or "").strip()
                if selected_subj != "All" and d_subj.lower() != selected_subj.lower():
                    continue

                # Filter by Status
                if selected_status != "All" and status != selected_status:
                    continue

                # Search query matching
                if search_query and search_query.strip():
                    sq = search_query.strip().lower()
                    searchable_text = f"{d.get('title', '')} {d.get('subject', '')} {d.get('type', '')} {d.get('venue', '')} {d.get('remarks', '')} {d.get('date', '')}".lower()
                    if sq not in searchable_text:
                        continue

                filtered_deadlines.append((sort_key, status, days_str, d))

            # Chronological sort (earliest valid date first, unclear dates at bottom)
            filtered_deadlines.sort(key=lambda x: x[0])

            # ------------------------------------------------------------------
            # DISPLAY TABLE WITH STATUS AND DAYS LEFT
            # ------------------------------------------------------------------
            if filtered_deadlines:
                df_rows = []
                for sort_key, status, days_str, d in filtered_deadlines:
                    df_rows.append({
                        "Status": status,
                        "Days Left": days_str,
                        "Type": d.get("type") or "Other",
                        "Subject": d.get("subject") or "—",
                        "Task / Title": d.get("title") or "—",
                        "Date": d.get("date") or "—",
                        "Time": d.get("time") or "—",
                        "Venue": d.get("venue") or "—",
                        "Remarks": d.get("remarks") or "—",
                    })

                df = pd.DataFrame(df_rows)

                st.dataframe(
                    df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Status": st.column_config.TextColumn("Status", help="Deadline status calculated from current date"),
                        "Days Left": st.column_config.TextColumn("Days Left", help="Time remaining or overdue count"),
                        "Type": st.column_config.TextColumn("Type", help="Category of deadline"),
                        "Subject": st.column_config.TextColumn("Subject", help="Course / Subject"),
                        "Task / Title": st.column_config.TextColumn("Task / Title", help="Deadline title or task"),
                        "Date": st.column_config.TextColumn("Date", help="Due date"),
                        "Time": st.column_config.TextColumn("Time", help="Due time"),
                        "Venue": st.column_config.TextColumn("Venue", help="Venue or submission portal"),
                        "Remarks": st.column_config.TextColumn("Remarks", help="Additional notes / instructions"),
                    },
                )
            else:
                st.info("No deadlines match your selected filter or search criteria.")

elif not st.session_state.analysis_complete:
    st.markdown(
        """
        <div class="custom-card" style="text-align: center; padding: 32px; color: #94a3b8;">
            <p style="font-size: 1.1rem; margin: 0;">📄 <b>No deadline data yet.</b> Upload your academic document and click <b>Analyze Document</b> to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ==============================================================================
# 4. AI DEADLINE ASSISTANT CHAT ("💬 Ask About Your Deadlines")
# ==============================================================================
st.subheader("💬 Ask About Your Deadlines")
st.caption("Ask anything about your exams, assignments, projects, or upcoming deadlines.")

# Check if document analysis is complete and deadlines are present
has_extracted_data = st.session_state.analysis_complete and len(st.session_state.extracted_deadlines) > 0

if not has_extracted_data:
    st.info("📄 Please upload and analyze an academic document first so I can answer questions about your deadlines.")
else:
    if not st.session_state.chat_messages:
        # Chat empty state suggestion helper (Requirement 10)
        st.markdown(
            """
            <div style="margin-bottom: 12px; color: #94a3b8; font-size: 0.9rem;">
                💡 <b>Try asking:</b> 
                <span style="color: #818cf8;">"What is my next deadline?"</span> &nbsp;
                <span style="color: #818cf8;">"When is my DBMS exam?"</span> &nbsp;
                <span style="color: #818cf8;">"What is due this week?"</span> &nbsp;
                <span style="color: #818cf8;">"Show my overdue assignments."</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Render Previous Conversation History
for message in st.session_state.chat_messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 2 & 3. QUICK-ACTION BUTTONS (Above Chat Input)
quick_actions = {
    "⚡ Next Deadline": "What is my next deadline?",
    "📝 My Exams": "Show all my exams.",
    "📚 My Assignments": "Show my assignments.",
    "🔴 Overdue": "What is overdue?",
    "📅 Due This Week": "What is due this week?",
    "📊 Deadline Summary": "Give me a summary of my deadlines.",
}

triggered_query = None

if has_extracted_data:
    st.markdown("<div style='margin-bottom: 6px; font-weight: 600; color: #818cf8; font-size: 0.88rem;'>🚀 Quick Actions</div>", unsafe_allow_html=True)
    qa_cols = st.columns(6)
    for idx, (label, query_text) in enumerate(quick_actions.items()):
        with qa_cols[idx % 6]:
            if st.button(label, key=f"qa_btn_{idx}", use_container_width=True):
                triggered_query = query_text

user_question = st.chat_input("Ask about exams, assignments, deadlines, or upcoming work...")

# Determine query to process (either from Quick Action button or Chat Input)
query_to_process = triggered_query or (user_question.strip() if user_question else None)

if query_to_process:
    clean_question = query_to_process.strip()

    # 1. Store user question and render in chat UI
    st.session_state.chat_messages.append({"role": "user", "content": clean_question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(clean_question)

    # 2. Check if deadline data exists
    if not has_extracted_data:
        no_data_reply = "Please upload and analyze an academic document first so I can answer questions about your deadlines."
        with st.chat_message("assistant", avatar="🤖"):
            st.info(no_data_reply)
        st.session_state.chat_messages.append({"role": "assistant", "content": no_data_reply})
    else:
        # 3. FAST-PATH LOCAL ENGINE: Answer directly from extracted_deadlines without Gemini API
        handled, local_reply = answer_local_deadline_query(clean_question, st.session_state.extracted_deadlines)

        if handled and local_reply:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(local_reply)
            st.session_state.chat_messages.append({"role": "assistant", "content": local_reply})
        else:
            # 4. Fallback to Gemini API ONLY for complex generative reasoning (e.g. study strategies)
            api_key = get_gemini_api_key()
            if not api_key:
                err_reply = "⚠️ Gemini API Key is missing. Please configure `GEMINI_API_KEY` in `.streamlit/secrets.toml` or set it in Sidebar Settings."
                with st.chat_message("assistant", avatar="🤖"):
                    st.error(err_reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": err_reply})
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("🤖 Thinking..."):
                        deadlines_text_context = format_deadlines_for_context(st.session_state.extracted_deadlines)
                        client = genai.Client(api_key=api_key)

                        past_history = st.session_state.chat_messages[:-1]

                        success, reply_text, err_detail = call_gemini_chat_with_retry(
                            client=client,
                            deadlines_text_context=deadlines_text_context,
                            user_query=clean_question,
                            chat_history=past_history,
                        )

                        if success and reply_text:
                            st.markdown(reply_text)
                            st.session_state.chat_messages.append({"role": "assistant", "content": reply_text})
                        else:
                            if is_api_key_error(err_detail):
                                fail_msg = "⚠️ Invalid Gemini API Key. Please get a valid key from https://aistudio.google.com/ and update Sidebar Settings."
                            else:
                                fail_msg = "⚠️ Gemini is temporarily unavailable, but I can still answer questions about your extracted deadlines."
                            st.warning(fail_msg)
                            st.session_state.chat_messages.append({"role": "assistant", "content": fail_msg})

st.markdown("---")

# ==============================================================================
# 5. DEADLINE SUMMARY & GMAIL SMTP SECTION
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
        st.warning("⚠️ No deadline summary is available yet. Please analyze an academic document first.")
    elif not st.session_state.user_email or "@" not in st.session_state.user_email.strip():
        st.warning("⚠️ Please enter your email address in Settings.")
    else:
        sender_email, app_password = get_email_secrets()
        if not sender_email or not app_password:
            st.error("❌ Sender email is not configured in Streamlit secrets.")
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
# 6. FOOTER
# ==============================================================================
st.markdown(
    """
    <div class="footer-text">
        <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">Built with Python • Streamlit • Google Gemini</p>
        <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.8rem;">AI-powered academic deadline management</p>
    </div>
    """,
    unsafe_allow_html=True,
)
