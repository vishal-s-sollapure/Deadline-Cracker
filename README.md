# 🎓 Deadline Tracker — AI Academic Deadline Assistant

A streamlined, intelligent Streamlit application designed for students and researchers to track academic deadlines, assignments, exams, projects, and key academic dates using **Google Gemini Vision AI** and **Gmail SMTP**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Vision%20AI-orange.svg)

---

## 🌟 Core Features

1. **📸 Vision-Based Deadline Extraction**: Upload syllabi, assignment prompt sheets, exam timetables, or notice board pictures (`.png`, `.jpg`, `.jpeg`, `.webp`). Gemini Vision analyzes the visual text and extracts structured deadline data.
2. **🛡️ Zero-Hallucination Guardrails**: Prompts explicitly prevent Gemini from inventing or guessing missing dates or information. Missing fields are safely flagged as `"Not Specified"`.
3. **📊 Visual Deadline Dashboard**: View extracted deadlines organized by course, task name, due date, type (Assignment, Exam, Project, Submission), and specific instructions.
4. **💬 Context-Aware AI Chat**: Ask follow-up questions (e.g. *"What is my due date for CS101?"*, *"Which project should I prioritize?"*) with Gemini answering directly grounded in your document.
5. **✉️ Email Summary Dispatcher**: Generate a structured deadline summary and send it directly to your email address via Gmail SMTP.

---

## 📁 Repository Structure

```text
deadline-tracker/
├── app.py                          # Main Streamlit web application
├── prompts.py                      # Centralized system prompts & AI guardrails
├── requirements.txt                # Required Python dependencies
├── README.md                       # Comprehensive setup and deployment guide
├── .gitignore                      # Excludes secrets and temporary runtime files
└── .streamlit/
    └── secrets.toml.example        # Configuration template for API keys & SMTP secrets
```

---

## ⚙️ Local Setup Instructions

### 1. Prerequisites
- Python 3.9 or higher
- A Google Gemini API Key ([Get one here](https://aistudio.google.com/))
- A Gmail account with 2-Step Verification enabled to generate a **Gmail App Password**.

### 2. Installation steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/deadline-tracker.git
cd deadline-tracker

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 3. Setup Secrets Configuration

Copy the sample secrets template to create your local `.streamlit/secrets.toml`:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and fill in your credentials:

```toml
GEMINI_API_KEY = "your_google_gemini_api_key_here"

SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_16_digit_app_password"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
```

> 🔒 **Security Note**: Never commit `.streamlit/secrets.toml` to GitHub. It is excluded via `.gitignore`.

---

## 🔑 How to Generate a Gmail App Password

For email sending via Gmail SMTP:
1. Go to your **Google Account Settings** (`myaccount.google.com`).
2. Navigate to **Security** and ensure **2-Step Verification** is turned ON.
3. Search for **App Passwords** in the top search bar.
4. Create a new App Password (name it e.g. `Deadline Tracker App`).
5. Copy the generated **16-character code** (without spaces) into `SENDER_PASSWORD` in `.streamlit/secrets.toml` or the app's sidebar.

---

## 🚀 Running the Application

Launch the Streamlit development server:

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push your repository to **GitHub**.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"**, select your repository, branch (`main`), and set Main file path to `app.py`.
4. Click **"Advanced Settings"** or go to your app settings > **Secrets**.
5. Paste the contents of your `.streamlit/secrets.toml` (with real values) into the Streamlit Cloud Secrets Manager.
6. Click **Deploy!** 🚀

---

## 🛠️ Technology Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **AI Model**: [Google Gemini API](https://aistudio.google.com/) (`google-genai` / `gemini-2.5-flash`)
- **Vision & Image Processing**: [Pillow (PIL)](https://python-pillow.org/)
- **Email Protocol**: `smtplib` & `email.mime` (Python Standard Library)

---

## 📜 License

This project is open-source and intended for academic demonstration and learning purposes.
