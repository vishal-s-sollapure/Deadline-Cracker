# 🎓 Deadline Tracker — AI Academic Deadline Assistant

A modern, intelligent Streamlit application designed for students and researchers to track academic deadlines, assignments, exams, projects, and key academic dates using **Google Gemini Vision AI** and **Gmail SMTP**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI-orange.svg)

---

## 🌟 Core Features

1. **📄 Academic Document & PDF Analysis**: Upload syllabi, assignment sheets, timetables, exam schedules, or notices (`PNG`, `JPG`, `JPEG`, `WEBP`, `PDF`). Gemini analyzes document content and extracts structured deadlines.
2. **🛡️ Zero-Hallucination Guardrails & Local Query Engine**: Fast local lookup engine processes all deadline lookups (exams, assignments, date windows, subject search, next deadline) 100% locally in Python without consuming API quota or hallucinating details.
3. **📊 Smart Deadline Dashboard**: Displays 5 top metric cards (`📌 Total Deadlines`, `🔴 Overdue`, `🟠 Due Soon`, `📚 Assignments & Projects`, `📝 Exams`), interactive search, category/subject/status filters, and chronological table view.
4. **⚡ Next Deadline Highlight**: Automatically detects and highlights the earliest upcoming chronological deadline with smart status badges (`🔴 Overdue`, `🟠 Due Today`, `🟡 Due Soon`, `🟢 Upcoming`).
5. **💬 Interactive AI Deadline Chat**: Natural language chat grounded strictly in your extracted deadline data with 6 quick-action buttons.
6. **✉️ Gmail Summary Dispatcher**: Generates formatted deadline summaries and sends them directly to your email via Gmail SMTP.

---

## 📁 Repository Structure

```text
deadline-tracker/
├── app.py                          # Main Streamlit web application
├── prompts.py                      # System prompts & AI guardrails
├── requirements.txt                # Required Python dependencies
├── README.md                       # Setup and deployment guide
├── .gitignore                      # Excludes secrets & temporary files
└── .streamlit/
    └── secrets.toml.example        # Configuration template for API keys & SMTP secrets
```

---

## ⚙️ Local Setup Instructions

### 1. Prerequisites
- Python 3.9 or higher
- A Google Gemini API Key ([Get one here](https://aistudio.google.com/))
- A Gmail account with 2-Step Verification enabled to generate a **Gmail App Password**.

### 2. Installation Steps

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

EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_APP_PASSWORD = "your_16_digit_app_password"

# Optional Aliases
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
5. Copy the generated **16-character code** (without spaces) into `EMAIL_APP_PASSWORD` in `.streamlit/secrets.toml` or the app's sidebar.

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
4. Click **"Advanced Settings"** > **Secrets**.
5. Paste the contents of your `.streamlit/secrets.toml` (with real values) into the Secrets Manager.
6. Click **Deploy!** 🚀

---

## 🛠️ Technology Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **AI Engine**: [Google Gemini API](https://aistudio.google.com/) (`google-genai` / `gemini-3.8-flash`)
- **Vision & Document Handling**: Pillow (PIL) & PDF Stream Bytes
- **Email Protocol**: `smtplib` & `email.mime` (Python Standard Library)

---

## 🔗 Live Demo & Links

- **Live App**: *[Streamlit Community Cloud Link Placeholder]*
- **GitHub Repository**: *[GitHub Repository Link Placeholder]*

---

## 📜 License

This project is open-source and intended for portfolio demonstration and academic learning purposes.
