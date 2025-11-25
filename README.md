# 🔐 Secure AI Task Manager
A local, encrypted, AI-assisted task manager built as a personal engineering project.

This started as a simple offline task tracker but quickly grew into a system that:
- stores tasks fully encrypted,
- uses an in-memory SQLite database,
- includes NLP + semantic search,
- optionally integrates with a *local* LLM through Ollama,
- and exposes everything through a Streamlit UI.

No cloud APIs, no network calls — everything stays on your machine.

---

## 🚀 Why I Built This
I wanted a project that combines:
- security  
- offline AI  
- real NLP  
- a UI and real user experience  
- modular, production-like architecture  

It’s part research project, part engineering challenge, and part demonstration of strong fundamentals.

---

## 🧠 Features

### 1. Fully Encrypted Local Storage
- SQLite database runs entirely in RAM  
- On exit, the data is saved as an encrypted dump (`tasks.enc`)  
- Encryption uses:  
  - **Fernet (AES-128)** for data  
  - **Windows Credential Manager (DPAPI)** for storing the key securely  
- No plaintext DB file ever touches disk

### 2. Optional Local LLM Support (Ollama)
If Ollama is installed, the app can use local models like:
- `mistral:instruct`
- `llama2`
- `codeqwen:7b`

Used for advanced subtask generation or future extensions.

### 3. TF-IDF Semantic Search
Search tasks by meaning, not keywords.  
Implemented using scikit-learn’s:
- TfidfVectorizer  
- Cosine similarity  

### 4. NLP Utilities (spaCy-based)
- Text normalization  
- Tokenization  
- Verb/noun extraction  
- Simple linguistic analysis  
- Lightweight summarizer  

### 5. Streamlit Dashboard
- Add / edit / delete tasks  
- Task table view  
- AI Insights panel  
- Semantic search  
- Subtask generator (LLM or rule-based)  
- Encrypted DB save button  
- Optional sample data loader (`tests/sample_tasks.py`)

---

## 📂 Project Structure

```
secure-ai-task-manager/
│
├── app.py                 # Streamlit UI
├── main.py                # DB load + save orchestration
├── database.py            # In-memory SQLite CRUD
├── encrypted_storage.py   # DPAPI + Fernet encryption
├── ai_utils.py            # NLP utilities, TF-IDF search, LLM subtasks
│
├── tests/
│   └── sample_tasks.py    # Example tasks for testing/demo
│
├── data/
│   └── tasks.enc          # Encrypted SQLite dump (auto-created)
│
├── requirements.txt
└── README.md
```

---

## 🛠 Installation

Clone the repo:

```
git clone https://github.com/<your-username>/secure-ai-task-manager.git
cd secure-ai-task-manager
```

Create virtual environment:

```
python -m venv venv
```

Activate it:

Windows:
```
venv\Scripts\activate
```

macOS/Linux:
```
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Optional (if you want LLM features):

Install Ollama from:
```
https://ollama.ai
```

Then pull a model:
```
ollama pull mistral:instruct
```

---

## ▶️ Running the App

Initialize the encrypted, in-memory DB:

```
python main.py
```

Launch the Streamlit UI:

```
streamlit run app.py
```

Your browser will open automatically.

---

## 🧪 Sample Data

To load pre-made tasks for demo/testing:

```
python tests/sample_tasks.py
```

This writes sample tasks into the encrypted DB so they appear automatically when the UI runs.

---

## 🧱 Tech Stack

- Python 3.10+
- Streamlit (UI)
- SQLite (in-memory)
- cryptography (Fernet)
- Windows Credential Manager (DPAPI)
- spaCy
- scikit-learn
- Ollama (optional, for local LLMs)

---

## 🚧 Future Improvements

- Improve JSON reliability for LLM subtasks  
- Add automatic JSON repair  
- Add task category classifier  
- Add risk scoring and priority engine  
- Timeline / calendar view for due dates  
- Better multi-task linking (knowledge graph style)

---

