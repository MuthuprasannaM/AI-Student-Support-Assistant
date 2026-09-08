
# 🎓 AI Student Support Assistant

An intelligent, context-aware college support system built using **LangGraph**, **Ollama (Llama 3.2)**, and **FAISS**. The assistant handles student queries regarding academic regulations, syllabi, fee policies, exam timetables, and attendance calculations with multi-turn conversational memory.

---

## 🚀 Key Features

* **RAG (Retrieval-Augmented Generation)**: Queries institutional documents, regulations, and FAQs stored in local vector storage (FAISS).
* **Dynamic Tool Calling**:
  * `lookup_college_knowledge`: Retrieves context from institutional text/PDF documents.
  * `check_exam_schedule`: Fetches department-specific upcoming exam schedules.
  * `calculate_attendance_status`: Computes attendance percentage and informs how many classes can be safely skipped or must be attended to meet requirements (e.g., 75%).
  * `escalate_to_helpdesk`: Generates an administrative support ticket when an inquiry falls outside the documents.
* **Conversational Memory**: Built-in state management using LangGraph's `MemorySaver` to track student details across turns.
* **100% Local Execution**: Runs privately using local LLMs and embeddings via Ollama.

---

## 🛠️ Tech Stack

* **Orchestration**: LangGraph, LangChain Core
* **LLM**: Ollama (`llama3.2`)
* **Embeddings**: Ollama (`nomic-embed-text`)
* **Vector Store**: FAISS (Facebook AI Similarity Search)
* **User Interface**: Streamlit / CLI

---

## 📂 Project Structure

```text
AI-Student-Support-Assistant/
├── documents/
│   ├── regulations.txt
│   └── notices_and_faqs.txt
├── .gitignore
├── app.py              # Streamlit Web Application
├── main.py             # LangGraph workflow, tools, and CLI runner
└── README.md

```

---

## ⚙️ Installation & Setup

### 1. Prerequisites

Install [Ollama](https://ollama.com/) and download the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text

```

### 2. Clone the Repository

```bash
git clone [https://github.com/MuthuprasannaM/AI-Student-Support-Assistant.git](https://github.com/MuthuprasannaM/AI-Student-Support-Assistant.git)
cd AI-Student-Support-Assistant

```

### 3. Set Up Virtual Environment & Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required packages
pip install langgraph langchain-ollama langchain-community langchain-core faiss-cpu typing-extensions streamlit

```

---

## 💻 How to Run

### Option A: Run in Terminal (CLI Mode)

```bash
python main.py

```

### Option B: Run with Streamlit Web UI

```bash
streamlit run app.py

```

---

## 🧪 Example Test Prompts

1. **Memory**: `"Hi, my name is Alex and I'm a Computer Science student."`
2. **Policy RAG**: `"What is the minimum attendance required for final exams?"`
3. **Dynamic Schedule**: `"When are my upcoming exams?"`
4. **Attendance Calculation**: `"I have attended 28 out of 45 classes in Data Structures. Can I sit for the exam?"`


