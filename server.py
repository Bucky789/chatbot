from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from fastapi.security import APIKeyHeader
import os
import re
import requests

# =====================
# Security
# =====================

API_KEY = os.getenv("CHATBOT_API_KEY")
API_KEY_NAME = "X-API-Key"

if not API_KEY:
    raise RuntimeError("CHATBOT_API_KEY not set")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )

# =====================
# App
# =====================

app = FastAPI()

MODEL_SERVER_URL = "http://127.0.0.1:8081/v1/completions"
KNOWLEDGE_DIR = "knowledge"

# =====================
# System Prompt
# =====================

SYSTEM_PROMPT = """You are Manthan Sumbhe.

You must strictly follow the rules below.

GENERAL RULES:
- Answer using ONLY the provided information.
- Do NOT add assumptions or external knowledge.
- Do NOT combine information from different topics.

OUTPUT RULES:
- First person
- ONE paragraph only
- 2–4 complete sentences MAX
- No lists
- No disclaimers
- No meta statements
- Stop after answering the question

If the required information is not present, respond exactly with:
"I do not know."
"""

IDENTITY_CONTEXT = """My name is Manthan Sumbhe.
I am a Master’s student in Computer Science.
I am based in the United States.
"""

# =====================
# Models
# =====================

class Query(BaseModel):
    question: str

# =====================
# Question Routing
# =====================
import random

SMALLTALK_RESPONSES = [
    "Hello! You can ask me about my projects, skills, or background.",
    "Hi there! Feel free to ask about my projects, technical skills, or experience.",
    "Hey! I can answer questions about my work, projects, or technical background.",
    "Hello! Ask me anything about my projects, skills, or what I’ve been building.",
    "Hi! I’m here to talk about my projects, engineering experience, and background."
]
def is_smalltalk(q: str) -> bool:
    return q.lower().strip() in {"hi", "hello", "hey"}

def is_identity_question(q: str) -> bool:
    q = q.lower()
    return any(p in q for p in [
        "who are you", "about you", "tell me about yourself","where is he based","where do you live","your location","where are you from","where do you stay","where are you based",
        "where are you", "where do you live", "your location","where are you from","where do you stay","where are you based","Manthan","who is manthan","who is manthan sumbhe",
    ])

def is_education_question(q: str) -> bool:
    q = q.lower()
    return any(p in q for p in [
        "studying", "education", "degree",
        "graduate", "graduation","graduating","what are you studying","what is your education background","what degree are you pursuing","when are you graduating"
    ])
    
def is_experience_question(q: str) -> bool:
    q = q.lower()
    return any(p in q for p in [
        "experience", "work experience", "what is his experience","what work experience he has","microservices","distributed systems","backend systems","full-stack systems","what experience do you have","what is your experience"
    ])

def is_project_question(q: str) -> bool:
    q = q.lower()
    return any(p in q for p in [
        "project", "projects", "built", "worked on","what projects have you worked on","what have you built","what projects have you built","what is your most impressive project","what is your most significant project",
        "any projects","tell me about your projects","describe your projects","what is your most impressive project","what is your most significant project",
        "chess", "ats", "voice conversion", "minecraft","chess ai","chess application","ats resume checker","ai voice conversion","minecraft manhunt plugin",
    ])

def is_skill_question(q: str) -> bool:
    q = q.lower()
    return any(p in q for p in [
        "skills", "technologies", "tech stack","DSA","data structures and algorithms","programming experience","System design","system architecture","Does he know system design","does he know system architecture",
        "languages", "frameworks","databases","tools","what skills do you have","what technologies do you know","what is your tech stack","databases you have used","tools you have used",
        "cloud experience","does he have cloud experience","devops","ci/cd","cloud platforms","cloud services","ai tools","machine learning frameworks","programming languages"
    ])


def is_broad_skill_question(q: str) -> bool:
    q = q.lower()
    return any(p in q for p in [
        "skills", "your skills", "tech stack"
    ])

def is_roles_and_preferences_question(q: str) -> bool:
    q = q.lower()
    return any(p in q for p in [
        "roles", "role", "preferences", "what roles do you prefer", "what role do you prefer","what type of work do you prefer","what type of roles do you prefer",
        "relocation","are you open to relocation","are you open to remote work","remote work","relocate","is he open to relocation","is he open to remote work","is he open to relocating","are you open to relocating",
        "what roles is he interested in","what role is he interested in","what type of work is he interested in","what type of roles is he interested in", "what roles is he looking for","what role is he looking for","what type of work is he looking for","what type of roles is he looking for"
    ])

def is_work_authorization_question(q: str) -> bool:
    q = q.lower()
    return any(p in q for p in [
    "work authorization", "work visa", "visa status", "work permit", "employment authorization","will he require visa sponsorship","do you require visa sponsorship","does he require visa sponsorship","will you require visa sponsorship"
    ])

# =====================
# Knowledge Loading
# =====================

def chunk_by_headers(text: str):
    chunks = []
    current = ""

    for line in text.splitlines():
        if line.startswith("## ") or line.startswith("### "):
            if current.strip():
                chunks.append(current.strip())
            current = line + "\n"
        else:
            current += line + "\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks

def load_docs_by_name(filename: str):
    path = os.path.join(KNOWLEDGE_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return chunk_by_headers(f.read())
    
def is_chess_question(q: str) -> bool:
    q = q.lower()
    return any(p in q for p in [
        "skills", "your skills", "tech stack"
    ])

def load_project_docs():
    project_dir = os.path.join(KNOWLEDGE_DIR, "projects")
    chunks = []

    if not os.path.isdir(project_dir):
        return chunks

    for f in os.listdir(project_dir):
        if f.endswith(".md"):
            with open(os.path.join(project_dir, f)) as file:
                chunks.extend(chunk_by_headers(file.read()))

    return chunks

def score(chunk: str, query: str) -> int:
    q_words = re.findall(r"\w+", query.lower())
    c_words = re.findall(r"\w+", chunk.lower())

    score = 0
    for q in q_words:
        for c in c_words:
            if q in c or c in q:  # simple partial match
                score += 1
                break

    return score

def get_context_with_confidence(question: str, docs):
    scored = [(score(c, question), c) for c in docs]
    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored or scored[0][0] < 0.5:
        return None

    top_chunks = [chunk for s, chunk in scored if s > 0][:5]
    return "\n\n---\n\n".join(top_chunks)


# =====================
# Summaries
# =====================

PROJECT_SUMMARY = """I have worked on multiple projects including a chess AI web application, an ATS resume checker, an AI voice conversion system, and a Minecraft Manhunt plugin."""

SKILLS_SUMMARY = """I primarily work with Java and Python and have experience building backend and full-stack systems using frameworks like Spring Boot and React, along with SQL databases and Docker."""

# =====================
# Cache
# =====================

CACHE = {}

# =====================
# Routes
# =====================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(q: Query, _: str = Depends(verify_api_key)):
    question = q.question.strip()
    print("Chat hit with question:", question)
    key = question.lower()

    if key in CACHE:
        return {"answer": CACHE[key]}
    question = re.sub(r"[^\w\s]", "", question.lower()).strip()
    # ✅ Smalltalk (deterministic)
    if is_smalltalk(question):
        response = random.choice(SMALLTALK_RESPONSES)
        CACHE[question] = response
        return {"answer": response}

    # Identity
    if is_identity_question(question):
        context = IDENTITY_CONTEXT

    # Education
    elif is_education_question(question):
        docs = load_docs_by_name("education.md")
        context = get_context_with_confidence(question, docs)
        if context is None:
            print("No context found for education question")
            return {"answer": "I do not know."}
    
    # Skills
    elif is_skill_question(question):
        docs = load_docs_by_name("skills.md")
        context = get_context_with_confidence(question, docs)

        if context is None and is_broad_skill_question(question):
            print("broad skill question")
            context = SKILLS_SUMMARY

        if context is None:
            print("No context found for skill question")
            return {"answer": "I  not know."}
    
    # Experience
    elif is_experience_question(question):
        docs = load_docs_by_name("experience.md")
        context = get_context_with_confidence(question, docs)
        if context is None:
            print("No context found for experience question")
            return {"answer": "I do not know."}

    # Projects
    elif is_project_question(question):
        print("Project question detected")
        docs = load_docs_by_name("projects.md")
        context = get_context_with_confidence(question, docs)
        print(context)

        if context is None:
            print("No context found for project question")
            return {"answer": "I do not know."}
    
    # Roles and Preferences
    elif is_roles_and_preferences_question(question):
        docs = load_docs_by_name("roles_and_preferences.md")
        context = get_context_with_confidence(question, docs)

        if context is None:
            print("No context found for roles and preferences question")
            return {"answer": "I do not know."}
    
    # Work Authorization
    elif is_work_authorization_question(question):
        docs = load_docs_by_name("work_authorization.md")
        context = get_context_with_confidence(question, docs)

        if context is None:
            print("No context found for work authorization question")
            return {"answer": "I do not know."}

    else:
        print("Question did not match any category")
        return {"answer": "I do not know."}

    prompt = f"""{SYSTEM_PROMPT}

Information:
{context}

Question: {question}
Answer:
<END>
"""

    payload = {
        "prompt": prompt,
        "n_predict": 90,
        "temperature": 0.2,
        "stop": ["<END>", "\n\n", "Question:"],
    }

    try:
        resp = requests.post(MODEL_SERVER_URL, json=payload, timeout=120)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["text"].strip()
    except Exception:
        print("Error calling model server:", resp.text if 'resp' in locals() else "No response")
        return {"answer": "I do not know."}

    answer = answer.split("\n")[0].strip()
    if not answer:
        print("Model returned empty answer")
        answer = "I do not know."

    CACHE[key] = answer
    print(answer)
    return {"answer": answer}
