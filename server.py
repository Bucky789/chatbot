from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import os
import re
import uuid

app = FastAPI()

LLAMA_PATH = "/data/data/com.termux/files/home/llama.cpp/build/bin/llama-cli"
MODEL_PATH = "/data/data/com.termux/files/home/llama.cpp/models/phi-2.Q4_K_M.gguf"

BASE_CMD = [
    LLAMA_PATH,
    "-m", MODEL_PATH,
    "-t", "4",
    "-c", "1024",
    "--temp", "0.2",
    "-n", "80",
    "--simple-io"
]

KNOWLEDGE_DIR = "knowledge"

SYSTEM_PROMPT = """You are Manthan Sumbhe.
Answer ONLY using the provided information.
If the information is not present, say you do not know.
Answer professionally and in first person.
"""

class Query(BaseModel):
    question: str


def load_docs():
    docs = []
    for f in os.listdir(KNOWLEDGE_DIR):
        if f.endswith(".md"):
            with open(os.path.join(KNOWLEDGE_DIR, f)) as file:
                docs.append(file.read())
    return docs


def chunk(text, size=500):
    chunks, cur = [], ""
    for line in text.splitlines():
        if len(cur) + len(line) > size:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur:
        chunks.append(cur)
    return chunks


def score(chunk, query):
    q = set(re.findall(r"\w+", query.lower()))
    c = set(re.findall(r"\w+", chunk.lower()))
    return len(q & c)


DOCS = []
for d in load_docs():
    DOCS.extend(chunk(d))

@app.get("/health")
def health():
    print("HEALTH HIT")
    return {"status": "ok"}
    
def get_context(question):
    ranked = sorted(
        [(score(c, question), c) for c in DOCS],
        reverse=True
    )
    return "\n---\n".join([c for s, c in ranked if s > 0][:3])


@app.post("/chat")
def chat(q: Query):
    print("CHAT HIT WITH:", q.question)

    prompt = f"""{SYSTEM_PROMPT}

Information:
{get_context(q.question)}

Question: {q.question}
Answer:
"""

    proc = subprocess.Popen(
        [
            LLAMA_PATH,
            "-m", MODEL_PATH,
            "-t", "4",
            "-c", "1024",
            "--temp", "0.2",
            "-n", "80",
            "-p", prompt,
            "--simple-io"
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,   # 🔑 THIS IS THE FIX
        text=True
    )

    stdout, stderr = proc.communicate()

    if stderr:
        print("LLAMA STDERR:", stderr)

    answer = (stdout or "").strip()

    if not answer:
        answer = "I don't have enough information to answer that."

    return {"answer": answer}

