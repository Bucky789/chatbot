from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import os
import re

app = FastAPI()

LLAMA_PATH = "/data/data/com.termux/files/home/llama.cpp/build/bin/llama-cli"
MODEL_PATH = "/data/data/com.termux/files/home/llama.cpp/models/phi-2.Q4_K_M.gguf"

BASE_CMD = [
    LLAMA_PATH,
    "-m", MODEL_PATH,
    "-t", "4",
    "-c", "1024",
    "--temp", "0.2",
    "--n-predict", "80",
    "--no-display-prompt"
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


@app.post("/chat")
def chat(q: Query):
    ranked = sorted(
        [(score(c, q.question), c) for c in DOCS],
        reverse=True
    )
    context = "\n---\n".join([c for s, c in ranked if s > 0][:3])

    prompt = f"""{SYSTEM_PROMPT}

Information:
{context}

Question: {q.question}
Answer:
"""

    result = subprocess.run(
        BASE_CMD + ["-p", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print("STDERR:", result.stderr)  # debug
    print("STDOUT:", result.stdout)  # debug

    answer = result.stdout.strip()

    if not answer:
        answer = "I don't have enough information to answer that."

    return {"answer": answer}
