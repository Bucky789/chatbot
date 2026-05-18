## ATS Resume Checker Chrome Extension

### What it is
An ATS resume checker built as a Chrome extension that evaluates how well a resume matches the job description on the page a user is currently viewing. The tool is designed to be fast, local, and frictionless, without requiring account creation or manual uploads each time.

### Why I built it
While applying to jobs, I repeatedly encountered friction when using online ATS checkers, such as mandatory sign-ups, repeated uploads, and manual copying of job descriptions. I built this project to eliminate those steps and make ATS feedback instant and practical.

### Architecture
The system is implemented as a Chrome extension using Manifest V3, which extracts the job description directly from the active page. A Node.js and Express backend handles resume analysis and scoring. The backend runs locally and is dockerized for easy setup. A lightweight local LLM is used via Ollama to support analysis while keeping all data on the user’s machine.

### How it works
The extension automatically pulls the job description from the page the user is viewing. The resume content is analyzed against that description using ATS-style, deterministic scoring rather than relying on non-deterministic AI output. The system prioritizes transparency and repeatability in scoring.

### Key Design Decisions
The project deliberately avoids cloud-based processing so that no resume data leaves the user’s machine. AI is used as a supporting component rather than a black box, with clear system boundaries and deterministic logic guiding final scores.

### Challenges
Key challenges included reliably extracting job descriptions from different websites, designing scoring logic that felt realistic without being misleading, and integrating a local LLM efficiently without adding noticeable latency.

### What I learned
This project taught me how to design practical developer tools, reduce real user friction, and build AI-assisted systems without blindly trusting model outputs. It also strengthened my understanding of local AI deployment, Docker-based workflows, and Chrome extension development.

### What I’d improve today
I would improve robustness across more job board layouts, add better feedback explanations for scores, and continue refining the scoring heuristics based on real-world usage.
