## Projects

### ATS Resume Checker Chrome Extension

I built an ATS resume checker as a Chrome extension to remove friction from the job application process.

The motivation for this project came from repeatedly having to search for ATS tools, sign up, upload resumes, and paste job descriptions just to get a score. I wanted a faster and more practical solution.

What the project does:
- Pulls the job description directly from the job posting page the user is viewing
- Compares the job description against the user’s resume
- Generates a deterministic, ATS-style compatibility score
- Avoids random or hallucinated AI output by using rule-based and deterministic scoring
- Runs fully locally so no resume or job data leaves the user’s machine

Technical details:
- Chrome Extension built using Manifest V3
- Backend built with Node.js and Express
- Local LLM inference via Ollama
- Fully Dockerized backend for easy setup and reproducibility
- Version control using Git and GitHub

This project taught me how to design systems around AI responsibly, focusing on reliability and user trust instead of blindly relying on generative output.

---

### AI Voice Conversion Studio

I built a full-stack AI music web application that converts vocals from one voice to another using deep learning.

The application allows users to upload a song and receive an AI-generated remix with the vocals transformed into a different voice.

What the application does:
- Accepts audio uploads in mp3 or wav format
- Separates vocals and instrumentals using Spleeter
- Converts the vocal track into a new voice using Retrieval-Based Voice Conversion (RVC)
- Merges the converted vocals back with the instrumental track
- Produces the final AI-generated audio within seconds

Technical details:
- Frontend built with React.js
- Backend implemented using Flask and Python
- Machine learning models built with PyTorch
- Audio processing using Librosa and SoundFile
- Model training and experimentation using Conda environments

For testing, I trained a custom AI voice model using the built-in training pipeline, which produced highly accurate results.

This project gave me hands-on experience building real-time ML pipelines and integrating large AI models into a production-ready full-stack application.

---

### Minecraft Manhunt Plugin (Java)

I built a custom Minecraft Manhunt plugin inspired by Dream’s Manhunt gameplay.

The plugin recreates the Manhunt experience where one player acts as the speedrunner while others hunt them using tracking compasses across different dimensions.

What the plugin includes:
- Compass-based player tracking in both the Overworld and the Nether
- Automatic compass re-assignment when a hunter dies
- A custom `/startmanhunt` command to initialize the game

Technical details:
- Implemented in Java
- Built using the Spigot API
- Developed and tested using Eclipse and a local Minecraft server
- Compatible with Minecraft version 1.16.1

This project helped me understand event-driven programming, game mechanics, and plugin development, and gave me a deeper appreciation for how complex game systems are built.
