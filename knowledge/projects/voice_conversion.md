## AI Voice Conversion Studio

### What it is
AI Voice Conversion Studio is a full-stack web application that transforms the vocals of a song into a different voice using AI-based voice conversion. Users can upload an audio file and receive a converted version where the vocals are replaced while the instrumental track is preserved.

### Why I built it
I wanted hands-on experience building an end-to-end AI-powered media application and explore how modern voice conversion models can be integrated into a real, user-facing system. The project was also a way to learn how to manage complex ML pipelines in a production-style setup.

### Architecture
The frontend is built with React and provides a clean, responsive interface for uploading audio and interacting with the system. The backend is implemented in Python using Flask and manages audio processing, model inference, and file orchestration. AI components are built using PyTorch, with Spleeter used for vocal and instrumental separation and Retrieval-Based Voice Conversion (RVC) models for voice transformation.

### How it works
Users upload an audio file in mp3 or wav format. The backend first separates vocals and instrumentals using Spleeter. The vocal track is then converted into a target voice using an RVC-based deep learning model. Finally, the converted vocals are merged back with the original instrumental track to generate the final output.

### Key Technical Details
The system uses audio processing libraries such as Librosa and SoundFile and runs within a managed Conda environment. For testing and experimentation, a custom voice model was trained using the built-in training pipeline to evaluate conversion quality and model behavior.

### Challenges
Major challenges included handling large model inference efficiently, managing audio quality across multiple processing steps, and keeping end-to-end latency low enough for a reasonable user experience.

### What I learned
This project deepened my understanding of real-time ML pipelines, audio processing workflows, and the practical challenges of integrating AI inference into a full-stack application. It also reinforced the importance of making complex AI systems intuitive for end users.

### What I’d improve today
I would optimize inference speed further, improve resource management for concurrent requests, and add monitoring around model performance and audio quality.
