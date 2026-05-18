## Chess AI Web Application (WhiteToMove)

### What it is
WhiteToMove is a web-based chess application that allows users to play chess either against another human or against an AI opponent. The application is fully functional and deployed, with a clean and modern user interface.

### Why I built it
I play chess regularly and wanted to challenge myself to see if I could build a complete, working chess application from scratch in a single day. What started as a personal experiment turned into a polished, production-ready project.

### Architecture
The frontend is built using React and handles the chessboard UI, move interactions, and game state visualization. The core chess logic is implemented manually rather than relying on an external chess library. A lightweight Stockfish integration is used for AI gameplay, with careful synchronization between the UI state and the engine responses.

### Features
The application supports human vs human gameplay and human vs AI gameplay. It enforces full chess rules including legal move validation, castling, check, checkmate, and stalemate. Pawn promotion is supported with user choice between queen, rook, bishop, and knight. The app also tracks move history and captured pieces.

### Challenges
The most challenging part was implementing correct chess rules and ensuring consistency between the UI, the internal game state, and the AI engine. Handling pawn promotion while keeping the UI and AI in sync required careful state management.

### What I learned
This project strengthened my understanding of state-heavy frontend applications, game logic implementation, and synchronizing external engines with a UI. It also reinforced the value of building projects around personal interests to stay motivated and learn faster.

### What I’d improve today
I would further optimize performance, improve AI difficulty tuning, and add features such as online multiplayer and stronger engine configuration controls.
