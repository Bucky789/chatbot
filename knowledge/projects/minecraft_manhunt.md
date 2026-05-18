## Minecraft Manhunt Plugin

### What it is
A custom Minecraft Manhunt plugin built in Java that recreates the popular Manhunt game mode, where one player acts as the speedrunner and the remaining players act as hunters. The plugin supports real-time compass tracking across dimensions and manages game state automatically.

### Why I built it
As a long-time Minecraft player, I wanted to understand how game mechanics work under the hood and challenge myself to build a real plugin rather than just play mods. This project was inspired by the Manhunt game mode and served as a hands-on way to learn event-driven systems and game server development.

### Architecture
The plugin is built in Java using the Spigot API and runs on a local Minecraft server (version 1.16.1). It uses event listeners to track player actions such as movement, deaths, and dimension changes. Game state is managed internally to coordinate roles, compass updates, and command handling.

### Features
The plugin includes a compass that tracks the speedrunner’s location in real time across both the Overworld and the Nether. Hunters automatically receive a new tracking compass when they die. A custom `/startmanhunt` command initializes the game and assigns roles.

### Challenges
One of the main challenges was implementing reliable compass tracking across different dimensions while keeping updates accurate and performant. Handling edge cases such as player deaths and dimension transitions required careful event handling and state management.

### What I learned
This project taught me how to work with event-driven architectures, design real-time multiplayer logic, and write defensive code for unpredictable player behavior. It also gave me practical experience with Java-based plugin development and server-side debugging.

### What I’d improve today
I would add better configurability, support for additional dimensions, and cleaner abstractions for game state management to make the plugin easier to extend.
