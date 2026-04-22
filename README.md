## ITC 2026 AI Agent

This is a submission for the ITC 2026 competition under the AI category

This AI is a chatbot for the Cal Poly Pomona (CPP) website where users can ask it questions and it'll respond with information from the supplied corpus

## Repository Setup

The code is split into 3 files:

``server.ts`` -> Express.js server handling the website

``agent.ts`` -> LLM Handler, handles the backend processing of messages with Claude

``search_corpus.ts`` -> Search functionality for efficiently searching through the corpus for relevant information

**search_corpus on first query first generates a cache file, this is a one time thing (file is too big for Github). This is why on first message it will take longer than usual**

## Setup Instructions

# Step 1 - Git and Runtime Setup

Download the [Git CLI](https://git-scm.com/install/) and the [Node.js runtime](https://nodejs.org/en)

**When installing the Node.js runtime, it is CRITICAL that you install the npm package manager in addition to the runtime**

# Step 2 - Download Repository

Open the terminal and run the following command: ``git clone https://github.com/Grenrobotics/ITC-2026-AI-Project.git``

# Step 3 - Install Dependencies

Open the terminal, change the current directory to the downloaded repository, and run the command ``npm install``

# Step 4 - Configuration

Rename ``ENV_FORMAT.txt`` to ``.env`` and paste your Claude API key after the equal sign

# Step 5 - Execution

If on Windows, run ``npm run winStart``

If on Linux, run ``npm run linuxStart``

Navigate to localhost:80 to access the chatbot and then chat to your hearts content