# AI Avatar Agent - Backend

This is the Python backend for the **AI Avatar Agent Made by Aly Omer**. It uses LiveKit for real-time communication and Beyond Presence for the AI avatar.

## Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended for package management)

## Setup

1.  **Clone the repository** (if you haven't already).

2.  **Install dependencies**:
    ```bash
    uv sync
    ```
    Or with pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables**:
    Create a `.env.local` file in this directory with the following keys:
    ```env
    LIVEKIT_URL=...
    LIVEKIT_API_KEY=...
    LIVEKIT_API_SECRET=...
    BEY_API_KEY=...
    GOOGLE_API_KEY=...
    ```

## Running the Agent

To start the agent server:

```bash
python agent.py dev
```

The agent will connect to your LiveKit project and wait for a user to join a room.
