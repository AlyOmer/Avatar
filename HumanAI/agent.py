import logging
import os
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    Agent,
)

from livekit.plugins import (
    bey,
    google,
    noise_cancellation,
)

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bey-avatar-playground")

# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------
load_dotenv(".env.local")

BEY_API_KEY = os.getenv("BEY_API_KEY")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")

if not BEY_API_KEY:
    raise RuntimeError("❌ BEY_API_KEY not set")
if not LIVEKIT_URL:
    raise RuntimeError("❌ LIVEKIT_URL not set")

# -------------------------------------------------------------------
# Agent Server
# -------------------------------------------------------------------
server = AgentServer()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice AI assistant."
        )


# -------------------------------------------------------------------
# RTC Session Entry Point
# -------------------------------------------------------------------
@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    logger.info("🚀 Starting LiveKit BEY avatar session")

    # ---------------------------------------------------------------
    # 1️⃣ Create Agent Session (LLM / Voice)
    # ---------------------------------------------------------------
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            voice="Puck",
            temperature=0.7,
        )
    )

    # ---------------------------------------------------------------
    # 2️⃣ START AGENT SESSION FIRST (CRITICAL)
    # ---------------------------------------------------------------
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=agents.room_io.RoomOptions(
            audio_input=agents.room_io.AudioInputOptions(
                noise_cancellation=lambda params:
                noise_cancellation.BVCTelephony()
                if params.participant.kind
                == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            )
        ),
    )

    logger.info("✅ Agent session started")

    # ---------------------------------------------------------------
    # 3️⃣ Start BEY Avatar AFTER session start
    # ---------------------------------------------------------------
    avatar_id = "b9be11b8-89fb-4227-8f86-4a881393cbdb"

    avatar = bey.AvatarSession(
        avatar_id=avatar_id,
    )

    await avatar.start(session, ctx.room )

    logger.info("🎥 BEY avatar joined room")

    # ---------------------------------------------------------------
    # 4️⃣ Initial Greeting (Triggers Audio + Lip Sync)
    # ---------------------------------------------------------------
    await session.generate_reply(
        instructions="Hello! I am your AI assistant. Can you see me?"
    )

    logger.info("🎉 Avatar session fully running")


# -------------------------------------------------------------------
# Run App
# -------------------------------------------------------------------
if __name__ == "__main__":
    agents.cli.run_app(server)
