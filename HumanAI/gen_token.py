import os
import asyncio
from livekit import api
from dotenv import load_dotenv

load_dotenv(".env.local")

async def generate_token():
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    token = api.AccessToken(api_key, api_secret) \
        .with_identity("test_user") \
        .with_name("Test User") \
        .with_grants(api.VideoGrants(
            room_join=True,
            room="test-room",
        ))
    
    print(token.to_jwt())

if __name__ == "__main__":
    asyncio.run(generate_token())
