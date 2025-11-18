# config/openai_client.py

from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env once (safe to call multiple times)
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env file")

# Create a global client
client = OpenAI(api_key=api_key)

# Default model for your project
DEFAULT_MODEL = "gpt-4o-mini"  # or "gpt-4.1" / "gpt-4.1-preview"
