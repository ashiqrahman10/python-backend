import os
from typing import Any
from dotenv import load_dotenv
from gemini.initHist import init_hist
from google.generativeai import (
    GoogleGenerativeAI,
    HarmCategory,
    HarmBlockThreshold,
)

load_dotenv()
MODEL_NAME = "gemini-pro"
API_KEY = os.getenv("GEMINI_KEY")

generation_config = {
    "temperature": 0.9,
    "top_k": 1,
    "top_p": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
]

gemini_model: Any = None


async def setup_gemini_chat():
    global gemini_model
    gen_ai = GoogleGenerativeAI(API_KEY)
    gemini_model = gen_ai.get_generative_model(model=MODEL_NAME)


def start_gemini_chat(history: list = None):
    if history is None:
        history = []
    return gemini_model.start_chat(
        generation_config=generation_config,
        safety_settings=safety_settings,
        history=[*init_hist, *history],
    )