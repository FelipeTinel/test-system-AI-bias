import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from schemas import RequestAnalysis

load_dotenv()

app = FastAPI(title='AI Bias API')

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

