import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from app.schemas.analysis import RequestAnalysis
from app.routes.analysis import analysis_router

load_dotenv()

app = FastAPI(title='AI Bias API', version='1.0.0')

app.include_router('analysis_router')

@app.get('/')
def root():
    return {'message' : 'Analysis API is running'}

