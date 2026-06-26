from api.app.schemas.analysis import RequestAnalysis
from fastapi import APIRouter, HTTPException

analysis_router = APIRouter (prefix='/analysis', tags=['Bias Analysis'])

@analysis_router.post('/analyze')
def curriculum_analysis(data: RequestAnalysis):
    return { 'status' : 'Sucess!', 'provider' : data.ai }

