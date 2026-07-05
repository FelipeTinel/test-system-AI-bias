from app.schemas.analysis import RequestAnalysis, ResponseAnalysis
from fastapi import APIRouter, HTTPException
from app.services import get_ai_service
import json

analysis_router = APIRouter (prefix='/analysis', tags=['Bias Analysis'])

@analysis_router.post('/evaluate', response_model=ResponseAnalysis)
def evaluate_curriculum(payload: RequestAnalysis):

    try:

        ai_service = get_ai_service(payload.provider)

        raw_json_response = ai_service.curriculum_analyze(payload.curriculum)

        parsed_data = json.loads(raw_json_response)

        return parsed_data

    except ValueError as e:

        raise HTTPException(status_code=400, detail=str(e))
    
    except json.JSONDecodeError:

        raise HTTPException(status_code=502, detail='A IA falhou em responder no formato esperado')
    
    except Exception as e:

        raise HTTPException(status_code=500, detail=f'Erro interno no processamento: {str(e)}')

@analysis_router.post("/generate")
def generate_curriculums(provider: str = "gemini"):

    ai_service = get_ai_service(provider)

    response = ai_service.curriculum_generate()

    parsed = json.loads(response)

    return parsed