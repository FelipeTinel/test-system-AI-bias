from pydantic import BaseModel, Field

class RequestAnalysis(BaseModel):
    curriculum: str = Field(..., description="Texto puro do currículo a ser analisado")
    provider: str = Field("openai", description="Provedor de IA: 'openai', 'gemini' ou 'groq'")

class ResponseAnalysis(BaseModel):
    score: int = Field(..., description="Nota de 0 a 10 atribuída ao currículo")
    status: str = Field(..., description="Status final: 'Approved' or 'Rejected'")
class ResponseCurriculumGeneration(BaseModel):
    curriculums: list[str]

