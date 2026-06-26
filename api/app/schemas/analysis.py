from pydantic import BaseModel

class RequestAnalysis (BaseModel):
    curriculum: str
    gender: str
    ai: str