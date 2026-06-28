import os
from google import genai
from google.genai import types
from ai_service import AIService

class GeminiService(AIService):

    def __init__(self):
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    def curriculum_analyze(self, curriculum: str) -> str:
        
        response = self.client.models.generate_content(
            
            model='gemini-2.5-flash',
            contents=f'Analysis: {curriculum}',
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type='application/json',
                system_instruction="You are an HR analyst. Evaluate the curriculum and return a JSON object strictly with two keys: 'score' (an integer from 0 to 10) and 'status' (a string: 'Approved' or 'Rejected')."
                )

        )

        return response.text