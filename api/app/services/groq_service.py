import os
from groq import Groq
from .ai_service import AIService

class GroqService (AIService):

    def __init__(self):

        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    def curriculum_analyze(self, curriculum: str) -> str:
        
        response = self.client.chat.completions.create (

            model='llama-3.3-70b-versatile',
            temperature=0.0,
            response_format={"type" : "json_object"},
            messages=[
                {
                    "role": "system", 
                    "content": "You are an HR analyst. Evaluate the curriculum and return a JSON object strictly with two keys: 'score' (an integer from 0 to 10) and 'status' (a string: 'Approved' or 'Rejected')."
                },
                {
                    "role": "user", 
                    "content": f"Analysis: {curriculum}"
                }
                
                ]

        )

        return response.choices[0].message.content
    
    def curriculum_generate(self) -> str:
        raise NotImplementedError(
            "Curriculum generation is only supported by Gemini."
        )