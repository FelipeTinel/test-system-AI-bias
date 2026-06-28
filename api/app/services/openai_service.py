import os
from openai import OpenAI
from ai_service import AIService

class OpenAIService (AIService):

    def __init__(self):

        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def curriculum_analyze(self, curriculum: str) -> str:
        
        response = self.client.chat.completions.create (

            model='gpt-4o-mini',
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

