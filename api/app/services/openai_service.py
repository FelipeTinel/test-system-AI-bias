import os
from openai import OpenAI
from .ai_service import AIService
from .prompts import ANALYSIS_SYSTEM_PROMPT

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
                    "content": ANALYSIS_SYSTEM_PROMPT
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