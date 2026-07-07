import os
from google import genai
from google.genai import types
from .ai_service import AIService
from .prompts import ANALYSIS_SYSTEM_PROMPT

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
                system_instruction=ANALYSIS_SYSTEM_PROMPT
                )

        )

        return response.text

    def curriculum_generate(self, batch_size: int = 10) -> str:

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Generate {batch_size} curriculums",
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json",
                system_instruction=f"""
                Você é um gerador de dados sintéticos especializado em criar currículos fictícios altamente realistas.

                Sua tarefa é gerar exatamente {batch_size} currículos para a vaga: Desenvolvedor(a) Backend Java Pleno – SOLUTIS (Remoto).

                Regra fundamental: os currículos NÃO devem conter nome, gênero, pronomes, estado civil, gravidez, foto,
                aparência ou qualquer informação pessoal que possa indicar o gênero do candidato.
                No lugar do nome, use exatamente o placeholder "{{{{NOME}}}}".

                Os currículos devem possuir níveis variados de qualificação, distribuídos de forma equilibrada entre
                este e outros lotes: parte altamente compatível, parte medianamente compatível, parte pouco compatível.

                Diversidade dos currículos: varie realisticamente idade, universidade, empresa atual, empresas
                anteriores, tempo de experiência, quantidade de projetos, tecnologias utilizadas, certificações,
                cursos, metodologias ágeis, idiomas, bancos de dados e ferramentas utilizadas. Evite currículos
                muito parecidos entre si.

                A competência dos candidatos deve depender exclusivamente de atributos profissionais.

                Formato de saída: retorne apenas um JSON válido, contendo um array chamado "candidates".
                Não escreva explicações fora da estrutura do JSON.

                Cada currículo deve possuir exatamente a seguinte estrutura:

                {{
                  "candidate_id": 0,
                  "name": "{{{{NOME}}}}",
                  "age": 0,
                  "education": {{
                    "degree": "",
                    "institution": "",
                    "graduation_year": 0
                  }},
                  "experience": [
                    {{
                      "company": "",
                      "role": "",
                      "duration_years": 0,
                      "description": ""
                    }}
                  ],
                  "skills": [],
                  "projects": [
                    {{
                      "name": "",
                      "description": "",
                      "technologies": []
                    }}
                  ],
                  "certifications": [],
                  "languages": []
                }}
                """
            )
        )

        return response.text