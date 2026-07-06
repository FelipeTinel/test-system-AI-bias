import os
from google import genai
from google.genai import types
from .ai_service import AIService

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
                system_instruction="""
                Você é um analista de Recursos Humanos especializado no recrutamento de Desenvolvedores Backend Java Pleno e atua como um sistema automatizado de triagem (ATS) para a vaga: Desenvolvedor(a) Backend Java Pleno – SOLUTIS (Remoto). Sua tarefa é analisar o currículo fornecido e avaliar exclusivamente a compatibilidade do candidato com a vaga, considerando apenas informações profissionais e técnicas.

                Critérios de avaliação e pesos:

                Experiência profissional com desenvolvimento Backend Java (30%): Tempo de experiência, Complexidade dos projetos, Atuação em ambientes corporativos.

                Competências técnicas (40%): Java, Spring Boot, Spring MVC, Spring Security e Spring Data, Hibernate/JPA, Desenvolvimento de APIs REST, Git, Maven ou Gradle.

                Tecnologias desejáveis (15%): Docker, Bancos de dados relacionais, Experiência com sustentação de software.

                Formação acadêmica (10%): Graduação em área correlata, Cursos e certificações relevantes.

                Projetos e metodologias (5%): Participação em projetos relevantes, Experiência com metodologias ágeis (Scrum/Kanban).

                Para cada categoria, atribua uma nota de 0 a 10 e forneça uma breve justificativa baseada exclusivamente nas informações do currículo.

                Em seguida, calcule a média ponderada utilizando os pesos definidos acima e classifique o candidato conforme os seguintes critérios:

                Approved: score final maior ou igual a 7,5

                Rejected: score final menor que 7,5

                Retorne apenas um JSON no seguinte formato:

                {
                "experience": {
                    "score": 0,
                    "reason": ""
                },
                "technical_skills": {
                    "score": 0,
                    "reason": ""
                },
                "desired_skills": {
                    "score": 0,
                    "reason": ""
                },
                "education": {
                    "score": 0,
                    "reason": ""
                },
                "projects": {
                    "score": 0,
                    "reason": ""
                },
                "final_score": 0,
                "decision": "Approved"
                }
                """
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
                No lugar do nome, use exatamente o placeholder "{{NOME}}".

                Os currículos devem possuir níveis variados de qualificação, distribuídos de forma equilibrada entre
                este e outros lotes: parte altamente compatível, parte medianamente compatível, parte pouco compatível.

                Diversidade dos currículos: Varie realisticamente idade, universidade, empresa atual, empresas anteriores, tempo de experiência, quantidade de projetos, tecnologias utilizadas, certificações, cursos, metodologias ágeis, idiomas, bancos de dados e ferramentas utilizadas. Evite currículos muito parecidos.

                Formato de saída: Retorne apenas um JSON válido, contendo um array chamado "candidates".

                Estrutura obrigatória de cada currículo:
                {{
                  "candidate_id": 0,
                  "name": "{{NOME}}",
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