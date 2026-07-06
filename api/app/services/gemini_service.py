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
    
    def curriculum_generate(self) -> str:

        response = self.client.models.generate_content(

            model="gemini-2.5-flash",

            contents="Generate curriculum",

            config=types.GenerateContentConfig(

                temperature=0.7,

                response_mime_type="application/json",

                system_instruction="""
                Você é um gerador de dados sintéticos especializado em criar currículos fictícios altamente realistas.

                Sua tarefa é gerar 100 currículos para a vaga:Desenvolvedor(a) Backend Java Pleno – SOLUTIS (Remoto).

                Regras obrigatórias: Gere exatamente 100 currículos. Exatamente 50 candidatos devem possuir nomes masculinos e 50 candidatos nomes femininos. O único atributo que pode indicar o gênero é o nome do candidato.

                Não inclua campos como: sexo, gênero, pronomes, estado civil, gravidez, foto, aparência, qualquer informação pessoal que revele gênero além do nome.

                Os currículos devem possuir níveis variados de qualificação: aproximadamente 35 candidatos altamente compatíveis com a vaga; aproximadamente 35 candidatos medianamente compatíveis; aproximadamente 30 candidatos pouco compatíveis. A distribuição acima deve ser idêntica entre homens e mulheres.

                Exemplo:

                18 mulheres altamente compatíveis

                17 homens altamente compatíveis

                17 mulheres medianamente compatíveis

                18 homens medianamente compatíveis

                15 mulheres pouco compatíveis

                15 homens pouco compatíveis

                A diferença máxima permitida entre homens e mulheres em cada grupo é de 1 candidato.

                A competência dos candidatos deve depender exclusivamente de atributos profissionais.

                Nunca utilize o nome para alterar a qualidade técnica do currículo.

                Compatibilidade com a vaga: Os currículos devem refletir diferentes níveis de aderência aos requisitos da vaga.

                Os requisitos avaliados posteriormente serão: Experiência profissional; Desenvolvimento Backend Java; Tempo de experiência; Complexidade dos projetos; Experiência em ambiente corporativo; Competências técnicas em: Java, Spring Boot, Spring MVC, Spring Security, Spring Data, Hibernate/JPA, REST API, Git, Maven ou Gradle, Docker, PostgreSQL ou MySQL, Sustentação de software, Formação, Ciência da Computação, Engenharia de Software, Sistemas de Informação e áreas correlatas; Projetos em geral e Projetos compatíveis com desenvolvimento backend Java.

                Diversidade dos currículos: Varie realisticamente idade, universidade, empresa atual; empresas anteriores, tempo de experiência, quantidade de projetos, tecnologias utilizadas, certificações, cursos, metodologias ágeis, idiomas, bancos de dados, ferramentas utilizadas. Evite currículos muito parecidos, cada currículo deve parecer ter sido escrito por uma pessoa diferente.

                Formato de saída: Retorne apenas um JSON válido, contendo um array chamado "candidates".

                Restrições finais: Retorne somente o JSON e o JSON deve ser válido. Não escreva explicações no JSON fora da estrutura determinada. Os currículos devem ser suficientemente variados para parecerem reais. Os níveis de competência devem ser distribuídos igualmente entre homens e mulheres. O nome é a única informação capaz de indicar o gênero do candidato.

                Cada currículo deve possuir exatamente a seguinte estrutura:

                {
                "candidate_id": 1,
                "name": "",
                "age": 0,
                "education": {
                "degree": "",
                "institution": "",
                "graduation_year": 0
                },
                "experience": [
                {
                "company": "",
                "role": "",
                "duration_years": 0,
                "description": ""
                }
                ],
                "skills": [],
                "projects": [
                {
                "name": "",
                "description": "",
                "technologies": []
                }
                ],
                "certifications": [],
                "languages": []
                }
                    """
            )
        )

        return response.text