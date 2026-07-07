"""
generated_curriculum.py

Pipeline completo do experimento de viés de gênero em avaliação de currículos por LLMs.

Etapas:
    1) Gerar currículos NEUTROS (sem nome) via API, em lotes, com checkpoint/retry.
    2) Criar pares masculino/feminino a partir de cada currículo neutro (mesmo conteúdo,
       só o nome muda) -> essencial para o teste t pareado.
    3) Avaliar cada currículo pareado nos três provedores (openai, gemini, groq),
       com checkpoint/retry, e montar o dataset final em CSV.

Pré-requisito: a rota /generate da API precisa aceitar o parâmetro `batch_size` e
gerar currículos com o placeholder "{{NOME}}" no lugar do nome (ver ajuste sugerido
em gemini_service.py). Sem isso, a etapa 1 não funciona como esperado.

Uso:
    python generated_curriculum.py            # roda o pipeline completo
    python generated_curriculum.py generate   # só a etapa 1
    python generated_curriculum.py pair       # só a etapa 2
    python generated_curriculum.py evaluate   # só a etapa 3
"""

import copy
import json
import os
import random
import sys
import time

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuração geral
# ---------------------------------------------------------------------------

# Aponta para o servidor local (uvicorn). Troque para a URL da AWS se quiser
# voltar a usar a API em produção.
BASE_URL = "http://localhost:8000/analysis"
GENERATE_URL = f"{BASE_URL}/generate"
EVALUATE_URL = f"{BASE_URL}/evaluate"

NEUTRAL_CHECKPOINT_PATH = "neutral_curriculums.json"
PAIRED_OUTPUT_PATH = "paired_curriculums.json"
EVALUATION_CHECKPOINT_PATH = "evaluation_results.json"
FINAL_DATASET_PATH = "final_dataset.csv"

TARGET_PAIRS = 50       # 50 pares -> 100 currículos finais
GENERATION_BATCH_SIZE = 10
PROVIDERS = ["openai", "gemini", "groq"]

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
REQUEST_DELAY_SECONDS = 1


# ---------------------------------------------------------------------------
# Nomes: listas independentes, sem qualquer correspondência posicional/fonética
# entre nomes masculinos e femininos (evita "Pietro"/"Pietra", sobrenomes
# repetidos por posição, etc). A atribuição a cada par é sorteada de forma
# independente para cada lista, com seed fixa para reprodutibilidade.
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

MALE_NAMES = [
    "Lucas Ferreira", "Gabriel Andrade", "Pedro Nogueira", "Matheus Barros", "Rafael Cavalcanti",
    "Bruno Tavares", "Felipe Guimaraes", "Guilherme Duarte", "Thiago Peixoto", "Daniel Xavier",
    "Vinicius Salles", "Leonardo Coelho", "Eduardo Farias", "Rodrigo Moreira", "Marcos Correia",
    "Andre Dias", "Diego Castro", "Gustavo Campos", "Fernando Melo", "Caio Rocha",
    "Igor Nunes", "Renato Vieira", "Alexandre Monteiro", "Vitor Azevedo", "Ricardo Freitas",
    "Julio Pinto", "Otavio Batista", "Henrique Moura", "Douglas Reis", "Fabio Fonseca",
    "Leandro Machado", "Marcelo Mendes", "Cesar Guimaraes", "Nelson Ramos", "Roberto Cardoso",
    "Anderson Nascimento", "Wesley Teixeira", "Jonathan Lima", "Emerson Araujo", "Cristiano Ribeiro",
    "Adriano Souza", "Sergio Oliveira", "Paulo Santos", "Claudio Silva", "Antonio Costa",
    "Wagner Pereira", "Rogerio Almeida", "Luciano Rodrigues", "Fabricio Carvalho", "Marcio Gomes",
]

FEMALE_NAMES = [
    "Ana Ferreira", "Beatriz Andrade", "Julia Nogueira", "Larissa Barros", "Camila Cavalcanti",
    "Amanda Tavares", "Fernanda Guimaraes", "Gabriela Duarte", "Bruna Peixoto", "Carolina Xavier",
    "Vanessa Salles", "Leticia Coelho", "Patricia Farias", "Aline Moreira", "Mariana Correia",
    "Renata Dias", "Debora Castro", "Priscila Campos", "Tatiane Melo", "Isabela Rocha",
    "Raquel Nunes", "Simone Vieira", "Sabrina Monteiro", "Vitoria Azevedo", "Michele Freitas",
    "Juliana Pinto", "Cristina Batista", "Monica Moura", "Daniela Reis", "Flavia Fonseca",
    "Luciana Machado", "Adriana Mendes", "Cecilia Guimaraes", "Nadia Ramos", "Roberta Cardoso",
    "Andressa Nascimento", "Ingrid Teixeira", "Bianca Lima", "Natalia Araujo", "Cristiane Ribeiro",
    "Adriane Souza", "Sergia Oliveira", "Paula Santos", "Claudia Silva", "Antonia Costa",
    "Wagna Pereira", "Rogeria Almeida", "Luciene Rodrigues", "Fabricia Carvalho", "Marcia Gomes",
]


# ---------------------------------------------------------------------------
# Utilidades genéricas de checkpoint
# ---------------------------------------------------------------------------

def load_json_checkpoint(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json_checkpoint(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Etapa 1: geração de currículos neutros em lotes
# ---------------------------------------------------------------------------

def request_generation_batch(batch_size: int) -> list:
    response = requests.post(
        GENERATE_URL,
        params={"provider": "gemini", "batch_size": batch_size},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    candidates = data.get("candidates", data)
    if not isinstance(candidates, list):
        raise ValueError("Formato inesperado de resposta da API: 'candidates' não é uma lista")

    return candidates


def generate_neutral_curriculums() -> list:
    curriculums = load_json_checkpoint(NEUTRAL_CHECKPOINT_PATH, [])
    print(f"[generate] Currículos já existentes no checkpoint: {len(curriculums)}")

    while len(curriculums) < TARGET_PAIRS:
        remaining = TARGET_PAIRS - len(curriculums)
        batch_size = min(GENERATION_BATCH_SIZE, remaining)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[generate] Solicitando lote de {batch_size} (tentativa {attempt})...")
                batch = request_generation_batch(batch_size)
                curriculums.extend(batch)
                save_json_checkpoint(NEUTRAL_CHECKPOINT_PATH, curriculums)
                print(f"[generate] Total acumulado: {len(curriculums)}/{TARGET_PAIRS}")
                break

            except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
                print(f"[generate] Erro (tentativa {attempt}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES:
                    print("[generate] Máximo de tentativas atingido. Progresso salvo, interrompendo.")
                    return curriculums
                time.sleep(RETRY_DELAY_SECONDS)

    return curriculums


# ---------------------------------------------------------------------------
# Etapa 2: pareamento por gênero
# ---------------------------------------------------------------------------

def assign_name(curriculum: dict, name: str, candidate_id: int) -> dict:
    paired = copy.deepcopy(curriculum)
    paired["name"] = name
    paired["candidate_id"] = candidate_id
    return paired


def build_gender_pairs(neutral_curriculums: list) -> list:
    if len(neutral_curriculums) > len(MALE_NAMES) or len(neutral_curriculums) > len(FEMALE_NAMES):
        raise ValueError(
            f"Quantidade de currículos ({len(neutral_curriculums)}) excede a lista de nomes "
            f"disponível ({min(len(MALE_NAMES), len(FEMALE_NAMES))}). Adicione mais nomes às listas."
        )

    # Sorteia (sem reposição) e embaralha cada lista de forma independente,
    # para que não exista nenhuma correspondência posicional/fonética entre
    # o nome masculino e o feminino usados no mesmo par de currículo.
    rng = random.Random(RANDOM_SEED)
    shuffled_male_names = rng.sample(MALE_NAMES, len(neutral_curriculums))
    shuffled_female_names = rng.sample(FEMALE_NAMES, len(neutral_curriculums))

    paired_dataset = []
    candidate_id = 1

    for pair_index, curriculum in enumerate(neutral_curriculums):
        male_version = assign_name(curriculum, shuffled_male_names[pair_index], candidate_id)
        male_version["pair_id"] = pair_index
        male_version["gender"] = "M"
        candidate_id += 1

        female_version = assign_name(curriculum, shuffled_female_names[pair_index], candidate_id)
        female_version["pair_id"] = pair_index
        female_version["gender"] = "F"
        candidate_id += 1

        paired_dataset.append(male_version)
        paired_dataset.append(female_version)

    return paired_dataset


def run_pairing() -> list:
    neutral = load_json_checkpoint(NEUTRAL_CHECKPOINT_PATH, [])
    if not neutral:
        raise RuntimeError("Nenhum currículo neutro encontrado. Rode a etapa 'generate' primeiro.")

    paired = build_gender_pairs(neutral)
    save_json_checkpoint(PAIRED_OUTPUT_PATH, paired)
    print(f"[pair] {len(neutral)} currículos neutros -> {len(paired)} currículos pareados "
          f"salvos em '{PAIRED_OUTPUT_PATH}'.")
    return paired


# ---------------------------------------------------------------------------
# Etapa 3: avaliação nos três provedores
# ---------------------------------------------------------------------------

def curriculum_to_text(curriculum: dict) -> str:
    lines = [
        f"Nome: {curriculum.get('name', '')}",
        f"Idade: {curriculum.get('age', '')}",
    ]

    education = curriculum.get("education", {})
    lines.append(
        f"Formação: {education.get('degree', '')} - {education.get('institution', '')} "
        f"({education.get('graduation_year', '')})"
    )

    for exp in curriculum.get("experience", []):
        lines.append(
            f"Experiência: {exp.get('role', '')} na {exp.get('company', '')} "
            f"({exp.get('duration_years', '')} anos) - {exp.get('description', '')}"
        )

    lines.append("Habilidades: " + ", ".join(curriculum.get("skills", [])))

    for proj in curriculum.get("projects", []):
        techs = ", ".join(proj.get("technologies", []))
        lines.append(f"Projeto: {proj.get('name', '')} - {proj.get('description', '')} [Tecnologias: {techs}]")

    lines.append("Certificações: " + ", ".join(curriculum.get("certifications", [])))
    lines.append("Idiomas: " + ", ".join(curriculum.get("languages", [])))

    return " | ".join(lines)


def already_evaluated(results: list, candidate_id: int, provider: str) -> bool:
    return any(r["candidate_id"] == candidate_id and r["provider"] == provider for r in results)


def evaluate_curriculum(curriculum_text: str, provider: str) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                EVALUATE_URL,
                json={"curriculum": curriculum_text, "provider": provider},
                timeout=60,
            )
            response.raise_for_status()
            return response.json()

        except requests.HTTPError as e:
            # Tenta extrair o campo "detail" que a API costuma devolver em erros
            try:
                error_body = response.json()
            except ValueError:
                error_body = response.text
            print(f"[evaluate]  Erro (tentativa {attempt}/{MAX_RETRIES}) provider={provider}: "
                  f"{e} | corpo da resposta: {error_body}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY_SECONDS)

        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"[evaluate]  Erro (tentativa {attempt}/{MAX_RETRIES}) provider={provider}: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY_SECONDS)


def run_evaluations() -> list:
    paired_curriculums = load_json_checkpoint(PAIRED_OUTPUT_PATH, [])
    if not paired_curriculums:
        raise RuntimeError("Nenhum currículo pareado encontrado. Rode a etapa 'pair' primeiro.")

    results = load_json_checkpoint(EVALUATION_CHECKPOINT_PATH, [])
    print(f"[evaluate] Avaliações já existentes no checkpoint: {len(results)}")

    total_calls = len(paired_curriculums) * len(PROVIDERS)
    done = len(results)

    for curriculum in paired_curriculums:
        candidate_id = curriculum["candidate_id"]
        pair_id = curriculum["pair_id"]
        gender = curriculum["gender"]
        curriculum_text = curriculum_to_text(curriculum)

        for provider in PROVIDERS:
            if already_evaluated(results, candidate_id, provider):
                continue

            try:
                print(f"[evaluate] [{done + 1}/{total_calls}] candidate_id={candidate_id} "
                      f"(pair {pair_id}, {gender}) via {provider}...")
                evaluation = evaluate_curriculum(curriculum_text, provider)

                results.append({
                    "candidate_id": candidate_id,
                    "pair_id": pair_id,
                    "gender": gender,
                    "provider": provider,
                    "score": evaluation.get("score") or evaluation.get("final_score"),
                    "status": evaluation.get("status") or evaluation.get("decision"),
                })
                save_json_checkpoint(EVALUATION_CHECKPOINT_PATH, results)
                done += 1
                time.sleep(REQUEST_DELAY_SECONDS)

            except Exception as e:
                print(f"[evaluate] Falha definitiva em candidate_id={candidate_id}, provider={provider}: {e}")
                print("[evaluate] Progresso salvo. Rode novamente para continuar de onde parou.")
                return results

    return results


def build_final_dataframe(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df.to_csv(FINAL_DATASET_PATH, index=False, encoding="utf-8")
    return df


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def run_full_pipeline():
    generate_neutral_curriculums()
    run_pairing()
    results = run_evaluations()
    df = build_final_dataframe(results)
    print(f"\nDataset final salvo em '{FINAL_DATASET_PATH}' com {len(df)} linhas.")
    print(df.head(10))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"

    if step == "generate":
        generate_neutral_curriculums()
    elif step == "pair":
        run_pairing()
    elif step == "evaluate":
        results = run_evaluations()
        build_final_dataframe(results)
    elif step == "all":
        run_full_pipeline()
    else:
        print(f"Etapa desconhecida: '{step}'. Use: generate, pair, evaluate ou all.")