# Sistema de Análise de viés de IA
Observações escritas por Felipe Tinel (digo por que alguns momentos me refiro em primeira pessoa)

## Como começar a trabalhar:

O rito inicial pra começar a mexer no projeto é: </br>

1 ) Entrar nas pastas api ou AI_bias pelo terminal; </br>
2 ) Executar os seguintes comandos:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Apenas uma vez em cada uma das pastas (cada uma tem um venv diferente); </br></br>
3 ) Uma vez dentro da pasta que você vai mexer naquele momento, rodar:
```bash
source .venv/bin/activate
```
4 ) Caso queira sair do venv, basta rodar:
```bash
deactivate
```

## API:

Atualmente existem duas pastas, api e AI bias. A pasta api, como o nome diz, tem a api usada para o projeto.
Tudo presente nela envolve criar requisições para a openai, google e groq de maneira automatizada. </br>

A URL para usa-la é essa aqui:

</br>

#### https://u1t86mt1g3.execute-api.us-east-1.amazonaws.com/Prod/analysis/

</br>

</br>

A rota "analysis/" vai retornar um JSON com uma mensagem dizendo "Is working". Caso não retorne, alguma coisa deu errado. </br>
Pode tentar resolver ou me informar. </br> </br>

A rota que retorna uma análise feita por uma IA é "analysis/evaluate", ou simplesmente:

</br>

#### https://u1t86mt1g3.execute-api.us-east-1.amazonaws.com/Prod/analysis/evaluate

</br>

Você manda um JSON assim:

```bash
{

    curriculum : (string de linha única de um currículo)
    provider : (uma das três ias do projeto, que são openai, gemini, groq, escritos dessa forma)

}
```

E recebe outro JSON assim:

```bash
{

    score : (um valor inteiro de 0 a 10)
    status : (uma string dizendo "Approved" ou "Rejected")

}
```

Em cima disso, é possível criar um dataset com os resultado obtidos pelos currículos enviados. </br>

Por enquanto, só existem essas rotas funcionais de fato. Caso alguma coisa não ocorra como previsto nesse README, repito, me informe. </br>

Obs: Qualquer alteração feita em /api, também me fala pra que eu possa dar deploy.

## Criação do experimento:

Na pasta "AI_bias", no momento em que estou escrevendo isso, não tem nada de fato feito ainda. A ideia é que tenha as seguintes coisas:

```bash
- /uma pasta/arquivo com a criação do primeiro dataset com os currículos sintéticos.
- /uma pasta/arquivo para a criação do dataset com o resultado de cada uma das IAs.
- /uma pasta/arquivo com a realização dos experimentos em questão.
- /uma pasta com o gráfico dos resultados.
```

Em geral, acho que é isso
