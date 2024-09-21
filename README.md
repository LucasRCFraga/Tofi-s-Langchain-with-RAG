# Tofi's Langchain chatbot with RAG

## Descrição do projeto (PT-BR)

O objetivo deste projeto é criar um chatbot utilizando a biblioteca Langchain, aplicando a técnica de Recuperação de Informação Automática (RAG), com aplicação de tratamento de prompts avançados utilizando
RAG-Fusion como método de combinação de múltiplas fontes de informação.

## Conteúdo

- app.py: Código da aplicação
- Streamlit_server.py: Código da aplicação utilizando streamlit
- data/observations.py: Minhas observações sobre testes do código durante o desenvolvimento
- requirements.txt: Dependências do projeto

## Como rodar o projeto

1. Clone o repositório
2. Rode o comando pip install -r requirements.txt
3. É necessário ter uma API key da OpenAI (OPENAI_API_KEY) no seu .env
4. Rode o comando streamlit run Streamlit_server.py
5. Caso queira rodar o app.py, é necessário utilizar o comando python app.py

--------------------------------------------------------------------------------

Ideias para um eventual v0.02
    -Finalizar a implementação do Streamlit_server.py com funcionalidades que o app.py já tem;
    -Implementar mais melhorias na interface do Streamlit(Um botão para trocar entre RAG e um chatbot simples);
    -Mudança do idioma da página do Streamlit baseado no idioma do browser do usuário;
    -Implementar uma função de autenticação de usuários;
    -Mais opções de documentos além de apenas PDFs;

## Project Description (EN-US)

## Project Description

The goal of this project is to create a chatbot using the Langchain library, applying the Retrieval-Augmented Generation (RAG) technique, with advanced prompt processing using RAG-Fusion as a method for combining multiple information sources.

## Contents

- app.py: Application code
- Streamlit_server.py: Application code using Streamlit
- data/observations.py: My observations on code tests during development
- requirements.txt: Project dependencies

## How to run the project

1. Clone the repository
2. Run the command pip install -r requirements.txt
3. You need to have an OpenAI API key (OPENAI_API_KEY) in your .env file
4. Run the command streamlit run Streamlit_server.py
5. If you want to run app.py, you need to use the command python app.py

--------------------------------------------------------------------------------

Ideas for a potential v0.02
    - Complete the implementation of Streamlit_server.py with functionalities that app.py already has;
    - Implement more improvements in the Streamlit interface (A button to switch between RAG and a simple chatbot);
    - Change the language of the Streamlit page based on the user's browser language;
    - Implement a user authentication function;
    - More document options besides just PDFs;
