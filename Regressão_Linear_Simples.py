import streamlit as st

def home_page():
    st.title("Visualização Interativa ANOVA")
    
    st.markdown("""
    ### Bem-vindo à Ferramenta de Visualização ANOVA!
    
    Esta é uma ferramenta educacional interativa desenvolvida para auxiliar no entendimento 
    dos componentes da Análise de Variância (ANOVA) em regressão linear simples.
    
    #### 🎯 Objetivo
    Facilitar a compreensão visual dos quadrados das diferenças baseados no método dos 
    Quadrados Mínimos, permitindo uma melhor interpretação dos componentes da tabela ANOVA.
    
    #### 📊 Recursos Principais
    - **Visualização dos Componentes ANOVA:**
        - SQE (Soma dos Quadrados do Erro)
        - SQR (Soma dos Quadrados da Regressão)
        - SQT (Soma dos Quadrados Total)
    
    #### 🔍 Tipos de Análise Disponíveis
    1. **Regressão Linear Simples sem Repetição**
    2. **Regressão Linear Simples com Repetição**
    
    #### 📝 Como Usar
    1. Selecione o tipo de regressão desejado no menu lateral
    2. Insira seus dados ou importe um arquivo CSV
    3. Explore os gráficos e resultados interativos
    
    #### 💡 Dica
    Utilize esta ferramenta como complemento ao seu estudo de estatística, 
    experimentando diferentes conjuntos de dados para melhor compreensão dos conceitos.
    """)
    
    st.sidebar.markdown("""
    #### ℹ️ Sobre
    Desenvolvido como ferramenta educacional para auxiliar 
    no ensino e aprendizagem de Análise de Variância.
    """)

if __name__ == '__main__':
    home_page()