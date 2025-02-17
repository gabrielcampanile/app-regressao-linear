import streamlit as st
import numpy as np
import pandas as pd
from utils.plots import plot_with_repetition
from utils.stats import calculate_repeated_measures_stats
from scipy.stats import f as f_distribution

def page_with_repetition():
    st.title('Regressão Linear Simples Com Repetição')
    
    # Entrada de dados
    st.sidebar.header('Entrada de Dados')
    data_input_method = st.sidebar.radio(
        "Método de entrada:",
        ['Inserir dados', 'Importar CSV']
    )
    
    # Initialize X and y
    X = None
    y = None
    
    if data_input_method == 'Inserir dados':
        data = {
            'X': [2, 5, 1, 3, 4, 1, 5, 3, 4, 2],
            'Y': [50, 57, 41, 54, 54, 38, 63, 48, 59, 46]
        }
        X = np.array(data['X'])
        y = np.array(data['Y'])
    else:
        uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                X = df.iloc[:, 0].values
                y = df.iloc[:, 1].values
            except:
                st.error('Erro ao ler o arquivo CSV. Verifique o formato.')
                return
    
    if X is None or y is None:
        st.info('Por favor, selecione um método de entrada de dados.')
        return
    
    # Cálculos das estatísticas com repetição
    stats = calculate_repeated_measures_stats(X, y)
    
    # Inicialização dos nomes das variáveis
    x_name = "X"
    y_name = "Y"
    
    # Definição das cores (mesmas do gráfico)
    colors = {
        'SQT': '🟥 SQT',           # vermelho
        'SQReg': '🟩 SQReg',       # verde
        'SQErroReg': '🟦 SQErroReg', # azul
        'SQTrat': '⬛ SQTrat',     # cinza
        'SQFA': '🟪 SQFA',         # roxo
        'SQEP': '🟫 SQEP'          # marrom
    }

    # Seleção dos quadrados com cores correspondentes
    show_squares = st.sidebar.multiselect(
        'Mostrar Quadrados:',
        options=['SQT', 'SQReg', 'SQErroReg', 'SQTrat', 'SQFA', 'SQEP'],
        default=['SQT', 'SQReg', 'SQErroReg'],
        format_func=lambda x: colors[x],
        help="Selecione os quadrados para visualizar no gráfico"
    )
    
    # Row 1 - Dados e Gráfico
    st.subheader('Dados e Visualização')
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.caption("📝 Edite os dados diretamente na tabela")
        
        # Editor de dados principal
        edited_df = st.data_editor(
            pd.DataFrame({
                'X': X,
                'Y': y
            }),
            num_rows="dynamic",
            column_config={
                "X": st.column_config.NumberColumn(
                    "X",
                    help="Clique para editar os valores",
                    min_value=None,
                    max_value=None,
                    step=0.1,
                    required=True
                ),
                "Y": st.column_config.NumberColumn(
                    "Y",
                    help="Clique para editar os valores",
                    min_value=None,
                    max_value=None,
                    step=0.1,
                    required=True
                )
            },
            hide_index=False,  # Mostra o índice
            key='data_editor'
        )
        
        # Editor de nomes das variáveis
        st.caption("📝 Edite os nomes das variáveis")
        var_names_df = st.data_editor(
            pd.DataFrame({
                'Nome atual': ['X', 'Y'],
                'Novo nome': [x_name, y_name]
            }),
            column_config={
                "Nome atual": st.column_config.TextColumn(
                    "Nome atual",
                    help="Nome atual da variável",
                    disabled=True
                ),
                "Novo nome": st.column_config.TextColumn(
                    "Novo nome",
                    help="Digite o novo nome da variável"
                )
            },
            hide_index=True,
            key='var_names_editor'
        )
        
        # Atualiza os nomes das variáveis
        x_name = var_names_df['Novo nome'][0]
        y_name = var_names_df['Novo nome'][1]
        
        # Recalcula as estatísticas
        stats = calculate_repeated_measures_stats(X, y)
    
    with col2:
        fig = plot_with_repetition(X, y, stats, show_squares, x_name, y_name)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 - Resultados
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Tabela ANOVA')
        
        # Criação da tabela ANOVA
        anova_data = pd.DataFrame({
            'F.V.': ['Regressão linear', 'Falta de ajuste', 'Tratamento', 'Erro Puro', 'Erro Reg.', 'Total'],
            'gl': [
                1,  # Regressão
                stats['gl_lof'],  # Falta de ajuste
                stats['X_unique'].size - 1,  # Tratamento
                stats['gl_ep'],  # Erro puro
                stats['gl_error'],  # Erro reg
                len(X) - 1  # Total
            ],
            'SQ': [
                f"{stats['sqreg']:.4f}",
                f"{stats['sqlof']:.4f}",
                f"{stats['sqtrat']:.4f}",
                f"{stats['sqep']:.4f}",
                f"{stats['sqerror']:.4f}",
                f"{stats['sqt']:.4f}"
            ],
            'QM': [
                f"{stats['qm_reg']:.4f}",
                f"{stats['qm_lof']:.4f}",
                f"{stats['sqtrat']/(stats['X_unique'].size - 1):.4f}",
                f"{stats['qm_ep']:.4f}",
                f"{stats['qm_error']:.4f}",
                ''
            ],
            'F': [
                f"{stats['f_reg']:.4f}",
                f"{stats['f_lof']:.4f}",
                '',
                '',
                '',
                ''
            ]
        })
        
        # Exibição da tabela com estilo
        st.dataframe(
            anova_data,
            column_config={
                "F.V.": "Fonte de Variação",
                "gl": "Graus de Liberdade",
                "SQ": "Soma de Quadrados",
                "QM": "Quadrado Médio",
                "F": "F calculado"
            },
            hide_index=True,
            use_container_width=True
        )

    with col2:
        st.subheader('Estatísticas do Modelo')
        st.write(f"Coeficiente de Correlação (r) = {np.sqrt(stats['r2']):.4f}")
        st.write(f"Coeficiente de Determinação (R²) = {stats['r2']:.4f}")
        st.write(f"Coeficiente Angular (β₁) = {stats['model'].coef_[0]:.4f}")
        st.write(f"Intercepto (β₀) = {stats['model'].intercept_:.4f}")

    # Row 3 - Fórmulas e Explicações
    st.subheader('Decomposição da Soma de Quadrados')
    
    # 1. Decomposição Principal
    st.markdown("#### 1. Decomposição Principal")
    
    st.latex(r'''
    (y_{ij}-\bar{y}_{..}) = (\hat{y}_{ij}-\bar{y}_{..}) + (y_{ij}-\hat{y}_{ij})
    ''')

    st.latex(r'''
    \underbrace{\sum_{i=1}^a\sum_{j=1}^r(y_{ij}-\bar{y}_{..})^2}_{\color{red}{\text{SQT}}} = 
    \underbrace{\sum_{i=1}^a\sum_{j=1}^r(\hat{y}_{ij}-\bar{y}_{..})^2}_{\color{green}{\text{SQReg}}} + 
    \underbrace{\sum_{i=1}^a\sum_{j=1}^r(y_{ij}-\hat{y}_{ij})^2}_{\color{blue}{\text{SQErroReg}}}
    ''')
    
    st.markdown("""
    **Onde:**
    - yᵢⱼ: Valor observado no i-ésimo nível com a j-ésima repetição
    - ȳ..: Média geral dos dados
    - ŷᵢⱼ: Valor predito pela regressão
    - SQT: Soma de Quadrados Total
    - SQReg: Soma de Quadrados da Regressão
    - SQErroReg: Soma de Quadrados do Erro da Regressão
    """)
    
    # 2. Decomposição Total-Tratamento
    st.markdown("#### 2. Decomposição Total-Tratamento")
    
    st.latex(r'''
    (y_{ij}-\bar{y}_{..}) = (\bar{y}_{i.}-\bar{y}_{..}) + (y_{ij}-\bar{y}_{i.})
    ''')

    st.latex(r'''
    \underbrace{\sum_{i=1}^a\sum_{j=1}^r(y_{ij}-\bar{y}_{..})^2}_{\color{red}{\text{SQT}}} = 
    \underbrace{r\sum_{i=1}^a(\bar{y}_{i.}-\bar{y}_{..})^2}_{\color{gray}{\text{SQTrat}}} + 
    \underbrace{\sum_{i=1}^a\sum_{j=1}^r(y_{ij}-\bar{y}_{i.})^2}_{\color{brown}{\text{SQEP}}}
    ''')
    
    st.markdown("""
    **Onde:**
    - yᵢⱼ: Valor observado no i-ésimo nível com a j-ésima repetição
    - ȳ..: Média geral dos dados
    - ȳᵢ.: Média do i-ésimo nível (tratamento)
    - SQT: Soma de Quadrados Total
    - SQTrat: Soma de Quadrados do Tratamento
    - SQEP: Soma de Quadrados do Erro Puro
    """)
    
    # 3. Decomposição da Falta de Ajuste
    st.markdown("#### 3. Decomposição da Falta de Ajuste")
    
    st.latex(r'''
    (\bar{y}_{i.}-\hat{y}_{ij}) = (\bar{y}_{i.}-\bar{y}_{..}) - (\hat{y}_{ij}-\bar{y}_{..})
    ''')

    st.latex(r'''
    \underbrace{\sum_{i=1}^a\sum_{j=1}^r(\bar{y}_{i.}-\hat{y}_{ij})^2}_{\color{purple}{\text{SQFA}}} = 
    \underbrace{r\sum_{i=1}^a(\bar{y}_{i.}-\bar{y}_{..})^2}_{\color{gray}{\text{SQTrat}}} - 
    \underbrace{\sum_{i=1}^a\sum_{j=1}^r(\hat{y}_{ij}-\bar{y}_{..})^2}_{\color{green}{\text{SQReg}}}
    ''')
    
    st.markdown("""
    **Onde:**
    - ȳᵢ.: Média do i-ésimo nível (tratamento)
    - ȳ..: Média geral dos dados
    - ŷᵢⱼ: Valor predito pela regressão
    - SQFA: Soma de Quadrados da Falta de Ajuste
    - SQTrat: Soma de Quadrados do Tratamento
    - SQReg: Soma de Quadrados da Regressão
    """)
    
    # 4. Decomposição do Erro
    st.markdown("#### 4. Decomposição do Erro")
    
    st.latex(r'''
    (\bar{y}_{i.}-\hat{y}_{ij}) = (y_{ij}-\hat{y}_{ij}) - (y_{ij}-\bar{y}_{i.})
    ''')

    st.latex(r'''
    \underbrace{\sum_{i=1}^a\sum_{j=1}^r(\bar{y}_{i.}-\hat{y}_{ij})^2}_{\color{purple}{\text{SQFA}}} = 
    \underbrace{\sum_{i=1}^a\sum_{j=1}^r(y_{ij}-\hat{y}_{ij})^2}_{\color{blue}{\text{SQErroReg}}} - 
    \underbrace{\sum_{i=1}^a\sum_{j=1}^r(y_{ij}-\bar{y}_{i.})^2}_{\color{brown}{\text{SQEP}}}
    ''')
    
    st.markdown("""
    **Onde:**
    - yᵢⱼ: Valor observado no i-ésimo nível com a j-ésima repetição
    - ȳᵢ.: Média do i-ésimo nível (tratamento)
    - ŷᵢⱼ: Valor predito pela regressão
    - SQFA: Soma de Quadrados da Falta de Ajuste
    - SQErroReg: Soma de Quadrados do Erro da Regressão
    - SQEP: Soma de Quadrados do Erro Puro
    """)
    
    # # 5. Relação com o Tratamento
    # st.markdown("#### 5. Relação com o Tratamento")
    # col1, col2 = st.columns([1, 1])
    
    # with col1:
    #     st.latex(r'''
    #     (\bar{y}_{i.} - \bar{y}_{..}) = (\hat{y}_{ij} - \bar{y}_{..}) + (\bar{y}_{i.} - \hat{y}_{ij})
    #     ''')
    
    # with col2:
    #     st.latex(r'''
    #     \underbrace{\sum n_i(\bar{y}_{i.} - \bar{y}_{..})^2}_{\text{SQTrat}} = 
    #     \underbrace{\sum\sum(\hat{y}_{ij} - \bar{y}_{..})^2}_{\text{SQReg}} + 
    #     \underbrace{\sum\sum(\bar{y}_{i.} - \hat{y}_{ij})^2}_{\text{SQFA}}
    #     ''')
    
    # st.markdown("""
    # **Onde:**
    # - SQTrat: Soma de Quadrados do Tratamento
    # - nᵢ: Número de repetições no i-ésimo nível
    # """)

if __name__ == "__main__":
    st.set_page_config(page_title="Regressão com Repetição", layout="wide")
    page_with_repetition()