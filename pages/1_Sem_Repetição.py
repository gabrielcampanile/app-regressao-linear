import streamlit as st
import numpy as np
import pandas as pd
from utils.plots import plot_no_repetition
from utils.stats import calculate_basic_stats
from sklearn.linear_model import LinearRegression

def page_no_repetition():
    st.title('Regressão Linear Simples Sem Repetição')
    
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
            'X': [150, 155, 160, 165, 170, 175, 180, 185],
            'Y': [125, 130, 140, 145, 150, 155, 160, 165]
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
    
    # Cálculos
    stats = calculate_basic_stats(X, y)
    
    # Inicialização dos nomes das variáveis
    x_name = "X"
    y_name = "Y"
    
    # Definição das cores (mesmas do gráfico)
    colors = {
        'SQT': '🟥 SQT',       # vermelho
        'SQReg': '🟩 SQReg',   # verde
        'SQE': '🟦 SQE'        # azul
    }
    
    # Seleção dos quadrados com cores correspondentes
    show_squares = st.sidebar.multiselect(
        'Mostrar Quadrados:',
        options=['SQT', 'SQReg', 'SQE'],
        default=['SQT', 'SQReg', 'SQE'],
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
        stats = calculate_basic_stats(X, y)
    
    with col2:
        fig = plot_no_repetition(X, y, stats, show_squares, x_name, y_name)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 - Resultados
    st.subheader('Resultados')
    col1, col2 = st.columns([2, 1])

    with col1:
        # Tabela ANOVA
        st.subheader('Tabela ANOVA')
        anova_data = pd.DataFrame({
            'Fonte': ['Regressão', 'Erro', 'Total'],
            'GL': [
                stats['gl_reg'],
                stats['gl_error'],
                stats['gl_total']
            ],
            'SQ': [
                f"{stats['sqreg']:.3f}",
                f"{stats['sqerror']:.3f}",
                f"{stats['sqt']:.3f}"
            ],
            'QM': [
                f"{stats['qm_reg']:.3f}",
                f"{stats['qm_error']:.3f}",
                ''
            ],
            'F': [
                f"{stats['f_stat']:.3f}",
                '',
                ''
            ]
        })
        st.dataframe(anova_data, use_container_width=True)

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
    (y_i-\bar{y}) = (\hat{y}_i-\bar{y}) + (y_i-\hat{y}_i)
    ''')

    st.latex(r'''
    \underbrace{\sum_{i=1}^n(y_i-\bar{y})^2}_{\color{red}{\text{SQT}}} = 
    \underbrace{\sum_{i=1}^n(\hat{y}_i-\bar{y})^2}_{\color{green}{\text{SQReg}}} + 
    \underbrace{\sum_{i=1}^n(y_i-\hat{y}_i)^2}_{\color{blue}{\text{SQE}}}
    ''')
    
    st.markdown("""
    **Onde:**
    - yᵢ: Valor observado
    - ȳ: Média geral dos dados
    - ŷᵢ: Valor predito pela regressão
    - SQT: Soma de Quadrados Total
    - SQReg: Soma de Quadrados da Regressão
    - SQE: Soma de Quadrados do Erro
    """)

if __name__ == "__main__":
    st.set_page_config(page_title="Regressão sem Repetição", layout="wide")
    page_no_repetition()