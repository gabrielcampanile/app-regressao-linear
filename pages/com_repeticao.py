import streamlit as st
import numpy as np
import pandas as pd
from utils.plots import plot_with_repetition
from utils.stats import calculate_basic_stats
from scipy.stats import f as f_distribution

def calculate_repeated_measures_stats(X, y):
    """Calcula estatísticas para regressão com repetição"""
    # Estatísticas básicas
    basic_stats = calculate_basic_stats(X, y)
    
    # Cálculos adicionais para repetição
    unique_X = np.unique(X)
    n = len(unique_X)  # número de níveis
    N = len(X)         # número total de observações
    
    # Médias por nível
    y_means = np.array([np.mean(y[X == x]) for x in unique_X])
    n_per_level = np.array([np.sum(X == x) for x in unique_X])
    
    # Cálculos para falta de ajuste
    sqtrat = np.sum(n_per_level * (y_means - basic_stats['y_mean'])**2)
    sqlof = basic_stats['sqreg'] - sqtrat
    
    # Cálculo do erro puro
    sqep = sum([(yi - y_means[i])**2 for i, x in enumerate(unique_X) for yi in y[X == x]])
    
    # Graus de liberdade
    gl_lof = n - 2
    gl_ep = N - n
    gl_error = N - 2
    
    # Quadrados médios
    qm_reg = basic_stats['sqreg']
    qm_lof = sqlof / gl_lof if gl_lof > 0 else 0
    qm_ep = sqep / gl_ep if gl_ep > 0 else 0
    qm_error = basic_stats['sqerror'] / gl_error
    
    # Estatísticas F
    f_reg = qm_reg / qm_error
    f_lof = qm_lof / qm_ep if qm_ep > 0 else 0
    
    # P-valores
    p_reg = 1 - f_distribution.cdf(f_reg, 1, gl_error)
    p_lof = 1 - f_distribution.cdf(f_lof, gl_lof, gl_ep) if gl_lof > 0 and qm_ep > 0 else 1
    
    # Combina as estatísticas básicas com as adicionais
    return {
        **basic_stats,
        'X_unique': unique_X,
        'y_means': y_means,
        'n_per_level': n_per_level,
        'sqtrat': sqtrat,
        'sqlof': sqlof,
        'sqep': sqep,
        'gl_lof': gl_lof,
        'gl_ep': gl_ep,
        'qm_reg': qm_reg,
        'qm_lof': qm_lof,
        'qm_ep': qm_ep,
        'qm_error': qm_error,
        'f_reg': f_reg,
        'f_lof': f_lof,
        'p_reg': p_reg,
        'p_lof': p_lof
    }

def page_with_repetition():
    st.title('Regressão Linear com Repetição')
    
    # Entrada de dados
    st.sidebar.header('Entrada de Dados')
    data_input_method = st.sidebar.radio(
        "Método de entrada:",
        ['Usar dados de exemplo', 'Inserir dados manualmente', 'Importar CSV']
    )
    
    # Initialize X and y
    X = None
    y = None
    
    if data_input_method == 'Usar dados de exemplo':
        X = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        y = np.array([2.1, 2.3, 2.2, 3.8, 4.2, 4.0, 6.1, 5.9, 6.0, 7.8, 8.2, 8.0])
    elif data_input_method == 'Inserir dados manualmente':
        default_data = pd.DataFrame({
            'X': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            'Y': [2.1, 2.3, 2.2, 3.8, 4.2, 4.0, 6.1, 5.9, 6.0, 7.8, 8.2, 8.0]
        })
        edited_data = st.sidebar.data_editor(default_data)
        X = edited_data['X'].values
        y = edited_data['Y'].values
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
    
    # Seleção dos quadrados
    show_squares = st.sidebar.multiselect(
        'Mostrar Quadrados:',
        ['SQT', 'SQReg', 'SQLof', 'SQTrat', 'SQEP', 'SQE'],
        default=['SQT', 'SQReg', 'SQE']
    )
    
    # Row 1 - Gráfico
    st.subheader('Visualização')
    fig = plot_with_repetition(X, y, stats, show_squares)
    st.plotly_chart(fig, use_container_width=True)
    
    # Row 2 - Fórmulas e Explicações
    st.subheader('Fórmulas e Componentes da ANOVA')
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.latex(r'''
        \begin{align*}
        \text{SQT} &= \sum_{i=1}^a \sum_{j=1}^{n_i} (y_{ij} - \bar{y})^2 \\
        \text{SQReg} &= \sum_{i=1}^a \sum_{j=1}^{n_i} (\hat{y_i} - \bar{y})^2 \\
        \text{SQTrat} &= \sum_{i=1}^a n_i(\bar{y_i} - \bar{y})^2 \\
        \text{SQLof} &= \text{SQReg} - \text{SQTrat} \\
        \text{SQEP} &= \sum_{i=1}^a \sum_{j=1}^{n_i} (y_{ij} - \bar{y_i})^2 \\
        \text{SQE} &= \text{SQLof} + \text{SQEP}
        \end{align*}
        ''')
    
    with col2:
        st.markdown("""
        **Componentes:**
        - **SQT (Vermelho)**: Soma de Quadrados Total
        - **SQReg (Verde)**: Soma de Quadrados da Regressão
        - **SQLof (Roxo)**: Soma de Quadrados da Falta de Ajuste
        - **SQTrat (Laranja)**: Soma de Quadrados de Tratamentos
        - **SQEP (Rosa)**: Soma de Quadrados do Erro Puro
        - **SQE (Azul)**: Soma de Quadrados do Erro (SQLof + SQEP)
        """)
    
    # Row 3 - Resultados
    st.subheader('Resultados')
    col1, col2 = st.columns([2, 1])

    with col1:
        # Tabela ANOVA
        anova_data = pd.DataFrame({
            'Fonte': ['Regressão', 'Falta de Ajuste', 'Erro Puro', 'Error Total', 'Total'],
            'GL': [
                stats['gl_reg'], 
                stats['gl_lof'],
                stats['gl_ep'],
                stats['gl_error'],
                stats['gl_total']
            ],
            'SQ': [
                f"{stats['sqreg']:.3f}",
                f"{stats['sqlof']:.3f}",
                f"{stats['sqep']:.3f}",
                f"{stats['sqerror']:.3f}",
                f"{stats['sqt']:.3f}"
            ],
            'QM': [
                f"{stats['qm_reg']:.3f}",
                f"{stats['qm_lof']:.3f}",
                f"{stats['qm_ep']:.3f}",
                f"{stats['qm_error']:.3f}",
                ''
            ],
            'F': [
                f"{stats['f_reg']:.3f}",
                f"{stats['f_lof']:.3f}",
                '',
                '',
                ''
            ],
            'p-valor': [
                f"{stats['p_reg']:.3f}",
                f"{stats['p_lof']:.3f}",
                '',
                '',
                ''
            ]
        })
        st.dataframe(anova_data, use_container_width=True)

    with col2:
        st.subheader('Estatísticas do Modelo')
        st.write(f"R² = {stats['r2']:.4f}")
        st.write(f"Coeficiente Angular (β₁) = {stats['model'].coef_[0]:.4f}")
        st.write(f"Intercepto (β₀) = {stats['model'].intercept_:.4f}")

if __name__ == "__main__":
    st.set_page_config(page_title="Regressão com Repetição", layout="wide")
    page_with_repetition()