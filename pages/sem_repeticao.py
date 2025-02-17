import streamlit as st
import numpy as np
import pandas as pd
from utils.plots import plot_no_repetition
from utils.stats import calculate_basic_stats
from sklearn.linear_model import LinearRegression

def page_no_repetition():
    st.title('Regressão Linear sem Repetição')
    
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
        X = np.array([160, 165, 170, 175, 180, 185])
        y = np.array([60, 65, 70, 75, 80, 85])
    elif data_input_method == 'Inserir dados manualmente':
        default_data = pd.DataFrame({
            'X': [160, 165, 170, 175, 180, 185],
            'Y': [60, 65, 70, 75, 80, 85]
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
    
    # Cálculos
    stats = calculate_basic_stats(X, y)
    
    # Seleção dos quadrados
    show_squares = st.sidebar.multiselect(
        'Mostrar Quadrados:',
        ['SQT', 'SQReg', 'SQE'],
        default=['SQT', 'SQReg', 'SQE']
    )
    
    # Row 1 - Gráfico
    st.subheader('Visualização')
    fig = plot_no_repetition(X, y, stats, show_squares)
    st.plotly_chart(fig, use_container_width=True)
    
    # Row 2 - Fórmulas e Explicações
    st.subheader('Fórmulas e Componentes')
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.latex(r'''
        \begin{align*}
        \underbrace{\sum_{i=1}^n (y_i - \bar{y})^2}_{\text{SQT}} &= 
        \underbrace{\sum_{i=1}^n (\hat{y_i} - \bar{y})^2}_{\text{SQReg}} + 
        \underbrace{\sum_{i=1}^n (y_i - \hat{y_i})^2}_{\text{SQE}}
        \end{align*}
        ''')
    
    with col2:
        st.markdown("""
        **Componentes:**
        - **SQT (Vermelho)**: Variação total dos dados
        - **SQReg (Verde)**: Variação explicada pela regressão
        - **SQE (Azul)**: Variação não explicada (erro)
        """)
    
    # Row 3 - Resultados
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
            ],
            'p-valor': [
                f"{stats['p_value']:.3f}",
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
        
        # Interpretação dos resultados
        alpha = 0.05
        if stats['p_value'] < alpha:
            st.success(f"Regressão significativa (p < {alpha})")
        else:
            st.warning(f"Regressão não significativa (p > {alpha})")

if __name__ == "__main__":
    st.set_page_config(page_title="Regressão sem Repetição", layout="wide")
    page_no_repetition()