import plotly.graph_objects as go
import numpy as np

def create_square_shapes(x, y1, y2, color, opacity=0.3):
    """Cria um quadrado perfeito onde a diferença define o lado"""
    diff = abs(y2 - y1)
    center_y = (y1 + y2) / 2  # Ponto central entre y1 e y2
    
    return dict(
        type='rect',
        x0=x,
        x1=x + diff,  # Largura igual à altura
        y0=center_y - diff/2,  # Centraliza o quadrado verticalmente
        y1=center_y + diff/2,
        fillcolor=color,
        opacity=opacity,
        layer='below',
        line_width=1,
        line=dict(color=color)
    )

def adjust_plot_range(X, y, margin_percent=0.1):
    """Ajusta o range do plot com margem percentual"""
    x_min, x_max = X.min(), X.max()
    y_min, y_max = y.min(), y.max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Usa o maior range para manter aspecto quadrado
    max_range = max(x_range, y_range)
    
    x_margin = max_range * margin_percent
    y_margin = max_range * margin_percent
    
    return {
        'x_min': x_min - x_margin,
        'x_max': x_max + x_margin,
        'y_min': y_min - y_margin,
        'y_max': y_max + y_margin
    }

def plot_no_repetition(X, y, stats, show_squares, x_name="X", y_name="Y"):
    """Plota gráfico para regressão sem repetição"""
    fig = go.Figure()
    
    # Calcula ranges ajustados
    plot_range = adjust_plot_range(X, y)
    
    # Configuração do grid
    fig.update_layout(
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
    )
    
    # Cores para cada tipo de quadrado (sem repetição)
    colors = {
        'SQT': '#ff0000',     # vermelho
        'SQReg': '#00ff00',   # verde
        'SQE': '#0000ff'      # azul
    }
    
    # Linha de regressão
    x_line = np.linspace(X.min() - 1, X.max() + 1, 100)
    y_line = stats['model'].predict(x_line.reshape(-1, 1))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode='lines',
        name='Linha de Regressão',
        line=dict(color='green', width=2)
    ))
    
    # Linha média
    fig.add_trace(go.Scatter(
        x=[X.min() - 1, X.max() + 1],
        y=[stats['y_mean'], stats['y_mean']],
        mode='lines',
        name='Média Geral',
        line=dict(color='gray', dash='dash')
    ))
    
    # Adicionar quadrados selecionados
    for sq_type in show_squares:
        if sq_type == 'SQT':
            for i in range(len(X)):
                fig.add_shape(create_square_shapes(
                    X[i], y[i], stats['y_mean'], colors[sq_type]
                ))
        elif sq_type == 'SQReg':
            for i in range(len(X)):
                fig.add_shape(create_square_shapes(
                    X[i], stats['y_pred'][i], stats['y_mean'], colors[sq_type]
                ))
        elif sq_type == 'SQE':
            for i in range(len(X)):
                fig.add_shape(create_square_shapes(
                    X[i], y[i], stats['y_pred'][i], colors[sq_type]
                ))
    
    # Pontos originais
    fig.add_trace(go.Scatter(
        x=X, y=y,
        mode='markers',
        name='Dados Originais',
        marker=dict(size=10, color='black')
    ))
    
    # Layout atualizado
    fig.update_layout(
        title='Visualização da Regressão Linear sem Repetição',
        xaxis_title=x_name,
        yaxis_title=y_name,
        showlegend=True,
        height=600,  # Aumenta altura do gráfico
        xaxis=dict(
            range=[plot_range['x_min'], plot_range['x_max']],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            range=[plot_range['y_min'], plot_range['y_max']],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
    )
    
    return fig

def plot_with_repetition(X, y, stats, show_squares, x_name="X", y_name="Y"):
    """Plota gráfico para regressão com repetição"""
    fig = go.Figure()
    
    # Calcula ranges ajustados
    plot_range = adjust_plot_range(X, y)
    
    # Configuração do grid
    fig.update_layout(
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
    )
    
    # Cores para cada tipo de quadrado (com repetição)
    colors = {
        'SQT': '#ff0000',     # vermelho
        'SQReg': '#00ff00',   # verde
        'SQErroReg': '#0000ff', # azul
        'SQTrat': '#808080',  # cinza
        'SQFA': '#800080',    # roxo
        'SQEP': '#8b4513'     # marrom
    }
    
    # Linha de regressão
    x_line = np.linspace(stats['X_unique'].min() - 1, stats['X_unique'].max() + 1, 100)
    y_line = stats['model'].predict(x_line.reshape(-1, 1))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode='lines',
        name='Linha de Regressão',
        line=dict(color='green', width=2)
    ))
    
    # Linha média geral
    fig.add_trace(go.Scatter(
        x=[X.min() - 1, X.max() + 1],
        y=[stats['y_mean'], stats['y_mean']],
        mode='lines',
        name='Média Geral',
        line=dict(color='gray', dash='dash')
    ))
    
    # Adicionar quadrados selecionados
    for sq_type in show_squares:
        if sq_type == 'SQT':
            for i in range(len(X)):
                fig.add_shape(create_square_shapes(
                    X[i], y[i], stats['y_mean'], colors[sq_type]
                ))
        elif sq_type == 'SQReg':
            for i in range(len(X)):
                fig.add_shape(create_square_shapes(
                    X[i], stats['y_pred'][i], stats['y_mean'], colors[sq_type]
                ))
        elif sq_type == 'SQErroReg':
            for i in range(len(X)):
                fig.add_shape(create_square_shapes(
                    X[i], y[i], stats['y_pred'][i], colors[sq_type]
                ))
        elif sq_type == 'SQTrat':
            for i, x in enumerate(stats['X_unique']):
                y_mean_i = stats['y_means'][i]
                fig.add_shape(create_square_shapes(
                    x, y_mean_i, stats['y_mean'], colors[sq_type]
                ))
        elif sq_type == 'SQFA':
            for i, x in enumerate(stats['X_unique']):
                y_mean_i = stats['y_means'][i]
                y_pred_i = stats['model'].predict([[x]])[0]
                fig.add_shape(create_square_shapes(
                    x, y_mean_i, y_pred_i, colors[sq_type]
                ))
        elif sq_type == 'SQEP':
            for i in range(len(X)):
                fig.add_shape(create_square_shapes(
                    X[i], y[i], stats['y_means_repeated'][i], colors[sq_type]
                ))
    
    # Pontos originais
    fig.add_trace(go.Scatter(
        x=X, y=y,
        mode='markers',
        name='Dados Originais',
        marker=dict(size=10, color='black')
    ))
    
    # Médias por tratamento
    fig.add_trace(go.Scatter(
        x=stats['X_unique'],
        y=stats['y_means'],
        mode='markers',
        name='Médias por Tratamento',
        marker=dict(size=12, color='red', symbol='star')
    ))
    
    # Layout atualizado
    fig.update_layout(
        title='Visualização da Regressão Linear com Repetição',
        xaxis_title=x_name,
        yaxis_title=y_name,
        showlegend=True,
        height=600,  # Aumenta altura do gráfico
        xaxis=dict(
            range=[plot_range['x_min'], plot_range['x_max']],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            range=[plot_range['y_min'], plot_range['y_max']],
            scaleanchor="x",
            scaleratio=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
    )
    
    return fig
