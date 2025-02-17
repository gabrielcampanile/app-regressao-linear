import plotly.graph_objects as go
import numpy as np

def create_square_shapes(x, y1, y2, color, opacity=0.3):
    """Cria quadrado onde a diferença define tanto altura quanto largura"""
    diff = abs(y2 - y1)
    return dict(
        type='rect',
        x0=x,
        x1=x + diff,
        y0=min(y1, y2),
        y1=min(y1, y2) + diff,
        fillcolor=color,
        opacity=opacity,
        layer='below',
        line_width=1,
        line=dict(color=color)
    )

def plot_no_repetition(X, y, stats, show_squares):
    """Plota gráfico para regressão sem repetição"""
    fig = go.Figure()
    
    # Configuração do grid
    fig.update_layout(
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
    )
    
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
        name='Média de Y',
        line=dict(color='gray', dash='dash')
    ))
    
    # Adicionar quadrados selecionados
    colors = {'SQT': 'red', 'SQReg': 'green', 'SQE': 'blue'}
    
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
    
    fig.update_layout(
        title='Visualização da Regressão Linear sem Repetição',
        xaxis_title='X',
        yaxis_title='Y',
        showlegend=True
    )
    
    return fig

def plot_with_repetition(X, y, stats, show_squares):
    """Plota gráfico para regressão com repetição"""
    fig = go.Figure()
    
    # Configuração do grid
    fig.update_layout(
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
    )
    
    # Linha de regressão
    x_line = np.linspace(stats['X_unique'].min() - 1, stats['X_unique'].max() + 1, 100)
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
        name='Média de Y',
        line=dict(color='gray', dash='dash')
    ))
    
    # Cores para cada tipo de quadrado
    colors = {
        'SQReg': 'green',
        'SQLof': 'purple',
        'SQTrat': 'orange',
        'SQEP': 'pink',
        'SQE': 'blue',
        'SQT': 'red'
    }
    
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
        elif sq_type == 'SQLof':
            for i, x in enumerate(stats['X_unique']):
                y_mean_i = stats['y_means'][i]
                y_pred_i = stats['model'].predict([[x]])[0]
                fig.add_shape(create_square_shapes(
                    x, y_mean_i, y_pred_i, colors[sq_type]
                ))
        elif sq_type == 'SQTrat':
            for i, x in enumerate(stats['X_unique']):
                y_mean_i = stats['y_means'][i]
                fig.add_shape(create_square_shapes(
                    x, y_mean_i, stats['y_mean'], colors[sq_type]
                ))
        elif sq_type == 'SQEP':
            for i in range(len(X)):
                fig.add_shape(create_square_shapes(
                    X[i], y[i], stats['y_means_repeated'][i], colors[sq_type]
                ))
        elif sq_type == 'SQE':
            for i in range(len(X)):
                fig.add_shape(create_square_shapes(
                    X[i], y[i], stats['y_pred'][i], colors[sq_type]
                ))
    
    # Pontos originais e médias
    fig.add_trace(go.Scatter(
        x=X, y=y,
        mode='markers',
        name='Dados Originais',
        marker=dict(size=10, color='black')
    ))
    
    fig.add_trace(go.Scatter(
        x=stats['X_unique'],
        y=stats['y_means'],
        mode='markers',
        name='Médias por Tratamento',
        marker=dict(size=12, color='red', symbol='star')
    ))
    
    fig.update_layout(
        title='Visualização da Regressão Linear com Repetição',
        xaxis_title='X',
        yaxis_title='Y',
        showlegend=True
    )
    
    return fig
