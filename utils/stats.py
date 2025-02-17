import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import f as f_distribution
def calculate_basic_stats(X, y):
    """Calcula estatísticas básicas para o modelo"""
    model = LinearRegression()
    X_reshaped = X.reshape(-1, 1)
    model.fit(X_reshaped, y)
    y_pred = model.predict(X_reshaped)
    y_mean = np.mean(y)
    
    # Cálculos básicos
    n = len(X)
    gl_reg = 1  # Para regressão linear simples
    gl_error = n - 2  # n - (p + 1), onde p é o número de variáveis preditoras
    gl_total = n - 1
    
    # Somas de quadrados
    sqt = np.sum((y - y_mean) ** 2)
    sqreg = np.sum((y_pred - y_mean) ** 2)
    sqerror = np.sum((y - y_pred) ** 2)
    
    # Quadrados médios
    qm_reg = sqreg / gl_reg
    qm_error = sqerror / gl_error
    
    # Estatística F e p-valor
    f_stat = qm_reg / qm_error
    p_value = 1 - f_distribution.cdf(f_stat, gl_reg, gl_error)
    
    # R²
    r2 = sqreg / sqt
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_mean': y_mean,
        'sqt': sqt,
        'sqreg': sqreg,
        'sqerror': sqerror,
        'gl_reg': gl_reg,
        'gl_error': gl_error,
        'gl_total': gl_total,
        'qm_reg': qm_reg,
        'qm_error': qm_error,
        'f_stat': f_stat,
        'p_value': p_value,
        'r2': r2
    }