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

def calculate_repeated_measures_stats(X, y):
    """Calcula estatísticas para regressão com repetição"""
    # Estatísticas básicas
    model = LinearRegression()
    X_reshaped = X.reshape(-1, 1)
    model.fit(X_reshaped, y)
    y_pred = model.predict(X_reshaped)
    y_mean = np.mean(y)
    
    # Identificação dos níveis únicos e médias por nível
    X_unique = np.unique(X)
    n = len(X_unique)  # número de níveis (a)
    N = len(X)         # número total de observações
    
    # Médias por nível e contagem de repetições
    y_means = np.array([np.mean(y[X == x]) for x in X_unique])
    n_per_level = np.array([np.sum(X == x) for x in X_unique])
    
    # Vetor de médias repetido para cada observação
    y_means_repeated = np.array([np.mean(y[X == xi]) for xi in X])
    
    # Cálculos das somas de quadrados
    sqt = np.sum((y - y_mean) ** 2)
    sqreg = np.sum((y_pred - y_mean) ** 2)
    sqtrat = np.sum(n_per_level * (y_means - y_mean)**2)
    sqlof = sqreg - sqtrat
    sqep = np.sum((y - y_means_repeated)**2)
    sqerror = np.sum((y - y_pred) ** 2)
    
    # Graus de liberdade
    gl_reg = 1
    gl_lof = n - 2
    gl_ep = N - n
    gl_error = N - 2
    gl_total = N - 1
    
    # Quadrados médios
    qm_reg = sqreg / gl_reg if gl_reg > 0 else 0
    qm_lof = sqlof / gl_lof if gl_lof > 0 else 0
    qm_ep = sqep / gl_ep if gl_ep > 0 else 0
    qm_error = sqerror / gl_error if gl_error > 0 else 0
    
    # Estatísticas F
    f_reg = qm_reg / qm_error if qm_error > 0 else 0
    f_lof = qm_lof / qm_ep if qm_ep > 0 else 0
    
    # P-valores
    p_reg = 1 - f_distribution.cdf(f_reg, gl_reg, gl_error)
    p_lof = 1 - f_distribution.cdf(f_lof, gl_lof, gl_ep) if gl_lof > 0 and qm_ep > 0 else 1
    
    # R²
    r2 = sqreg / sqt
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_mean': y_mean,
        'X_unique': X_unique,
        'y_means': y_means,
        'y_means_repeated': y_means_repeated,
        'n_per_level': n_per_level,
        'sqt': sqt,
        'sqreg': sqreg,
        'sqtrat': sqtrat,
        'sqlof': sqlof,
        'sqep': sqep,
        'sqerror': sqerror,
        'gl_reg': gl_reg,
        'gl_lof': gl_lof,
        'gl_ep': gl_ep,
        'gl_error': gl_error,
        'gl_total': gl_total,
        'qm_reg': qm_reg,
        'qm_lof': qm_lof,
        'qm_ep': qm_ep,
        'qm_error': qm_error,
        'f_reg': f_reg,
        'f_lof': f_lof,
        'p_reg': p_reg,
        'p_lof': p_lof,
        'r2': r2
    }