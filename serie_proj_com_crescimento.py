import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import statsmodels.api as sm
from datetime import datetime
from statsmodels.graphics.tsaplots import month_plot, plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.arima.model import ARIMA 
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import jarque_bera
import scipy.stats as sct
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Ignorar avisos
warnings.filterwarnings("ignore")

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ==========================================
# 1. Carregamento e Preparação dos Dados
# ==========================================
print("Carregando dados...")
try:
    carteira = pd.read_excel('proj.xlsx')
except FileNotFoundError:
    print("ERRO: O arquivo 'proj.xlsx' não foi encontrado.")
    exit()

# Indexando pela data
carteira = carteira.set_index('date')

# --- TRANSFORMAÇÃO LOGARÍTMICA (Mantida para precisão de crescimento) ---
carteira['log_valor'] = np.log(carteira['valor_carteira'])

# Ajuste do índice de datas
datas = pd.date_range(start=datetime(2023, 1, 1), end=datetime(2025, 12, 1), freq='MS')
carteira_full = pd.DataFrame(data=carteira['log_valor'].values, index=datas, columns=['log_valor'])
carteira_full['valor_real'] = carteira['valor_carteira'].values

print("\n--- Dados Carregados ---")
print(carteira_full.head())

# GRÁFICO 1: Série Original vs Log
fig, ax = plt.subplots(2, 1, figsize=(15, 8))
carteira_full['valor_real'].plot(ax=ax[0], title='Série Original (Reais)')
carteira_full['log_valor'].plot(ax=ax[1], title='Série Transformada (Log) - Note a linearização', color='orange')
plt.tight_layout()
plt.show()

# ==========================================
# 2. Decomposição da Série (Aditiva no Log)
# ==========================================
print("\nRealizando decomposição...")
# Decomposição Aditiva no Log equivale a Multiplicativa no Real
result = seasonal_decompose(carteira_full['log_valor'], model='additive')

# GRÁFICO 2: Decomposição Completa
result.plot()
plt.show()

# GRÁFICO 3: Sazonalidade e Resíduos Detalhados
plt.figure(figsize=(15, 6))
result.seasonal.plot(legend=True, label='Sazonalidade')
result.resid.plot(legend=True, label='Resíduos')
plt.title('Componentes: Sazonalidade e Resíduos (Escala Log)')
plt.show()

# GRÁFICO 4: Tendência Isolada
plt.figure(figsize=(15, 6))
result.trend.plot(legend=True, color='r', label='Tendência')
plt.title('Tendência Extraída (Trend)')
plt.show()

# Criando a série dessazonalizada
carteira_full['sazonalidade_log'] = result.seasonal.values
carteira_full['dessazon_log'] = carteira_full['log_valor'] - carteira_full['sazonalidade_log']

# ==========================================
# 3. Análise de Estacionariedade (ACF/PACF)
# ==========================================
print("\n--- Analisando Estacionariedade e Diferenciação ---")

# Diferenciação (necessária para ARIMA capturar crescimento)
diff_log = diff(carteira_full['dessazon_log'], k_diff=1)

# GRÁFICO 5: Série Diferenciada
plt.figure(figsize=(15, 6))
diff_log.plot(title="Série Dessazonalizada e Diferenciada (1ª ordem)")
plt.show()

# GRÁFICO 6: ACF e PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(diff_log.dropna(), lags=15, ax=ax1, title='Autocorrelação (ACF) - Ajuda a definir Q')
plot_pacf(diff_log.dropna(), lags=15, ax=ax2, title='Autocorrelação Parcial (PACF) - Ajuda a definir P')
plt.tight_layout()
plt.show()

# ==========================================
# 4. Grid Search Otimizado (Trend + Log)
# ==========================================
p = [0, 1, 2]
d = [1] 
q = [0, 1, 2]
trend_params = ['c', 't'] # Força o modelo a buscar crescimento (constante ou linear)

pdq_combinations = list(itertools.product(p, d, q, trend_params))

print("\n--- Buscando melhor modelo... ---")
melhor_aic = float('inf')
melhor_param = None
melhor_trend = None

for param in pdq_combinations:
    order_param = (param[0], param[1], param[2])
    trend_param = param[3]
    try:
        model = ARIMA(carteira_full['dessazon_log'], order=order_param, trend=trend_param)
        model_fit = model.fit()
        if model_fit.aic < melhor_aic:
            melhor_aic = model_fit.aic
            melhor_param = order_param
            melhor_trend = trend_param
    except:
        continue

print(f"\nModelo Vencedor: ARIMA{melhor_param} trend='{melhor_trend}' (AIC: {melhor_aic:.2f})")

# Ajustando o modelo final
modelo_final = ARIMA(carteira_full['dessazon_log'], order=melhor_param, trend=melhor_trend).fit()
print(modelo_final.summary())

# ==========================================
# 5. Diagnóstico de Resíduos (Gráficos recuperados)
# ==========================================
print("\n--- Diagnóstico Visual dos Resíduos ---")
residuals = pd.DataFrame(modelo_final.resid)

# GRÁFICO 7: Painel de Diagnóstico (4 gráficos)
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# 1. Resíduos ao longo do tempo
residuals.plot(title="Resíduos ao longo do tempo", ax=ax[0][0], legend=False)

# 2. Histograma e Densidade (KDE)
sns.histplot(residuals, kde=True, ax=ax[0][1])
ax[0][1].set_title("Histograma de Densidade (Curva de Sino?)")

# 3. ACF dos Resíduos (não deve ter padrão)
plot_acf(residuals, lags=15, ax=ax[1][0], title="ACF dos Resíduos")

# 4. QQ Plot (Normalidade)
qqplot(residuals[0], line='s', ax=ax[1][1])
ax[1][1].set_title("QQ Plot (Normalidade)")

plt.suptitle("Diagnóstico de Qualidade do Modelo", fontsize=16)
plt.show()

# Teste Jarque-Bera Final
jb_test = jarque_bera(modelo_final.resid)
print(f"\nTeste Jarque-Bera (Normalidade): Prob={jb_test[1]:.4f} (Ideal > 0.05)")

# ==========================================
# 6. Backtesting (Validação)
# ==========================================
qtd_teste = 12 
treino_log = carteira_full['dessazon_log'].iloc[:-qtd_teste]
teste_log = carteira_full['dessazon_log'].iloc[-qtd_teste:]

# Treina modelo de teste
modelo_backtest = ARIMA(treino_log, order=melhor_param, trend=melhor_trend).fit()

# Previsão
start = len(treino_log)
end = len(treino_log) + len(teste_log) - 1
pred_log = modelo_backtest.predict(start=start, end=end)

# Reconstrução (Log -> Real)
pred_com_sazonalidade_log = pred_log + carteira_full['sazonalidade_log'].iloc[-qtd_teste:].values
pred_valores_reais = np.exp(pred_com_sazonalidade_log)
y_true = carteira_full['valor_real'].iloc[-qtd_teste:]

# GRÁFICO 8: Backtesting
plt.figure(figsize=(15, 6))
plt.plot(carteira_full.index, carteira_full['valor_real'], label='Histórico Real', linewidth=2)
plt.plot(teste_log.index, pred_valores_reais, label='Previsão do Modelo (Backtest)', color='red', linestyle='--', linewidth=2)
plt.title(f'Teste de Validação (MAE: {mean_absolute_error(y_true, pred_valores_reais):,.2f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 7. Previsão Futura e Exportação
# ==========================================
forecast_steps = 12
forecast_log = modelo_final.predict(start=len(carteira_full), end=len(carteira_full)+forecast_steps-1)
sazonalidade_futura = carteira_full['sazonalidade_log'].iloc[-12:].values

# Reconstrução Final
forecast_final_log = forecast_log + sazonalidade_futura
forecast_final_reais = np.exp(forecast_final_log)

# Datas Futuras
future_dates = pd.date_range(start=carteira_full.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
forecast_series = pd.Series(forecast_final_reais.values, index=future_dates)

# GRÁFICO 9: Previsão Final
plt.figure(figsize=(15, 7))
# Histórico
plt.plot(carteira_full.index, carteira_full['valor_real'], label='Histórico Real', color='blue')
# Previsão
plt.plot(forecast_series.index, forecast_series.values, label='Previsão de Crescimento 2026', color='green', linewidth=3)
# Área de destaque
plt.axvspan(forecast_series.index[0], forecast_series.index[-1], color='green', alpha=0.1)

plt.title('Projeção de Expansão da Carteira (Otimizada)', fontsize=14)
plt.ylabel('Valor da Carteira (R$)')
plt.xlabel('Data')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Exportação
print("\n--- Exportando Resultados ---")
try:
    df_export = forecast_series.to_frame(name='valor_previsto')
    df_export.index.name = 'data_previsao'
    df_export = df_export.reset_index()
    
    nome_arquivo = 'resultado_crescimento.xlsx'
    df_export.to_excel(nome_arquivo, index=False)
    print(f"Arquivo '{nome_arquivo}' gerado com sucesso!")
    print(df_export.head())
except Exception as e:
    print(f"Erro na exportação: {e}")