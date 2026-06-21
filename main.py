# pip install pandas numpy matplotlib apimoex requests tensorflow xgboost scikit-learn scipy

"""Либы

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import apimoex
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBRegressor
from scipy.optimize import minimize

import warnings
warnings.filterwarnings('ignore')

"""Загрузка котировок по тикерам"""

tickers = ['SBER', 'GAZP', 'LKOH', 'GMKN', 'ROSN']
start_date = '2015-01-01'
end_date = '2025-12-31'

def get_moex_data(ticker, start, end):
    with requests.Session() as session:
        data = apimoex.get_board_history(session, ticker, start, end, board='TQBR')
        df = pd.DataFrame(data)
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
        df.set_index('TRADEDATE', inplace=True)
        return df['CLOSE']

prices = pd.DataFrame()
for ticker in tickers:
    print(f"Загрузка данных {ticker}...")
    prices[ticker] = get_moex_data(ticker, start_date, end_date)

# Очистка от пропусков (Forward fill) и расчет доходностей
prices = prices.fillna(method='ffill').dropna()
returns = np.log(prices / prices.shift(1)).dropna()

print("Размерность данных о доходностях:", returns.shape)

"""Подготовка выборок"""

split_date = '2021-12-31'
train_ret = returns.loc[:split_date]
test_ret = returns.loc[split_date:] # Включаем сам split_date для создания лагов

window_size = 60 # Смотрим на 60 дней назад

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Словари для хранения моделей и прогнозов
lstm_models = {}
xgb_models = {}
scalers = {}

lstm_predictions = pd.DataFrame(index=test_ret.index[window_size:], columns=tickers)
xgb_predictions = pd.DataFrame(index=test_ret.index[window_size:], columns=tickers)

"""Обучение моделей"""

for ticker in tickers:
    print(f"\nОбучение моделей для {ticker}...")

    # Масштабирование
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_ret[[ticker]])
    test_scaled = scaler.transform(test_ret[[ticker]])
    scalers[ticker] = scaler

    # Подготовка X и y
    X_train, y_train = create_sequences(train_scaled, window_size)
    X_test, y_test = create_sequences(test_scaled, window_size)

    # --- LSTM ---
    lstm = Sequential()
    lstm.add(LSTM(50, return_sequences=False, input_shape=(window_size, 1)))
    lstm.add(Dropout(0.2))
    lstm.add(Dense(1))
    lstm.compile(optimizer='adam', loss='mse')

    # Обучаем
    lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    lstm_models[ticker] = lstm

    # Прогноз LSTM
    pred_lstm_scaled = lstm.predict(X_test, verbose=0)
    lstm_predictions[ticker] = scaler.inverse_transform(pred_lstm_scaled).flatten()

    # --- XGBoost ---
    # Для XGBoost нужно 2D измерение: (samples, window_size)
    X_train_xgb = X_train.reshape((X_train.shape[0], X_train.shape[1]))
    X_test_xgb = X_test.reshape((X_test.shape[0], X_test.shape[1]))

    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, objective='reg:squarederror')
    xgb.fit(X_train_xgb, y_train)
    xgb_models[ticker] = xgb

    # Прогноз XGBoost
    pred_xgb_scaled = xgb.predict(X_test_xgb).reshape(-1, 1)
    xgb_predictions[ticker] = scaler.inverse_transform(pred_xgb_scaled).flatten()

print("\nОбучение завершено!")

"""Функция оптимизации (Mean-Variance)"""

def get_optimal_weights(expected_returns, cov_matrix, risk_free_rate=0.0):
    num_assets = len(expected_returns)

    # Целевая функция: максимизация Шарпа (минимизация отрицательного Шарпа)
    def neg_sharpe(weights):
        port_return = np.sum(expected_returns * weights)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - risk_free_rate) / port_volatility

    # Ограничения: сумма весов = 1, веса >= 0 (long only)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Начальное приближение (равномерный портфель)
    init_guess = num_assets * [1. / num_assets, ]

    opt_result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return opt_result.x
# Ячейка 6: Бэктестинг
test_dates = lstm_predictions.index
portfolio_returns = {'Markowitz (Hist)': [], 'LSTM': [], 'XGBoost': [], 'Equal Weight': []}

# Для расчета ковариации берем скользящее окно истории (например, 252 дня)
hist_window = 252

for date in test_dates:
    # 1. Историческая ковариация на момент времени t
    hist_data = returns.loc[:date].iloc[-hist_window - 1:-1]
    cov_matrix = hist_data.cov().values

    # 2. Ожидаемые доходности (mu) для разных стратегий
    mu_hist = hist_data.mean().values  # Историческая средняя
    mu_lstm = lstm_predictions.loc[date].values
    mu_xgb = xgb_predictions.loc[date].values

    # 3. Получаем веса
    w_hist = get_optimal_weights(mu_hist, cov_matrix)
    w_lstm = get_optimal_weights(mu_lstm, cov_matrix)
    w_xgb = get_optimal_weights(mu_xgb, cov_matrix)
    w_equal = np.array([1 / len(tickers)] * len(tickers))

    # 4. Фактическая доходность на шаге t
    actual_returns = returns.loc[date].values

    # 5. Считаем доходность портфелей
    portfolio_returns['Markowitz (Hist)'].append(np.sum(w_hist * actual_returns))
    portfolio_returns['LSTM'].append(np.sum(w_lstm * actual_returns))
    portfolio_returns['XGBoost'].append(np.sum(w_xgb * actual_returns))
    portfolio_returns['Equal Weight'].append(np.sum(w_equal * actual_returns))

# Переводим в DataFrame
port_ret_df = pd.DataFrame(portfolio_returns, index=test_dates)

# Расчет кумулятивной доходности
cum_returns = np.exp(port_ret_df.cumsum())

# Визуализация
plt.figure(figsize=(12, 6))
for col in cum_returns.columns:
    plt.plot(cum_returns.index, cum_returns[col], label=col)
plt.title('Сравнение накопленной доходности портфелей (Out-of-Sample: 2022-2026)')
plt.xlabel('Дата')
plt.ylabel('Накопленная доходность (1 = 100%)')
plt.legend()
plt.grid(True)
plt.show()

# Расчет метрик (Шарп)
sharpe_ratios = (port_ret_df.mean() / port_ret_df.std()) * np.sqrt(252)
print("\nКоэффициенты Шарпа (годовые):")
print(sharpe_ratios)

"""**Loss Function**"""

import seaborn as sns

# 1. График коэффициентов Шарпа (Столбчатая диаграмма)
plt.figure(figsize=(10, 6))
colors = ['steelblue', 'darkorange', 'forestgreen', 'firebrick']
bars = plt.bar(sharpe_ratios.index, sharpe_ratios.values, color=colors, edgecolor='black')

plt.axhline(0, color='black', linewidth=1) # Линия нуля
plt.title('Сравнение риск-скорректированной доходности (Коэффициент Шарпа)', fontsize=14)
plt.ylabel('Коэффициент Шарпа (годовой)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Добавление значений над столбцами
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (0.02 if yval > 0 else -0.05),
             round(yval, 3), ha='center', va='bottom' if yval > 0 else 'top', fontsize=11, fontweight='bold')
plt.show()

# 2. График функции потерь (Loss Function) для одной из акций (например, SBER)
# Для этого нам нужно немного переобучить одну сеть, чтобы сохранить историю
# (в основном коде мы не сохраняли history для экономии памяти).
ticker_for_plot = 'SBER'
print(f"Генерация графика Loss для {ticker_for_plot}...")

# Берем данные из памяти прошлого шага
X_train_plot, y_train_plot = create_sequences(scalers[ticker_for_plot].transform(train_ret[[ticker_for_plot]]), window_size)

model_plot = Sequential()
model_plot.add(LSTM(50, return_sequences=False, input_shape=(window_size, 1)))
model_plot.add(Dropout(0.2))
model_plot.add(Dense(1))
model_plot.compile(optimizer='adam', loss='mse')

# Обучаем с выделением валидационной выборки
history = model_plot.fit(X_train_plot, y_train_plot, epochs=30, batch_size=32, validation_split=0.1, verbose=0)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss (Обучающая)', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss (Валидационная)', linewidth=2)
plt.title(f'Кривая обучения LSTM-сети (Функция потерь MSE) - {ticker_for_plot}', fontsize=14)
plt.xlabel('Эпоха (Epoch)', fontsize=12)
plt.ylabel('Ошибка (Mean Squared Error)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()