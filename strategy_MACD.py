import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from pydantic import BaseModel
import numpy as np

# 1. 參數模型
class StrategyParams(BaseModel):
    ticker: str = "2330.TW"
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    sl_pct: float = 0.05
    tp_pct: float = 0.10
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # 新增均線參數
    ma1_period: int = 5
    ma2_period: int = 10
    ma3_period: int = 20

# 2. 策略邏輯 (維持黃金交叉邏輯)
class MyMacdStrategy(Strategy):
    fast = 12
    slow = 26
    signal = 9
    sl_pct = 0.05
    tp_pct = 0.10
    
    def init(self):
        close = pd.Series(self.data.Close)
        ema_f = close.ewm(span=self.fast, adjust=False).mean()
        ema_s = close.ewm(span=self.slow, adjust=False).mean()
        
        self.dif = self.I(lambda: ema_f - ema_s, name='DIF')
        self.dea = self.I(lambda: pd.Series(self.dif).ewm(span=self.signal, adjust=False).mean(), name='DEA')
        self.hist = self.I(lambda: self.dif - self.dea, name='HIST')

    def next(self):
        price = self.data.Close[-1]
        dif_curr = self.dif[-1]
        dif_prev = self.dif[-2]
        dea_curr = self.dea[-1]
        dea_prev = self.dea[-2]

        # 黃金交叉 (買進)
        if not self.position:
            if dif_prev < dea_prev and dif_curr > dea_curr:
                self.buy(sl=price*(1-self.sl_pct), tp=price*(1+self.tp_pct))

        # 死亡交叉 (賣出)
        else:
            if dif_prev > dea_prev and dif_curr < dea_curr:
                self.position.close()

# 3. 執行分析
def run_simple_analysis(params: StrategyParams):
    df = yf.download(params.ticker, start=params.start_date, end=params.end_date, auto_adjust=True)
    if df.empty: raise ValueError(f"找不到 {params.ticker} 資料")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    bt = Backtest(df, MyMacdStrategy, cash=100_000, commission=.002)
    stats = bt.run(
        fast=params.macd_fast, slow=params.macd_slow, signal=params.macd_signal,
        sl_pct=params.sl_pct, tp_pct=params.tp_pct
    )

    # 整理交易數據
    trades_df = stats['_trades']
    trade_markers = []
    trade_list = []
    if not trades_df.empty:
        for i, row in trades_df.iterrows():
            trade_markers.append({"date": str(row['EntryTime'].date()), "price": row['EntryPrice'], "type": "buy"})
            trade_markers.append({"date": str(row['ExitTime'].date()), "price": row['ExitPrice'], "type": "sell"})
            trade_list.append({
                "entry_date": str(row['EntryTime'].date()),
                "entry_price": round(row['EntryPrice'], 2),
                "exit_date": str(row['ExitTime'].date()),
                "exit_price": round(row['ExitPrice'], 2),
                "pl": round(row['PnL'], 0),
                "return": round(row['ReturnPct'] * 100, 2),
                "duration": str(row['Duration'].days) + " 天"
            })

    # 計算 MACD 與 均線數據 供前端繪圖
    close = df['Close']
    
    # MACD
    ema_f = close.ewm(span=params.macd_fast, adjust=False).mean()
    ema_s = close.ewm(span=params.macd_slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=params.macd_signal, adjust=False).mean()
    hist = dif - dea

    # 均線 (MA)
    ma1 = close.rolling(window=params.ma1_period).mean()
    ma2 = close.rolling(window=params.ma2_period).mean()
    ma3 = close.rolling(window=params.ma3_period).mean()

    sharpe = stats['Sharpe Ratio']
    sharpe = round(float(sharpe), 2) if not (np.isnan(sharpe) or np.isinf(sharpe)) else 0.0
    
    result = {
        "return": round(float(stats['Return [%]']), 2),
        "mdd": round(float(stats['Max. Drawdown [%]']), 2),
        "sharpe": sharpe,
        "win_rate": round(float(stats['Win Rate [%]']), 2),
        "pl_ratio": round(float(stats['Profit Factor']), 2) if not np.isnan(stats['Profit Factor']) else 0.0,
        "trades": int(stats['# Trades']),
        "equity": stats['_equity_curve']['Equity'].apply(float).tolist()
    }

    return {
        "time": [str(d.date()) for d in df.index],
        "open": df['Open'].tolist(),
        "high": df['High'].tolist(),
        "low": df['Low'].tolist(),
        "close": df['Close'].tolist(),
        # 回傳均線數據 (處理 NaN 為 None 以便前端不繪製)
        "ma1": ma1.where(pd.notnull(ma1), None).tolist(),
        "ma2": ma2.where(pd.notnull(ma2), None).tolist(),
        "ma3": ma3.where(pd.notnull(ma3), None).tolist(),
        
        "macd_dif": dif.replace({np.nan: None}).tolist(),
        "macd_dea": dea.replace({np.nan: None}).tolist(),
        "macd_hist": hist.replace({np.nan: None}).tolist(),
        "stats": result,
        "trade_markers": trade_markers,
        "trade_list": trade_list
    }