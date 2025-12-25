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
    ma1_period: int = 5
    ma2_period: int = 10
    ma3_period: int = 20
    initial_cash: int = 100000
    commission_discount: float = 10.0
    # 修改：只保留模式與最大檔數
    trade_mode: str = "all"  # "all" or "fixed"
    max_pos: int = 1         # 最大同時持倉筆數 (預設1)

# 2. 策略邏輯
class MyMacdStrategy(Strategy):
    fast = 12
    slow = 26
    signal = 9
    sl_pct = 0.05
    tp_pct = 0.10
    trade_mode = "all"
    max_pos = 1
    
    def init(self):
        close = pd.Series(self.data.Close)
        ema_f = close.ewm(span=self.fast, adjust=False).mean()
        ema_s = close.ewm(span=self.slow, adjust=False).mean()
        self.dif = self.I(lambda: ema_f - ema_s, name='DIF')
        self.dea = self.I(lambda: pd.Series(self.dif).ewm(span=self.signal, adjust=False).mean(), name='DEA')
        self.hist = self.I(lambda: self.dif - self.dea, name='HIST')

    def next(self):
        dif = self.dif
        dea = self.dea
        price = self.data.Close[-1]
        
        if len(dif) < 2: return

        # 進場：黃金交叉
        if dif[-2] < dea[-2] and dif[-1] > dea[-1]:
            # ★★★ 資金管理邏輯 ★★★
            if self.trade_mode == "fixed":
                # 固定 1000 股模式
                # 檢查：目前的持倉筆數 < 設定的最大筆數，才准買
                if len(self.trades) < self.max_pos:
                    self.buy(size=1000, sl=price*(1-self.sl_pct), tp=price*(1+self.tp_pct))
            else:
                # 梭哈模式：不限制筆數(反正錢用完就買不了)，直接用剩餘資金買進
                # 這裡加個簡單判斷，如果已經有倉位就不再買(避免零錢加碼)，純粹做一筆 All in
                if len(self.trades) == 0:
                    self.buy(sl=price*(1-self.sl_pct), tp=price*(1+self.tp_pct))

        # 出場：死亡交叉 (全部平倉)
        if dif[-2] > dea[-2] and dif[-1] < dea[-1]:
            self.position.close()

# 3. 執行分析
def run_simple_analysis(params: StrategyParams):
    df = yf.download(params.ticker, start=params.start_date, end=params.end_date, auto_adjust=True)
    if df.empty: raise ValueError(f"找不到 {params.ticker} 資料")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    fee_rate = 0.001425 * (params.commission_discount / 10)
    tax_rate = 0.003
    approx_comm = fee_rate + (tax_rate / 2)

    MyMacdStrategy.trade_mode = params.trade_mode
    MyMacdStrategy.max_pos = params.max_pos

    bt = Backtest(df, MyMacdStrategy, cash=params.initial_cash, commission=approx_comm)
    stats = bt.run(
        fast=params.macd_fast, slow=params.macd_slow, signal=params.macd_signal,
        sl_pct=params.sl_pct, tp_pct=params.tp_pct
    )

    # 整理交易數據
    trades_df = stats['_trades']
    trade_markers = []
    trade_list = []
    current_equity = params.initial_cash

    if not trades_df.empty:
        for i, row in trades_df.iterrows():
            entry_price = row['EntryPrice']
            exit_price = row['ExitPrice']
            size = row['Size'] 

            cost_buy = entry_price * size * fee_rate
            cost_sell = exit_price * size * (fee_rate + tax_rate)
            total_cost = int(cost_buy + cost_sell)

            gross_profit = (exit_price - entry_price) * size
            net_pnl = int(gross_profit - total_cost)
            current_equity += net_pnl
            duration_days = row['Duration'].days

            trade_markers.append({"date": str(row['EntryTime'].date()), "price": entry_price, "type": "buy"})
            trade_markers.append({"date": str(row['ExitTime'].date()), "price": exit_price, "type": "sell"})
            
            trade_list.append({
                "entry_date": str(row['EntryTime'].date()),
                "entry_price": round(entry_price, 2),
                "exit_date": str(row['ExitTime'].date()),
                "exit_price": round(exit_price, 2),
                "duration": f"{duration_days} 天",
                "cost": total_cost,
                "pl": net_pnl,
                "return": round((net_pnl / (entry_price * size)) * 100, 2),
                "equity": int(current_equity)
            })

    # 指標與曲線
    close = df['Close']
    ma1 = close.rolling(window=params.ma1_period).mean()
    ma2 = close.rolling(window=params.ma2_period).mean()
    ma3 = close.rolling(window=params.ma3_period).mean()
    ema_f = close.ewm(span=params.macd_fast, adjust=False).mean()
    ema_s = close.ewm(span=params.macd_slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=params.macd_signal, adjust=False).mean()
    hist = dif - dea

    equity_curve = stats['_equity_curve']['Equity']
    if len(equity_curve) > len(df): equity_curve = equity_curve.iloc[-len(df):]
    
    result = {
        "return": round(float(stats['Return [%]']), 2),
        "mdd": round(float(stats['Max. Drawdown [%]']), 2),
        "sharpe": round(float(stats['Sharpe Ratio']), 2) if not np.isnan(stats['Sharpe Ratio']) else 0.0,
        "win_rate": round(float(stats['Win Rate [%]']), 2),
        "pl_ratio": round(float(stats['Profit Factor']), 2) if not np.isnan(stats['Profit Factor']) else 0.0,
        "trades": int(stats['# Trades']),
        "equity": equity_curve.apply(float).tolist()
    }

    return {
        "time": [str(d.date()) for d in df.index],
        "open": df['Open'].tolist(),
        "high": df['High'].tolist(),
        "low": df['Low'].tolist(),
        "close": df['Close'].tolist(),
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