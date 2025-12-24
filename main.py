import os
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from strategy_MACD import run_simple_analysis, StrategyParams
import uvicorn

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def read_root(
    request: Request, 
    ticker: str = "2330.TW", 
    start: str = None, 
    end: str = None, 
    sl: float = 0.05,
    tp: float = 0.10,
    m_fast: int = 12,
    m_slow: int = 26,
    m_sig: int = 9,
    # 新增均線參數
    ma1: int = 5,
    ma2: int = 10,
    ma3: int = 20
):
    try:
        if end is None:
            yesterday = datetime.now() - timedelta(days=1)
            end = yesterday.strftime("%Y-%m-%d")
        
        if start is None:
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            start = (end_dt - timedelta(days=365)).strftime("%Y-%m-%d")

        params = StrategyParams(
            ticker=ticker, start_date=start, end_date=end,
            sl_pct=sl, tp_pct=tp,
            macd_fast=m_fast, macd_slow=m_slow, macd_signal=m_sig,
            ma1_period=ma1, ma2_period=ma2, ma3_period=ma3
        )
        perf = run_simple_analysis(params)
        
        # 這裡將 ma1, ma2, ma3 也傳回前端，讓輸入框能記住剛輸入的值
        return templates.TemplateResponse("index_MACD.html", {
            "request": request, 
            **params.model_dump(), 
            "perf": perf,
            # 確保前端輸入框能顯示當前設定
            "ma1": ma1, "ma2": ma2, "ma3": ma3 
        })
    except Exception as e:
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)