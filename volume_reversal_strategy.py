import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from joblib import Parallel, delayed

# ==========================================
# 战法名称：极度缩量反包战法 (Volume Reversal Backtest)
# 核心逻辑：
# 1. 过滤：5-20元, 非ST, 非创业板.
# 2. 信号：
#    - 前期活跃：5日内曾有过放量(成交量 > MA20 * 1.5).
#    - 缩量回调：近2日成交量萎缩至 MA20 的 0.7 倍以下.
#    - 反包启动：今日收盘价 > 昨日最高价 且 今日阳线.
# 3. 回测：记录历史上所有符合点，统计 7, 14, 20, 60 天胜率及平均收益.
# ==========================================

STRATEGY_NAME = "backtest_reversal_strategy"
DATA_DIR = "stock_data"
NAMES_FILE = "stock_names.csv"

def backtest_logic(df):
    """
    建立虚拟持仓账本，扫描历史所有信号点
    """
    signals = []
    # 计算技术指标
    df['MA20_Vol'] = df['成交量'].rolling(window=20).mean()
    
    # 遍历历史 (预留60天观察期)
    for i in range(20, len(df) - 60):
        # 逻辑判断
        cond_active = (df['成交量'].iloc[i-5:i] > df['MA20_Vol'].iloc[i-5:i] * 1.5).any()
        cond_shrink = (df['成交量'].iloc[i-2:i] < df['MA20_Vol'].iloc[i] * 0.7).all()
        cond_reversal = (df['收盘'].iloc[i] > df['最高'].iloc[i-1]) and (df['收盘'].iloc[i] > df['开盘'].iloc[i])
        
        if cond_active and cond_shrink and cond_reversal:
            buy_price = df['收盘'].iloc[i]
            # 计算不同周期的涨跌幅
            res = {
                'date': df['日期'].iloc[i],
                'p7': (df['收盘'].iloc[i+7] - buy_price) / buy_price,
                'p14': (df['收盘'].iloc[i+14] - buy_price) / buy_price,
                'p20': (df['收盘'].iloc[i+20] - buy_price) / buy_price,
                'p60': (df['收盘'].iloc[i+60] - buy_price) / buy_price
            }
            signals.append(res)
    return signals

def analyze_stock(file_path, names_dict):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 100: return None
        
        code = os.path.basename(file_path).split('.')[0]
        name = names_dict.get(code, "未知")
        
        # 基础过滤
        if "ST" in name or code.startswith("30"): return None
        curr_price = df.iloc[-1]['收盘']
        if not (5.0 <= curr_price <= 20.0): return None

        # 1. 获取历史信号统计 (虚拟账本)
        history_signals = backtest_logic(df)
        if not history_signals:
            win_rate_20d = 0
            avg_ret_20d = 0
            total_hits = 0
        else:
            sig_df = pd.DataFrame(history_signals)
            total_hits = len(sig_df)
            win_rate_20d = (sig_df['p20'] > 0).sum() / total_hits
            avg_ret_20d = sig_df['p20'].mean()

        # 2. 判断今日是否有信号
        last_idx = len(df) - 1
        df['MA20_Vol'] = df['成交量'].rolling(window=20).mean()
        c1 = (df['成交量'].iloc[last_idx-5:last_idx] > df['MA20_Vol'].iloc[last_idx-5:last_idx] * 1.5).any()
        c2 = (df['成交量'].iloc[last_idx-2:last_idx] < df['MA20_Vol'].iloc[last_idx] * 0.7).all()
        c3 = (df['收盘'].iloc[last_idx] > df['最高'].iloc[last_idx-1]) and (df['收盘'].iloc[last_idx] > df['开盘'].iloc[last_idx])

        if c1 and c2 and c3:
            # 综合强度评估
            strength = "⭐⭐⭐⭐⭐" if win_rate_20d > 0.6 and avg_ret_20d > 0.05 else "⭐⭐⭐"
            advice = "全仓出击" if win_rate_20d > 0.7 else "观察试错"
            if total_hits < 3: advice = "历史数据不足，谨慎"

            return {
                "代码": code, "名称": name, "现价": curr_price,
                "20日历史胜率": f"{win_rate_20d*100:.1f}%",
                "历史平均20日收益": f"{avg_ret_20d*100:.1f}%",
                "历史发生次数": total_hits,
                "信号强度": strength, "操作建议": advice
            }
    except:
        return None

def main():
    names_df = pd.read_csv(NAMES_FILE)
    names_dict = dict(zip(names_df['code'].astype(str).str.zfill(6), names_df['name']))
    
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"正在进行全量并行回测，扫描 {len(files)} 只股票...")
    
    results = Parallel(n_jobs=-1)(delayed(analyze_stock)(f, names_dict) for f in files)
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        res_df = pd.DataFrame(valid_results).sort_values(by="20日历史胜率", ascending=False)
        
        now = datetime.now()
        out_dir = now.strftime("%Y-%m")
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f"{STRATEGY_NAME}_{now.strftime('%Y%m%d_%H%M')}.csv")
        
        res_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"扫描完毕，精选 {len(res_df)} 只个股。")
    else:
        print("今日无符合逻辑个股。")

if __name__ == "__main__":
    main()
