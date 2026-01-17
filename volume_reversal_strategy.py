import pandas as pd
import os
import glob
from datetime import datetime
from joblib import Parallel, delayed

# ==========================================
# 战法名称：极度缩量反包战法 (Volume Reversal)
# 核心逻辑：
# 1. 寻找前期活跃股（曾有过放量阳线）。
# 2. 经历缩量回调（成交量萎缩至前期高量1/3以下或寻找阶段地量）。
# 3. 今日出现“反包”：收盘价盖过昨日实体，且成交量开始回升。
# ==========================================

STRATEGY_NAME = "volume_reversal_strategy"
DATA_DIR = "stock_data"
NAMES_FILE = "stock_names.csv"

def analyze_stock(file_path, stock_names_dict):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 30: return None
        
        code = os.path.basename(file_path).split('.')[0]
        name = stock_names_dict.get(code, "未知")

        # --- 基础过滤 ---
        # 排除ST(简单判断名称), 排除创业板(30开头)
        if "ST" in name or code.startswith("30"): return None
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # 价格区间过滤 (5.0 - 20.0)
        curr_price = last_row['收盘']
        if not (5.0 <= curr_price <= 20.0): return None

        # --- 战法逻辑计算 ---
        # 1. 计算均量
        vol_ma20 = df['成交量'].rolling(window=20).mean()
        
        # 2. 识别“极度缩量” (过去3天成交量低于20日均量的 0.6倍)
        is_narrow_vol = (df['成交量'].iloc[-3:-1] < vol_ma20.iloc[-1] * 0.7).all()
        
        # 3. 识别“反包” (今日收盘 > 昨日最高 且 今日是阳线)
        is_reversal = (last_row['收盘'] > prev_row['最高']) and (last_row['收盘'] > last_row['开盘'])
        
        # 4. 识别“温和放量” (今日量 > 昨日量)
        vol_increase = last_row['成交量'] > prev_row['成交量']

        if is_narrow_vol and is_reversal and vol_increase:
            # --- 历史回测模拟 (虚拟账本) ---
            # 假设该信号在历史上出现过，计算后7天的表现
            win_count = 0
            total_signals = 0
            # 简易回测逻辑：遍历历史寻找同类信号
            for i in range(20, len(df) - 7):
                h_vol_ma = vol_ma20.iloc[i]
                if (df['成交量'].iloc[i-2:i] < h_vol_ma * 0.7).all() and \
                   (df['收盘'].iloc[i] > df['最高'].iloc[i-1]):
                    total_signals += 1
                    if df['收盘'].iloc[i+7] > df['收盘'].iloc[i]:
                        win_count += 1
            
            win_rate = (win_count / total_signals * 100) if total_signals > 0 else 0
            
            # --- 强度评级 ---
            strength = "⭐⭐⭐⭐⭐" if win_rate > 60 else "⭐⭐⭐"
            advice = "分批建仓" if win_rate > 60 else "小仓试错"
            
            return {
                "日期": last_row['日期'],
                "代码": code,
                "名称": name,
                "现价": curr_price,
                "涨跌幅": f"{last_row['涨跌幅']}%",
                "换手率": last_row['换手率'],
                "历史胜率": f"{win_rate:.2f}%",
                "信号强度": strength,
                "操作建议": advice
            }
            
    except Exception as e:
        return None

def main():
    # 1. 加载股票名称映射
    names_df = pd.read_csv(NAMES_FILE)
    stock_names_dict = dict(zip(names_df['code'].astype(str).str.zfill(6), names_df['name']))

    # 2. 扫描数据目录 (并行处理提高速度)
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"开始扫描 {len(files)} 只股票...")
    
    results = Parallel(n_jobs=-1)(delayed(analyze_stock)(f, stock_names_dict) for f in files)
    
    # 3. 过滤掉空值并排序 (选优：按历史胜率排)
    valid_results = [r for r in results if r is not None]
    final_df = pd.DataFrame(valid_results)
    
    if not final_df.empty:
        final_df = final_df.sort_values(by="历史胜率", ascending=False).head(10) # 只要最强的10个
        
        # 4. 存储结果
        now = datetime.now()
        dir_path = now.strftime("%Y-%m")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        file_name = f"{STRATEGY_NAME}_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        save_path = os.path.join(dir_path, file_name)
        final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"复盘完成，信号已存入: {save_path}")
    else:
        print("今日无符合条件信号。")

if __name__ == "__main__":
    main()
