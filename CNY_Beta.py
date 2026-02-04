import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime, timedelta

# 解决中文显示问题

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置区域
# ==========================================
SECTOR_MAPPING = {
    "航空机场": "航空机场",
    "造纸印刷": "造纸印刷",
    "银行": "银行",
    "房地产": "房地产开发",
    "食品饮料": "食品饮料",
    "半导体": "半导体",
    "家用电器": "家电行业",
    "纺织服装": "纺织服装",
    "航运港口": "航运港口"
}

# 基础分 (基本面逻辑预设)
FUNDAMENTAL_PRIORS = {
    "航空机场": 1.0, "造纸印刷": 0.8, "银行": 0.5, "房地产": 0.4, "食品饮料": 0.2,
    "半导体": -0.1, "家用电器": -0.6, "纺织服装": -0.8, "航运港口": -0.5
}

# ==========================================
# 2. 数据获取模块
# ==========================================
def get_real_data(lookback_days=365):
    print(">>> [1/3] 正在获取数据...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    # --- A. 获取汇率 (新浪接口) ---
    try:
        print("    -> 正在获取美元/人民币汇率 (新浪)...")
        df_fx = ak.currency_boc_sina(symbol="美元", start_date=start_str, end_date=end_str)
        df_fx['date'] = pd.to_datetime(df_fx['日期'])
        
        # 优先用中行折算价
        if '中行折算价' in df_fx.columns:
            df_fx['USD_CNY'] = pd.to_numeric(df_fx['中行折算价']) / 100
        else:
            df_fx['USD_CNY'] = pd.to_numeric(df_fx['现汇卖出价']) / 100
            
        df_fx = df_fx[['date', 'USD_CNY']].sort_values('date').set_index('date')
        df_fx = df_fx.resample('D').ffill() # 补全周末
        
    except Exception as e:
        print(f"!!! 汇率获取失败: {e}")
        return None

    # --- B. 获取行业数据 (东财接口) ---
    sector_prices = pd.DataFrame()
    print("    -> 正在获取行业板块数据 (这可能需要几十秒)...")
    
    for logic_name, em_name in SECTOR_MAPPING.items():
        try:
            df_board = ak.stock_board_industry_hist_em(
                symbol=em_name, start_date=start_str, end_date=end_str, adjust="qfq"
            )
            df_board['date'] = pd.to_datetime(df_board['日期'])
            df_board.set_index('date', inplace=True)
            sector_prices[logic_name] = df_board['收盘']
            time.sleep(0.3)
        except Exception as e:
            print(f"    [跳过] {em_name}: {e}")

    # --- C. 合并 ---
    if sector_prices.empty: return None
    df_final = pd.merge(sector_prices, df_fx, left_index=True, right_index=True, how='inner')
    return df_final

# ==========================================
# 3. 因子计算引擎 (只算因子，不合成)
# ==========================================
def calculate_raw_factors(df_data):
    """
    计算每个板块的统计Beta和基础分，返回原始因子表
    """
    if df_data is None: return None
    
    df_ret = df_data.pct_change().dropna()
    fx_ret = df_ret['USD_CNY']
    
    results = []
    
    print(">>> [2/3] 正在进行滞后回归分析...")
    for sector in SECTOR_MAPPING.keys():
        if sector not in df_ret.columns: continue
            
        sector_ret = df_ret[sector]
        
        # 寻找最佳滞后 Beta
        best_beta = 0
        best_r2 = -999
        
        for lag in range(11): # 0-10天滞后
            y = sector_ret.iloc[lag:].values
            X = fx_ret.shift(lag).iloc[lag:].values.reshape(-1, 1)
            
            if len(y) < 30: continue
            
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)
            
            if r2 > best_r2:
                best_r2 = r2
                best_beta = model.coef_[0]
        
        # 统计因子 (Stat_Factor):
        # Beta < 0 代表 汇率跌(升值) -> 股价涨。
        # 为了让因子方向一致 (越大越利好)，取 -Beta
        stat_raw = -best_beta
        
        # 基础因子 (Fund_Factor)
        fund_raw = FUNDAMENTAL_PRIORS.get(sector, 0)
        
        results.append({
            "板块": sector,
            "Stat_Raw": stat_raw,    # 原始统计分 (尚未归一化)
            "Fund_Raw": fund_raw     # 原始基础分 (-1 ~ 1)
        })
        
    return pd.DataFrame(results)

# ==========================================
# 4. 权重敏感度分析 (核心功能)
# ==========================================
def analyze_weight_sensitivity(df_factors):
    """
    对因子进行归一化，并测试不同权重下的排名
    """
    print("\n>>> [3/3] 正在进行权重敏感度分析...")
    
    df = df_factors.copy()
    
    # 1. 归一化 (Normalization)
    # 将统计分和基础分都缩放到 [-1, 1] 区间，保证权重计算公平
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[['Stat_Norm', 'Fund_Norm']] = scaler.fit_transform(df[['Stat_Raw', 'Fund_Raw']])
    
    # 2. 定义三种场景
    scenarios = [
        {"name": "交易型 (重盘面)", "w1": 0.8, "w2": 0.2},
        {"name": "均衡型 (推荐)",   "w1": 0.6, "w2": 0.4},
        {"name": "投资型 (重逻辑)", "w1": 0.3, "w2": 0.7}
    ]
    
    final_output = df[['板块']].copy()
    
    print("-" * 60)
    print(f"{'场景':<15} | {'Top 1':<10} | {'Top 2':<10} | {'Top 3':<10}")
    print("-" * 60)
    
    for s in scenarios:
        col_name = f"Score_{s['name'][:3]}" # Score_交易型
        # 计算综合分
        final_output[col_name] = (df['Stat_Norm'] * s['w1']) + (df['Fund_Norm'] * s['w2'])
        
        # 排序并打印 Top 3
        sorted_df = final_output.sort_values(col_name, ascending=False)
        top_sectors = sorted_df['板块'].head(3).tolist()
        
        print(f"{s['name']:<15} | {top_sectors[0]:<10} | {top_sectors[1]:<10} | {top_sectors[2]:<10}")

    # 3. 计算稳定性 (平均排名)
    # 简单的逻辑：算出三种模式下的平均分，得分越高的越稳
    score_cols = [c for c in final_output.columns if 'Score' in c]
    final_output['Avg_Score'] = final_output[score_cols].mean(axis=1)
    
    # 最终总排名
    final_rank = final_output.sort_values('Avg_Score', ascending=False).reset_index(drop=True)
    
    return final_rank

# ==========================================
# 5. 可视化
# ==========================================
def plot_final_result(df_rank):
    plt.figure(figsize=(12, 6))
    
    # 绘制 "Avg_Score" (综合稳定性得分)
    df_plot = df_rank.sort_values('Avg_Score', ascending=True) # 升序以便画横向柱状图
    
    colors = ['#d62728' if x > 0 else '#2ca02c' for x in df_plot['Avg_Score']]
    bars = plt.barh(df_plot['板块'], df_plot['Avg_Score'], color=colors)
    
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title('人民币升值利好系数 (基于多权重敏感度加权)', fontsize=14)
    plt.xlabel('综合强度得分 (归一化后)')
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w * 1.05 if w>0 else w*1.05-0.1, bar.get_y() + bar.get_height()/2, f'{w:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 获取真实数据
    df_data = get_real_data(lookback_days=365)
    
    if df_data is not None:
        # 2. 计算原始因子
        df_factors = calculate_raw_factors(df_data)
        
        # 3. 运行权重敏感度分析
        # 这里会打印出三种风格下的不同排名，帮你决定权重
        df_result = analyze_weight_sensitivity(df_factors)
        
        print("\n>>> 最终综合排名 (按稳定性排序):")
        print(df_result[['板块', 'Avg_Score', 'Score_交易型', 'Score_投资型']])
        
        # 4. 画图
        plot_final_result(df_result)