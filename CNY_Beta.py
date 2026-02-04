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
import scipy.optimize as sco

# ==========================================
# 6. Black-Litterman 核心引擎 (新增模块)
# ==========================================
class BlackLittermanStrategy:
    def __init__(self, price_data, sector_ranks, risk_aversion=2.5, tau=0.05):
        """
        初始化 BL 模型
        :param price_data: 包含各板块历史收盘价的 DataFrame
        :param sector_ranks: 上一步算出的板块评分表 (包含 'Avg_Score')
        :param risk_aversion: 风险厌恶系数 (Delta)，通常取 2.5-3.0
        :param tau: 观点不确定性系数，通常取 0.025-0.05
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.sector_ranks = sector_ranks.set_index('板块')
        
        # 1. 数据清洗：剔除汇率列，只保留板块价格
        self.prices = price_data.drop(columns=['USD_CNY', 'date'], errors='ignore')
        if 'date' in self.prices.index.names:
            pass # index is already date
        
        # 2. 计算历史收益率与协方差矩阵 (Sigma)
        self.returns = self.prices.pct_change().dropna()
        self.assets = self.returns.columns.tolist()
        self.n_assets = len(self.assets)
        
        # 年化协方差矩阵 (假设252个交易日)
        self.sigma = self.returns.cov() * 252

    def get_market_equilibrium(self):
        """
        计算市场隐含均衡收益 (Pi)
        由于很难实时获取板块的总市值，这里我们假设'等权重'为市场中性基准(Prior)，
        或者你可以理解为我们对市场市值的先验是无信息的。
        """
        # 假设市场权重 (等权) -> 也可以换成真实的流通市值权重
        w_mkt = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # Pi = Delta * Sigma * w_mkt
        # 这是如果不考虑人民币升值，市场“理应”给出的回报
        pi = self.risk_aversion * self.sigma.dot(w_mkt)
        return pi, w_mkt

    def mapping_views(self):
        """
        【关键步骤】将 'Avg_Score' (观点分) 映射为 'Q' (预期收益向量)
        逻辑：
        1. 之前的 Avg_Score 范围大约在 -1 到 1 之间。
        2. 我们不能直接说 1分 = 100% 收益。
        3. 我们用板块的'年化波动率'作为锚点。
           如果某板块得分 1.0 (极度看好)，我们预期它跑赢均衡收益 0.5 个标准差。
        """
        # 计算各板块年化波动率
        volatilities = np.sqrt(np.diag(self.sigma))
        
        P = np.eye(self.n_assets) # 观点矩阵 (绝对观点，对角阵)
        Q = np.zeros(self.n_assets) # 观点收益向量
        
        # 信心矩阵 Omega
        # 简化的 He-Litterman 方法: Omega = diag(tau * P * Sigma * P.T)
        omega = np.diag(np.diag(self.tau * self.sigma))
        
        print("\n>>> [BL模型] 正在将宏观因子映射为收益观点...")
        
        pi, _ = self.get_market_equilibrium()
        
        for i, asset in enumerate(self.assets):
            # 获取该板块的得分
            if asset in self.sector_ranks.index:
                score = self.sector_ranks.loc[asset, 'Avg_Score']
            else:
                score = 0
            
            # --- 核心映射逻辑 ---
            # 观点收益 Q = 隐含均衡收益 Pi + 主动观点
            # 主动观点 = 得分 * 波动率 * 激进系数 (0.5)
            # 含义：如果是满分，我预期它比市场隐含收益多涨 0.5 倍波动率
            active_view = score * volatilities[i] * 0.5
            Q[i] = pi[i] + active_view
            
            # 动态调整信心 (Omega)
            # 如果得分绝对值很高(>0.5)，说明信号强烈，我们缩小方差(增加信心)
            if abs(score) > 0.5:
                omega[i, i] *= 0.5 
                
            print(f"    -> {asset:<6} | 得分:{score:>5.2f} | 隐含收益:{pi[i]:.2%} -> BL观点收益:{Q[i]:.2%}")
            
        return P, Q, omega

    def optimize(self):
        """
        计算 BL 后验收益并优化权重
        """
        pi, w_mkt = self.get_market_equilibrium()
        P, Q, omega = self.mapping_views()
        
        # --- BL 核心公式 ---
        # 1. 计算中间项
        tau_sigma_inv = np.linalg.inv(self.tau * self.sigma)
        omega_inv = np.linalg.inv(omega)
        
        # 2. 计算后验协方差 (Posterior Sigma) 的逆
        # M = (tau*Sigma)^-1 + P.T * Omega^-1 * P
        M = tau_sigma_inv + P.T.dot(omega_inv).dot(P)
        M_inv = np.linalg.inv(M)
        
        # 3. 计算后验预期收益 (Posterior E[R])
        # E[R] = M^-1 * [ (tau*Sigma)^-1 * Pi + P.T * Omega^-1 * Q ]
        term1 = tau_sigma_inv.dot(pi)
        term2 = P.T.dot(omega_inv).dot(Q)
        bl_returns = M_inv.dot(term1 + term2)
        
        # 4. 计算后验协方差 (Posterior Covariance)
        # Sigma_BL = Sigma + M^-1
        bl_sigma = self.sigma + M_inv
        
        # --- 均值-方差优化 (Mean-Variance Optimization) ---
        # 目标：最大化夏普比率
        print("\n>>> [BL模型] 正在进行凸优化求解最优权重...")
        
        def neg_sharpe(weights):
            r = weights.dot(bl_returns)
            vol = np.sqrt(weights.T.dot(bl_sigma).dot(weights))
            return -r / vol # 负夏普，用于求最小
        
        # 约束条件
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # 权重和为1
        bounds = tuple((0.0, 0.4) for _ in range(self.n_assets)) # 风控：单板块最大仓位 40%
        
        init_guess = w_mkt
        opts = sco.minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not opts.success:
            print("!!! 优化失败，使用等权配置")
            return pd.Series(w_mkt, index=self.assets)
        
        return pd.Series(opts.x, index=self.assets)

# ==========================================
# 7. 主程序续写 (Integration)
# ==========================================
def run_bl_process(df_data, df_result):
    # 实例化策略
    bl_strategy = BlackLittermanStrategy(df_data, df_result)
    
    # 获取优化权重
    optimal_weights = bl_strategy.optimize()
    
    # 整理结果
    df_allocation = pd.DataFrame({
        '板块': optimal_weights.index,
        '建议权重': optimal_weights.values
    }).sort_values('建议权重', ascending=False)
    
    # 过滤掉权重极小的值显示
    df_allocation = df_allocation[df_allocation['建议权重'] > 0.001]
    
    print("\n" + "="*40)
    print("🏆 Black-Litterman 最终仓位建议")
    print("="*40)
    print(df_allocation)
    
    # 画饼图
    plt.figure(figsize=(10, 6))
    plt.pie(df_allocation['建议权重'], labels=df_allocation['板块'], autopct='%1.1f%%', startangle=140)
    plt.title('基于人民币升值因子的 BL 模型资产配置', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# ==========================================
# 更新 main 函数
# ==========================================
if __name__ == "__main__":
    # 1. 获取真实数据
    df_data = get_real_data(lookback_days=365)
    
    if df_data is not None:
        # 2. 计算原始因子
        df_factors = calculate_raw_factors(df_data)
        
        # 3. 运行权重敏感度分析 (得到 Avg_Score)
        df_result = analyze_weight_sensitivity(df_factors)
        
        print("\n>>> 最终综合排名 (按稳定性排序):")
        print(df_result[['板块', 'Avg_Score', 'Score_交易型', 'Score_投资型']])
        
        # 4. 画图 (Beta 排名)
        plot_final_result(df_result)
        
        # ----------------------------------------
        # >>> 续写部分：执行 BL 资产配置 <<<
        # ----------------------------------------
        run_bl_process(df_data, df_result)
