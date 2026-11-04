import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import KFold

df = pd.read_csv('../bank-additional-full.csv', sep=';')
train_df = df.iloc[:24712].copy()

num_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
cat_cols = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
feature_cols = num_cols + cat_cols

base_rate = (train_df['y'] == 'yes').mean()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for K in [8, 10, 15, 20]:
    oof_cluster_ids = np.zeros(len(train_df), dtype=int)
    oof_cluster_rates = np.zeros(len(train_df))
    oof_dists = np.zeros(len(train_df))
    
    for train_idx, val_idx in kf.split(train_df):
        tr = train_df.iloc[train_idx]
        va = train_df.iloc[val_idx]
        
        pipe = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ])
        X_tr = pipe.fit_transform(tr[feature_cols])
        X_va = pipe.transform(va[feature_cols])
        
        m_tr = np.mean(X_tr, axis=0)
        X_tr_c = X_tr - m_tr
        X_va_c = X_va - m_tr
        
        svd = TruncatedSVD(n_components=5, random_state=42)
        Z_tr = svd.fit_transform(X_tr_c)
        Z_va = svd.transform(X_va_c)
        
        model = AgglomerativeClustering(n_clusters=K, linkage='ward')
        tr_labels = model.fit_predict(Z_tr)
        
        centroids = np.zeros((K, Z_tr.shape[1]))
        c_conv_rate = np.zeros(K)
        for c in range(K):
            mask = (tr_labels == c)
            if mask.sum() > 0:
                centroids[c] = np.mean(Z_tr[mask], axis=0)
                c_conv_rate[c] = (tr.iloc[mask]['y'] == 'yes').mean()
                
        dists = np.linalg.norm(Z_va[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        va_labels = np.argmin(dists, axis=1)
        min_dists = np.min(dists, axis=1)
        
        oof_cluster_ids[val_idx] = va_labels
        oof_cluster_rates[val_idx] = c_conv_rate[va_labels]
        oof_dists[val_idx] = min_dists
        
    eval_df = pd.DataFrame({
        'cluster_id': oof_cluster_ids,
        'cluster_rate': oof_cluster_rates,
        'dist': oof_dists,
        'y': (train_df['y'] == 'yes').astype(int)
    })
    
    # Ranking candidates: First by highest cluster conversion rate, then closest to centroid (most representative)
    eval_df_sorted = eval_df.sort_values(by=['cluster_rate', 'dist'], ascending=[False, True])
    
    top_150 = eval_df_sorted.iloc[:150]
    conversions_top_150 = top_150['y'].sum()
    rate_top_150 = top_150['y'].mean()
    
    expected_random_conv = 150 * base_rate
    
    cost_per_call = 5
    revenue_per_conv = 1000
    
    total_cost = 150 * cost_per_call
    
    rev_dendro = conversions_top_150 * revenue_per_conv
    profit_dendro = rev_dendro - total_cost
    
    rev_random = expected_random_conv * revenue_per_conv
    profit_random = rev_random - total_cost
    
    net_benefit = profit_dendro - profit_random
    lift = rate_top_150 / base_rate
    
    top_cluster_size = (eval_df['cluster_rate'] == eval_df['cluster_rate'].max()).sum()
    max_c_rate = eval_df['cluster_rate'].max()
    
    print(f"--- K = {K} ---")
    print(f"Top Cluster Base Rate: {max_c_rate*100:.2f}% (Total members across OOF: {top_cluster_size})")
    print(f"Top 150 Candidates:")
    print(f"  Actual Conversions: {conversions_top_150} / 150 ({rate_top_150*100:.2f}%)")
    print(f"  Expected Random Conversions: {expected_random_conv:.2f} / 150 ({base_rate*100:.2f}%)")
    print(f"  Lift: {lift:.2f}x")
    print(f"Financials (150 calls @ $5 = $750 cost, $1,000/conversion):")
    print(f"  Dendrogram Revenue: ${rev_dendro:,.2f} | Net Profit: ${profit_dendro:,.2f}")
    print(f"  Random Revenue:     ${rev_random:,.2f} | Net Profit: ${profit_random:,.2f}")
    print(f"  Net Dollar Benefit: +${net_benefit:,.2f} (ROI: {profit_dendro/total_cost*100:.1f}%)")
    print()
