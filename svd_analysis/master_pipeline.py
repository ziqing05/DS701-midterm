import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'bank-additional-full.csv')
    df = pd.read_csv(data_path, sep=';')
    
    train_df = df.iloc[:24712].copy()
    
    tier_map = {
        'admin.': 'White-Collar', 'technician': 'White-Collar', 'management': 'White-Collar',
        'blue-collar': 'Blue-Collar/Services', 'services': 'Blue-Collar/Services', 'housemaid': 'Blue-Collar/Services',
        'entrepreneur': 'Independent/Biz', 'self-employed': 'Independent/Biz',
        'retired': 'Life-Stage', 'student': 'Life-Stage', 'unemployed': 'Life-Stage',
        'unknown': 'Other'
    }
    train_df['job_tier'] = train_df['job'].map(tier_map)
    tiers = ['White-Collar', 'Blue-Collar/Services', 'Independent/Biz', 'Life-Stage']
    
    num_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    cat_cols = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    feature_cols = num_cols + cat_cols
    
    base_rate = (train_df['y'] == 'yes').mean()
    print("=" * 80)
    print("DS701 MIDTERM: MASTER SVD & CLUSTERING PIPELINE")
    print(f"Training Rows: {len(train_df):,} | Base Conversion Rate: {base_rate*100:.2f}%")
    print("=" * 80)


    print("\n[STAGE 1] Running SVD Dimensionality Sweep (k in [2, 3, 5, 8, 10, 15, 20, 30])...")
    
    full_pipe = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])
    X_full = full_pipe.fit_transform(train_df[feature_cols])
    X_mean = np.mean(X_full, axis=0)
    X_centered = X_full - X_mean
    
    total_variance = np.sum(np.var(X_centered, axis=0))
    frobenius_full = np.linalg.norm(X_centered, 'fro')
    
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_centered), size=1200, replace=False)
    X_sub = X_centered[sample_idx]
    dist_raw = pdist(X_sub)
    Z_raw_ward = linkage(X_sub, method='ward')
    c_raw, _ = cophenet(Z_raw_ward, dist_raw)
    
    svd_k_list = [2, 3, 5, 8, 10, 15, 20, 30]
    svd_records = []
    
    for k in svd_k_list:
        svd = TruncatedSVD(n_components=k, random_state=42)
        Z_k = svd.fit_transform(X_centered)
        X_rec = svd.inverse_transform(Z_k)
        
        var_expl = np.sum(svd.explained_variance_)
        var_ratio = var_expl / total_variance
        recon_err = np.linalg.norm(X_centered - X_rec, 'fro') / frobenius_full
        
        Z_sub = Z_k[sample_idx]
        c_k, _ = cophenet(linkage(Z_sub, method='ward'), pdist(Z_sub))
        
        svd_records.append({
            'k': k,
            'variance_explained_ratio': var_ratio,
            'frobenius_relative_error': recon_err,
            'cophenetic_corr': c_k
        })
        print(f"  k={k:2d} | Variance Expl={var_ratio*100:.1f}% | Rel Error={recon_err*100:.1f}% | Cophenetic Corr={c_k:.3f}")
        
    svd_df = pd.DataFrame(svd_records)
    svd_df.to_csv(os.path.join(script_dir, 'master_svd_sweep.csv'), index=False)


    print("\n[STAGE 2] Running 5-Fold CV Clustering Sweep (k=5 SVD, K in [4, 6, 8, 10, 12, 15, 20])...")
    
    k_clusters_list = [4, 6, 8, 10, 12, 15, 20]
    methods = ['Dendrogram (Ward)', 'KMeans']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    clustering_results = []
    oof_predictions_store = {}
    
    for method in methods:
        print(f"\n  Evaluating {method}:")
        for K in k_clusters_list:
            oof_tier_preds = []
            oof_tier_actuals = []
            oof_edu_preds = []
            oof_house_preds = []
            oof_cluster_ids = []
            oof_cluster_conv_rates = []
            oof_indices = []
            
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
                
                # SVD k=5
                svd = TruncatedSVD(n_components=5, random_state=42)
                Z_tr = svd.fit_transform(X_tr_c)
                Z_va = svd.transform(X_va_c)
                
                if method == 'Dendrogram (Ward)':
                    model = AgglomerativeClustering(n_clusters=K, linkage='ward')
                    tr_labels = model.fit_predict(Z_tr)
                    
                    centroids = np.zeros((K, Z_tr.shape[1]))
                    for c in range(K):
                        mask = (tr_labels == c)
                        if mask.sum() > 0:
                            centroids[c] = np.mean(Z_tr[mask], axis=0)
                            
                    dists = np.linalg.norm(Z_va[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
                    va_labels = np.argmin(dists, axis=1)
                else:
                    model = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Z_tr)
                    tr_labels = model.labels_
                    va_labels = model.predict(Z_va)
                
                # Prevalence-Lift Assignment for Job Tier & Secondary Features
                base_p = tr['job_tier'].value_counts(normalize=True)
                base_edu = tr['education'].value_counts(normalize=True)
                
                c_tier_pred = {}
                c_edu_pred = {}
                c_house_pred = {}
                c_conv_rate = {}
                
                for c in range(K):
                    mask = (tr_labels == c)
                    if mask.sum() == 0:
                        c_tier_pred[c] = 'White-Collar'
                        c_edu_pred[c] = 'university.degree'
                        c_house_pred[c] = 'yes'
                        c_conv_rate[c] = base_rate
                        continue
                    
                    c_p = tr.iloc[mask]['job_tier'].value_counts(normalize=True)
                    lifts = {t: c_p.get(t, 0.0) / base_p.get(t, 1e-6) for t in tiers}
                    c_tier_pred[c] = max(lifts, key=lifts.get)
                    
                    c_edu_p = tr.iloc[mask]['education'].value_counts(normalize=True)
                    edu_lifts = {e: c_edu_p.get(e, 0.0) / base_edu.get(e, 1e-6) for e in base_edu.index}
                    c_edu_pred[c] = max(edu_lifts, key=edu_lifts.get)
                    
                    c_house_pred[c] = tr.iloc[mask]['housing'].mode()[0]
                    
                    c_conv_rate[c] = (tr.iloc[mask]['y'] == 'yes').mean()
                    
                oof_tier_preds.extend([c_tier_pred[c] for c in va_labels])
                oof_tier_actuals.extend(va['job_tier'].tolist())
                oof_edu_preds.extend([c_edu_pred[c] for c in va_labels])
                oof_house_preds.extend([c_house_pred[c] for c in va_labels])
                oof_cluster_ids.extend(va_labels.tolist())
                oof_cluster_conv_rates.extend([c_conv_rate[c] for c in va_labels])
                oof_indices.extend(val_idx.tolist())
                
            oof_tier_preds = np.array(oof_tier_preds)
            oof_tier_actuals = np.array(oof_tier_actuals)
            
            acc = accuracy_score(oof_tier_actuals, oof_tier_preds)
            macro_f1 = f1_score(oof_tier_actuals, oof_tier_preds, average='macro', zero_division=0)
            
            tier_recalls = {}
            for t in tiers:
                t_mask = (oof_tier_actuals == t)
                tier_recalls[t] = np.mean(oof_tier_preds[t_mask] == t) if t_mask.sum() > 0 else 0.0
                
            edu_f1 = f1_score(train_df.iloc[oof_indices]['education'], oof_edu_preds, average='macro', zero_division=0)
            house_f1 = f1_score(train_df.iloc[oof_indices]['housing'], oof_house_preds, average='macro', zero_division=0)
            
            clustering_results.append({
                'Method': method,
                'K': K,
                'Accuracy': acc,
                'Macro_F1': macro_f1,
                'Tiers_Predicted': len(set(oof_tier_preds)),
                'Recall_White_Collar': tier_recalls['White-Collar'],
                'Recall_Blue_Collar': tier_recalls['Blue-Collar/Services'],
                'Recall_Ind_Biz': tier_recalls['Independent/Biz'],
                'Recall_Life_Stage': tier_recalls['Life-Stage'],
                'Edu_Macro_F1': edu_f1,
                'House_Macro_F1': house_f1
            })
            
            oof_predictions_store[(method, K)] = pd.DataFrame({
                'idx': oof_indices,
                'pred_tier': oof_tier_preds,
                'actual_tier': oof_tier_actuals,
                'cluster_rate': oof_cluster_conv_rates,
                'y': (train_df.iloc[oof_indices]['y'] == 'yes').astype(int).values
            })
            
            print(f"    K={K:2d} | Macro F1={macro_f1:.4f} | Life-Stage Recall={tier_recalls['Life-Stage']*100:.1f}% | Ind/Biz Recall={tier_recalls['Independent/Biz']*100:.1f}%")
            
    clustering_df = pd.DataFrame(clustering_results)
    clustering_df.to_csv(os.path.join(script_dir, 'master_clustering_sweep.csv'), index=False)

#CBA
    print("\n[STAGE 3] Performing Cost-Benefit Analysis (CBA)...")
    print("  Parameters: Call Cost = $5.00 | Conversion Value = $100.00")
    
    call_cost = 5.0
    conversion_value = 100.0
    budget_target = 150
    
    cba_records = []
    
    rand_exp_conv = budget_target * base_rate
    rand_cost = budget_target * call_cost
    rand_rev = rand_exp_conv * conversion_value
    rand_profit = rand_rev - rand_cost
    rand_roi = (rand_profit / rand_cost) * 100
    
    cba_records.append({
        'Model': 'Random Selection',
        'K': '-',
        'Calls': budget_target,
        'Conversions': rand_exp_conv,
        'Conversion_Rate': base_rate,
        'Lift': 1.0,
        'Total_Cost': rand_cost,
        'Gross_Revenue': rand_rev,
        'Net_Profit': rand_profit,
        'ROI_Pct': rand_roi,
        'Net_Benefit_vs_Random': 0.0
    })
    
    for method in ['Dendrogram (Ward)', 'KMeans']:
        for K in [8, 10, 15, 20]:
            df_pred = oof_predictions_store[(method, K)].sort_values(by='cluster_rate', ascending=False)
            top_cohort = df_pred.iloc[:budget_target]
            
            act_conv = top_cohort['y'].sum()
            act_rate = top_cohort['y'].mean()
            lift = act_rate / base_rate
            
            cost = budget_target * call_cost
            rev = act_conv * conversion_value
            profit = rev - cost
            roi = (profit / cost) * 100
            net_benefit = profit - rand_profit
            
            cba_records.append({
                'Model': method,
                'K': K,
                'Calls': budget_target,
                'Conversions': act_conv,
                'Conversion_Rate': act_rate,
                'Lift': lift,
                'Total_Cost': cost,
                'Gross_Revenue': rev,
                'Net_Profit': profit,
                'ROI_Pct': roi,
                'Net_Benefit_vs_Random': net_benefit
            })
            
    cba_df = pd.DataFrame(cba_records)
    cba_df.to_csv(os.path.join(script_dir, 'master_cba_summary.csv'), index=False)
    
    print("\nCBA Summary Table (Top 150 Calls @ $5/call, $100/conversion):")
    print(cba_df[['Model', 'K', 'Conversions', 'Conversion_Rate', 'Net_Profit', 'ROI_Pct', 'Net_Benefit_vs_Random']].to_string(index=False))


    print("\n[STAGE 4] Generating Master 4-Panel Visualization Dashboard...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 13))
    

    ax1_twin = ax1.twinx()
    l1 = ax1.plot(svd_df['k'], svd_df['variance_explained_ratio'] * 100, marker='o', color='#2b6cb0', linewidth=2.5, label='Cumulative Variance Explained (%)')
    l2 = ax1_twin.plot(svd_df['k'], svd_df['cophenetic_corr'], marker='^', color='#38a169', linewidth=2.5, linestyle='--', label='Cophenetic Tree Correlation')
    ax1_twin.axhline(c_raw, color='gray', linestyle=':', label=f'Raw 58-D Cophenetic Baseline ({c_raw:.3f})')
    
    ax1.axvline(5, color='#e53e3e', linestyle='-.', alpha=0.8, label='Selected Sweet Spot (k=5)')
    ax1.set_xlabel('SVD Latent Rank (k)', fontsize=11)
    ax1.set_ylabel('Variance Explained (%)', fontsize=11, color='#2b6cb0')
    ax1_twin.set_ylabel('Cophenetic Tree Correlation', fontsize=11, color='#38a169')
    ax1.set_title('Panel A: SVD Regularization & Tree Fidelity', fontsize=12, fontweight='bold')
    
    lines = l1 + l2 + [ax1_twin.get_lines()[-1], ax1.get_lines()[-1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=9.5)
    ax1.grid(True, linestyle='--', alpha=0.5)


    dendro_res = clustering_df[clustering_df['Method'] == 'Dendrogram (Ward)']
    kmeans_res = clustering_df[clustering_df['Method'] == 'KMeans']
    
    ax2.plot(dendro_res['K'], dendro_res['Macro_F1'], marker='o', color='#2b6cb0', linewidth=2.5, label='Dendrogram Macro F1')
    ax2.plot(kmeans_res['K'], kmeans_res['Macro_F1'], marker='s', color='#dd6b20', linewidth=2.5, label='KMeans Macro F1')
    ax2.plot(dendro_res['K'], dendro_res['Recall_Life_Stage'], marker='D', color='#e53e3e', linewidth=2, linestyle='--', label='Life-Stage Recall (Dendrogram)')
    ax2.plot(dendro_res['K'], dendro_res['Recall_Ind_Biz'], marker='v', color='#805ad5', linewidth=2, linestyle='--', label='Ind/Biz Recall (Dendrogram)')
    ax2.axhline(0.1312, color='gray', linestyle=':', label='Majority Class Baseline F1 (0.1312)')
    
    ax2.set_xlabel('Number of Clusters (K)', fontsize=11)
    ax2.set_ylabel('Score / Recall', fontsize=11)
    ax2.set_title('Panel B: Multi-Tier Clustering Generalization\n', fontsize=12, fontweight='bold')
    ax2.set_xticks(k_clusters_list)
    ax2.legend(loc='lower right', fontsize=9.5)
    ax2.grid(True, linestyle='--', alpha=0.5)


    cba_subset = cba_df[cba_df['K'].isin(['-', 10, 15, 20])].copy()
    labels_cba = [f"{row['Model']}\n(K={row['K']})" if row['K'] != '-' else "Random Dialing\n(Population Base)" for _, row in cba_subset.iterrows()]
    profits = cba_subset['Net_Profit'].values
    colors = ['#e53e3e' if p < 0 else '#3182ce' if 'Dendrogram' in m else '#dd6b20' for p, m in zip(profits, cba_subset['Model'])]
    
    bars = ax3.bar(range(len(profits)), profits, color=colors, width=0.55)
    ax3.axhline(0, color='black', linewidth=1)
    ax3.set_ylabel('Net Profit ($) for 150 Calls', fontsize=11)
    ax3.set_title('Panel C: Cost-Benefit Analysis (Top 150 Leads)\nCost=$5/call, Revenue=$100/conversion', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(profits)))
    ax3.set_xticklabels(labels_cba, rotation=15, ha='right', fontsize=9)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar, p, conv, roi in zip(bars, profits, cba_subset['Conversions'], cba_subset['ROI_Pct']):
        y_pos = bar.get_height()
        if p >= 0:
            ax3.text(bar.get_x() + bar.get_width()/2.0, y_pos + 80, f"+${p:,.0f}\n({conv:.0f} conv, {roi:.0f}% ROI)", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        else:
            ax3.text(bar.get_x() + bar.get_width()/2.0, y_pos - 350, f"-${abs(p):,.0f}\n({conv:.1f} conv, {roi:.1f}% ROI)", ha='center', va='top', fontsize=8.5, fontweight='bold', color='crimson')

  
    budgets = np.arange(25, 601, 25)
    dendro10_df = oof_predictions_store[('Dendrogram (Ward)', 10)].sort_values(by='cluster_rate', ascending=False)
    dendro15_df = oof_predictions_store[('Dendrogram (Ward)', 15)].sort_values(by='cluster_rate', ascending=False)
    
    profit_rand_curve = (budgets * base_rate * conversion_value) - (budgets * call_cost)
    profit_d10_curve = [dendro10_df.iloc[:b]['y'].sum() * conversion_value - (b * call_cost) for b in budgets]
    profit_d15_curve = [dendro15_df.iloc[:b]['y'].sum() * conversion_value - (b * call_cost) for b in budgets]
    
    ax4.plot(budgets, profit_d15_curve, marker='D', color='#38a169', linewidth=2.5, label='Dendrogram (K=15) Prioritization')
    ax4.plot(budgets, profit_d10_curve, marker='o', color='#2b6cb0', linewidth=2.5, label='Dendrogram (K=10) Prioritization')
    ax4.plot(budgets, profit_rand_curve, color='#e53e3e', linestyle='--', linewidth=2, label='Random Selection (Negative ROI Baseline)')
    ax4.axhline(0, color='black', linewidth=0.8, linestyle=':')
    
    ax4.set_xlabel('Call Volume Budget (Number of Calls Made)', fontsize=11)
    ax4.set_ylabel('Net Profit ($)', fontsize=11)
    ax4.set_title('Panel D: Cumulative Net Profit vs. Contact Budget Curve\n', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    dashboard_path = os.path.join(script_dir, 'master_pipeline_dashboard.png')
    fig.savefig(dashboard_path, dpi=200)
    plt.close()
    print(f"\n[DONE] Saved consolidated visual dashboard to: {dashboard_path}")
    print("=" * 80)

if __name__ == '__main__':
    main()
