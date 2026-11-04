import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import KFold

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'bank-additional-full.csv')
    df = pd.read_csv(data_path, sep=';')
    
    # Strict project split: Training set is rows 0 to 24712
    train_df = df.iloc[:24712].copy()
    
    tier_map = {
        'admin.': 'White-Collar', 'technician': 'White-Collar', 'management': 'White-Collar',
        'blue-collar': 'Blue-Collar/Services', 'services': 'Blue-Collar/Services', 'housemaid': 'Blue-Collar/Services',
        'entrepreneur': 'Independent/Biz', 'self-employed': 'Independent/Biz',
        'retired': 'Life-Stage', 'student': 'Life-Stage', 'unemployed': 'Life-Stage',
        'unknown': 'Other'
    }
    train_df['job_tier'] = train_df['job'].map(tier_map)
    
    # Feature columns (strictly excluding job, duration, y)
    num_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    cat_cols = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    feature_cols = num_cols + cat_cols
    
    tiers = ['White-Collar', 'Blue-Collar/Services', 'Independent/Biz', 'Life-Stage']
    
    # Parameter sweep over cluster counts K
    k_values = [4, 6, 8, 10, 12, 15, 20]
    methods = ['Dendrogram (Ward)', 'KMeans']
    
    print("=== Starting Dendrogram (Ward) vs KMeans Sweep with Prevalence-Lift Assignment ===")
    print(f"Data rows: {len(train_df)} | SVD Components: 5 | K sweep: {k_values}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    best_predictions = {}
    
    for method in methods:
        print(f"\n--- Evaluating Method: {method} ---")
        for K in k_values:
            oof_tier_preds = []
            oof_tier_actuals = []
            oof_edu_preds = []
            oof_house_preds = []
            oof_indices = []
            
            for train_idx, val_idx in kf.split(train_df):
                tr = train_df.iloc[train_idx]
                va = train_df.iloc[val_idx]
                
                # Preprocess strictly on train fold
                pipe = ColumnTransformer([
                    ('num', StandardScaler(), num_cols),
                    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
                ])
                X_tr = pipe.fit_transform(tr[feature_cols])
                X_va = pipe.transform(va[feature_cols])
                
                m_tr = np.mean(X_tr, axis=0)
                X_tr_c = X_tr - m_tr
                X_va_c = X_va - m_tr
                
                # TruncatedSVD k=5
                svd = TruncatedSVD(n_components=5, random_state=42)
                Z_tr = svd.fit_transform(X_tr_c)
                Z_va = svd.transform(X_va_c)
                
                if method == 'Dendrogram (Ward)':
                    model = AgglomerativeClustering(n_clusters=K, linkage='ward')
                    tr_labels = model.fit_predict(Z_tr)
                    
                    # Compute centroids for out-of-fold nearest-cluster assignment
                    centroids = np.zeros((K, Z_tr.shape[1]))
                    for c in range(K):
                        mask = (tr_labels == c)
                        if mask.sum() > 0:
                            centroids[c] = np.mean(Z_tr[mask], axis=0)
                    
                    # Distance to centroids for validation fold
                    dists = np.linalg.norm(Z_va[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
                    va_labels = np.argmin(dists, axis=1)
                    
                else: # KMeans
                    model = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Z_tr)
                    tr_labels = model.labels_
                    va_labels = model.predict(Z_va)
                
                # Prevalence-Lift Assignment for Job Tier
                base_p = tr['job_tier'].value_counts(normalize=True)
                c_tier_pred = {}
                c_edu_pred = {}
                c_house_pred = {}
                
                for c in range(K):
                    mask = (tr_labels == c)
                    if mask.sum() == 0:
                        c_tier_pred[c] = 'White-Collar'
                        c_edu_pred[c] = 'university.degree'
                        c_house_pred[c] = 'yes'
                        continue
                    
                    # 1. Job Tier: Highest prevalence lift among main 4 tiers
                    c_p = tr.iloc[mask]['job_tier'].value_counts(normalize=True)
                    lifts = {t: c_p.get(t, 0.0) / base_p.get(t, 1e-6) for t in tiers}
                    c_tier_pred[c] = max(lifts, key=lifts.get)
                    
                    # 2. Downstream Education: Highest prevalence lift
                    base_edu = tr['education'].value_counts(normalize=True)
                    c_edu_p = tr.iloc[mask]['education'].value_counts(normalize=True)
                    edu_lifts = {e: c_edu_p.get(e, 0.0) / base_edu.get(e, 1e-6) for e in base_edu.index}
                    c_edu_pred[c] = max(edu_lifts, key=edu_lifts.get)
                    
                    # 3. Downstream Housing: Mode
                    c_house_pred[c] = tr.iloc[mask]['housing'].mode()[0]
                
                oof_tier_preds.extend([c_tier_pred[c] for c in va_labels])
                oof_tier_actuals.extend(va['job_tier'].tolist())
                oof_edu_preds.extend([c_edu_pred[c] for c in va_labels])
                oof_house_preds.extend([c_house_pred[c] for c in va_labels])
                oof_indices.extend(val_idx.tolist())
            
            oof_tier_preds = np.array(oof_tier_preds)
            oof_tier_actuals = np.array(oof_tier_actuals)
            
            # Metrics
            acc = accuracy_score(oof_tier_actuals, oof_tier_preds)
            macro_f1 = f1_score(oof_tier_actuals, oof_tier_preds, average='macro', zero_division=0)
            
            # Per-tier recall
            tier_recalls = {}
            for t in tiers:
                t_mask = (oof_tier_actuals == t)
                tier_recalls[t] = np.mean(oof_tier_preds[t_mask] == t) if t_mask.sum() > 0 else 0.0
            
            num_tiers_pred = len(set(oof_tier_preds))
            
            # Downstream metrics
            edu_acc = accuracy_score(train_df.iloc[oof_indices]['education'], oof_edu_preds)
            edu_f1 = f1_score(train_df.iloc[oof_indices]['education'], oof_edu_preds, average='macro', zero_division=0)
            house_acc = accuracy_score(train_df.iloc[oof_indices]['housing'], oof_house_preds)
            house_f1 = f1_score(train_df.iloc[oof_indices]['housing'], oof_house_preds, average='macro', zero_division=0)
            
            res_row = {
                'Method': method,
                'K': K,
                'Accuracy': acc,
                'Macro_F1': macro_f1,
                'Tiers_Predicted': num_tiers_pred,
                'Recall_White_Collar': tier_recalls['White-Collar'],
                'Recall_Blue_Collar': tier_recalls['Blue-Collar/Services'],
                'Recall_Ind_Biz': tier_recalls['Independent/Biz'],
                'Recall_Life_Stage': tier_recalls['Life-Stage'],
                'Edu_Accuracy': edu_acc,
                'Edu_Macro_F1': edu_f1,
                'House_Accuracy': house_acc,
                'House_Macro_F1': house_f1
            }
            results.append(res_row)
            print(f"K={K:2d} | Acc={acc*100:.2f}% | Macro F1={macro_f1:.4f} | Tiers Pred={num_tiers_pred}/4 | Recall: WC={tier_recalls['White-Collar']*100:.1f}%, BC={tier_recalls['Blue-Collar/Services']*100:.1f}%, Ind={tier_recalls['Independent/Biz']*100:.1f}%, Life={tier_recalls['Life-Stage']*100:.1f}%")
            
            if K == 10:
                best_predictions[method] = {
                    'preds': oof_tier_preds,
                    'actuals': oof_tier_actuals,
                    'indices': oof_indices
                }
    
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'dendrogram_vs_kmeans_sweep.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved sweep results to {csv_path}")
    
    # Visualization: 4-Panel Comparative Dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel A: Macro F1 vs K (Dendrogram vs KMeans)
    dendro_df = results_df[results_df['Method'] == 'Dendrogram (Ward)']
    kmeans_df = results_df[results_df['Method'] == 'KMeans']
    
    ax1.plot(dendro_df['K'], dendro_df['Macro_F1'], marker='o', linewidth=2.5, color='#2b6cb0', label='Dendrogram (Ward Linkage)')
    ax1.plot(kmeans_df['K'], kmeans_df['Macro_F1'], marker='s', linewidth=2.5, color='#dd6b20', label='KMeans Clustering')
    ax1.axhline(0.1312, color='gray', linestyle='--', label='Majority Best Guess (0.1312)')
    ax1.set_xlabel('Number of Clusters (K)', fontsize=11)
    ax1.set_ylabel('Out-of-Fold Macro F1', fontsize=11)
    ax1.set_title('Panel A: Macro F1 Across Cluster Counts (K)\n(Comparing Dendrogram vs. KMeans on SVD Space)', fontsize=12, fontweight='bold')
    ax1.set_xticks(k_values)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel B: Per-Tier Recall for Dendrogram (Ward) Across K
    ax2.plot(dendro_df['K'], dendro_df['Recall_White_Collar'] * 100, marker='o', label='White-Collar', color='#3182ce')
    ax2.plot(dendro_df['K'], dendro_df['Recall_Blue_Collar'] * 100, marker='^', label='Blue-Collar/Services', color='#38a169')
    ax2.plot(dendro_df['K'], dendro_df['Recall_Life_Stage'] * 100, marker='D', label='Life-Stage (Students/Retirees)', color='#e53e3e')
    ax2.plot(dendro_df['K'], dendro_df['Recall_Ind_Biz'] * 100, marker='s', label='Independent/Biz', color='#805ad5')
    ax2.set_xlabel('Number of Clusters (K)', fontsize=11)
    ax2.set_ylabel('Out-of-Fold Recall (%)', fontsize=11)
    ax2.set_title('Panel B: Dendrogram Per-Tier Recall Across K\n(All 4 Tiers Actively Predicted via Prevalence-Lift)', fontsize=12, fontweight='bold')
    ax2.set_xticks(k_values)
    ax2.legend(loc='upper right', fontsize=9.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Panel C: Downstream Conversion Rate by Predicted Tier (Dendrogram vs KMeans at K=10)
    overall_conv = (train_df['y'] == 'yes').mean()
    
    # Evaluate conversion rate by predicted tier for Dendrogram at K=10
    eval_df_dendro = pd.DataFrame({
        'pred_tier': best_predictions['Dendrogram (Ward)']['preds'],
        'y': train_df.iloc[best_predictions['Dendrogram (Ward)']['indices']]['y'].values
    })
    conv_dendro = eval_df_dendro.groupby('pred_tier')['y'].agg(
        rate=lambda s: (s == 'yes').mean(),
        count='count'
    ).reindex(tiers).fillna(0)
    
    eval_df_km = pd.DataFrame({
        'pred_tier': best_predictions['KMeans']['preds'],
        'y': train_df.iloc[best_predictions['KMeans']['indices']]['y'].values
    })
    conv_km = eval_df_km.groupby('pred_tier')['y'].agg(
        rate=lambda s: (s == 'yes').mean(),
        count='count'
    ).reindex(tiers).fillna(0)
    
    x = np.arange(len(tiers))
    width = 0.35
    
    rects1 = ax3.bar(x - width/2, conv_dendro['rate'] * 100, width, label='Dendrogram (K=10)', color='#2b6cb0')
    rects2 = ax3.bar(x + width/2, conv_km['rate'] * 100, width, label='KMeans (K=10)', color='#dd6b20')
    ax3.axhline(overall_conv * 100, color='crimson', linestyle='--', label=f'Overall Base Rate ({overall_conv*100:.2f}%)')
    ax3.set_ylabel('Term Deposit Conversion Rate (%)', fontsize=11)
    ax3.set_title('Panel C: Downstream Lead Quality by Inferred Tier (K=10)\n(Life-Stage Emerges with Massive Conversion Lift)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(tiers, rotation=15, ha='right', fontsize=9.5)
    ax3.legend(loc='upper right', fontsize=9.5)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    
    for r1, r2 in zip(rects1, rects2):
        h1 = r1.get_height()
        h2 = r2.get_height()
        ax3.text(r1.get_x() + r1.get_width()/2., h1 + 0.15, f"{h1/overall_conv/100:.1f}x", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1a365d')
        ax3.text(r2.get_x() + r2.get_width()/2., h2 + 0.15, f"{h2/overall_conv/100:.1f}x", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#7b341e')
        
    # Panel D: Downstream Secondary Feature Macro F1 (Education & Housing)
    ax4.plot(dendro_df['K'], dendro_df['Edu_Macro_F1'], marker='o', label='Education - Dendrogram', color='#2b6cb0')
    ax4.plot(kmeans_df['K'], kmeans_df['Edu_Macro_F1'], marker='s', label='Education - KMeans', color='#dd6b20')
    ax4.plot(dendro_df['K'], dendro_df['House_Macro_F1'], marker='^', linestyle=':', label='Housing - Dendrogram', color='#319795')
    ax4.plot(kmeans_df['K'], kmeans_df['House_Macro_F1'], marker='v', linestyle=':', label='Housing - KMeans', color='#b7791f')
    ax4.axhline(0.0541, color='gray', linestyle='--', label='Education Best Guess (0.0541)')
    ax4.set_xlabel('Number of Clusters (K)', fontsize=11)
    ax4.set_ylabel('Out-of-Fold Macro F1', fontsize=11)
    ax4.set_title('Panel D: Downstream Prediction on Education & Housing\n(Cluster Representation Generalization Across K)', fontsize=12, fontweight='bold')
    ax4.set_xticks(k_values)
    ax4.legend(loc='center right', fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'dendrogram_vs_kmeans_sweep.png')
    fig.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved comparative dashboard to {plot_path}")
    print("=== Sweep and Comparison Complete ===")

if __name__ == '__main__':
    main()
