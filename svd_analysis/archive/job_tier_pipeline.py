import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def map_job_tier(j):
    if j in ['admin.', 'technician', 'management']:
        return 'White-Collar'
    elif j in ['blue-collar', 'services', 'housemaid']:
        return 'Blue-Collar/Services'
    elif j in ['retired', 'student', 'unemployed']:
        return 'Life-Stage'
    elif j in ['entrepreneur', 'self-employed']:
        return 'Independent/Biz'
    return 'Other'

def main():
    print("=== Pipeline: 4 Job Tiers via SVD + Dendrogram, and Downstream Prediction ===")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'bank-additional-full.csv')
    
    df = pd.read_csv(data_path, sep=';')
    train_df = df.iloc[:24712].copy()
    
    # Create the 4 Tiers
    train_df['job_tier'] = train_df['job'].apply(map_job_tier)
    print("\nJob Tier Distribution in Training Set:")
    print(train_df['job_tier'].value_counts(normalize=True).round(4) * 100)
    
    tier_names = ['White-Collar', 'Blue-Collar/Services', 'Independent/Biz', 'Life-Stage']
    
    # Feature sets: exclude job, job_tier, duration, y
    num_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    cat_cols = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    feature_cols = num_cols + cat_cols
    
    # -------------------------------------------------------------
    # 1. SVD + DENDROGRAM GENERATION
    # -------------------------------------------------------------
    print("\n--- Step 1: Computing SVD (k=5) and SVD-Dendrogram ---")
    sample_size = 1500
    df_sample = train_df.sample(n=sample_size, random_state=42).copy()
    
    pipe = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])
    
    X_sample_p = pipe.fit_transform(df_sample[feature_cols])
    X_sample_c = X_sample_p - np.mean(X_sample_p, axis=0)
    
    svd_vis = TruncatedSVD(n_components=5, random_state=42)
    Z_sample = svd_vis.fit_transform(X_sample_c)
    
    Z_link = linkage(Z_sample, method='ward')
    
    # Cut tree into 8 clusters and determine each cluster's dominant job tier
    cl_sample = cut_tree(Z_link, n_clusters=8).flatten()
    df_sample['cl'] = cl_sample
    
    cl_tier_map = {}
    for c in range(8):
        sub = df_sample[df_sample['cl'] == c]
        top_tier = sub['job_tier'].mode()[0]
        purity = (sub['job_tier'] == top_tier).mean()
        cl_tier_map[c] = f"C{c}: {top_tier} ({purity*100:.0f}%)"
        
    print("Cluster Dominant Tiers (Sample Dendrogram):", cl_tier_map)
    
    # Plot SVD Dendrogram
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        Z_link,
        truncate_mode='lastp',
        p=25,
        leaf_rotation=90.,
        leaf_font_size=9.5,
        ax=ax
    )
    ax.set_title('SVD-Preconditioned Dendrogram for Occupational Tier Segmentation\n(Ward Linkage in Top 5 SVD Dimensions, N=1,500)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Cluster Leaf / Merged Nodes (Leaves segment White-Collar vs Blue-Collar vs Life-Stage)', fontsize=11)
    ax.set_ylabel('Ward Distance in SVD Latent Space', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    dendro_path = os.path.join(output_dir, 'tier_svd_dendrogram.png')
    fig.savefig(dendro_path, dpi=200)
    plt.close()
    print(f"Saved SVD Dendrogram to {dendro_path}")
    
    # -------------------------------------------------------------
    # 2. 5-FOLD CROSS-VALIDATION: PREDICTING JOB TIER & DOWNSTREAM FEATURES
    # -------------------------------------------------------------
    print("\n--- Step 2: 5-Fold Cross-Validation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_tier_preds = pd.Series(index=train_df.index, dtype='object')
    oof_edu_preds = pd.Series(index=train_df.index, dtype='object')
    oof_housing_preds = pd.Series(index=train_df.index, dtype='object')
    oof_cluster_ids = pd.Series(index=train_df.index, dtype='int')
    
    tier_base_class = train_df['job_tier'].mode()[0]
    edu_base_class = train_df['education'].mode()[0]
    housing_base_class = train_df['housing'].mode()[0]
    
    fold_tier_accs, fold_tier_f1s = [], []
    fold_edu_accs, fold_edu_f1s = [], []
    fold_house_accs, fold_house_f1s = [], []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        X_tr = train_df.iloc[train_idx][feature_cols]
        y_tr_tier = train_df.iloc[train_idx]['job_tier']
        y_tr_edu = train_df.iloc[train_idx]['education']
        y_tr_house = train_df.iloc[train_idx]['housing']
        
        X_va = train_df.iloc[val_idx][feature_cols]
        y_va_tier = train_df.iloc[val_idx]['job_tier']
        y_va_edu = train_df.iloc[val_idx]['education']
        y_va_house = train_df.iloc[val_idx]['housing']
        
        # Preprocess strictly on train fold
        p = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ])
        X_tr_p = p.fit_transform(X_tr)
        X_va_p = p.transform(X_va)
        
        m_tr = np.mean(X_tr_p, axis=0)
        X_tr_c = X_tr_p - m_tr
        X_va_c = X_va_p - m_tr
        
        # SVD k=5
        svd = TruncatedSVD(n_components=5, random_state=42)
        Z_tr = svd.fit_transform(X_tr_c)
        Z_va = svd.transform(X_va_c)
        
        # KMeans (K=10) on SVD space
        km = KMeans(n_clusters=10, random_state=42, n_init=10).fit(Z_tr)
        
        # Profile cluster modes for: Tier, Education, Housing
        cluster_tier_mode = {}
        cluster_edu_mode = {}
        cluster_house_mode = {}
        
        for c in range(10):
            mask = (km.labels_ == c)
            if mask.sum() > 0:
                cluster_tier_mode[c] = y_tr_tier[mask].mode()[0]
                cluster_edu_mode[c] = y_tr_edu[mask].mode()[0]
                cluster_house_mode[c] = y_tr_house[mask].mode()[0]
            else:
                cluster_tier_mode[c] = tier_base_class
                cluster_edu_mode[c] = edu_base_class
                cluster_house_mode[c] = housing_base_class
                
        # Predict on held-out validation fold
        va_cl = km.predict(Z_va)
        pred_tier = [cluster_tier_mode[c] for c in va_cl]
        pred_edu = [cluster_edu_mode[c] for c in va_cl]
        pred_house = [cluster_house_mode[c] for c in va_cl]
        
        oof_tier_preds.iloc[val_idx] = pred_tier
        oof_edu_preds.iloc[val_idx] = pred_edu
        oof_housing_preds.iloc[val_idx] = pred_house
        oof_cluster_ids.iloc[val_idx] = va_cl
        
        fold_tier_accs.append(accuracy_score(y_va_tier, pred_tier))
        fold_tier_f1s.append(f1_score(y_va_tier, pred_tier, average='macro', zero_division=0))
        
        fold_edu_accs.append(accuracy_score(y_va_edu, pred_edu))
        fold_edu_f1s.append(f1_score(y_va_edu, pred_edu, average='macro', zero_division=0))
        
        fold_house_accs.append(accuracy_score(y_va_house, pred_house))
        fold_house_f1s.append(f1_score(y_va_house, pred_house, average='macro', zero_division=0))

    train_df['pred_tier'] = oof_tier_preds
    train_df['pred_edu'] = oof_edu_preds
    train_df['pred_housing'] = oof_housing_preds
    train_df['oof_cluster'] = oof_cluster_ids
    
    # Baselines
    base_tier_acc = (train_df['job_tier'] == tier_base_class).mean()
    base_tier_f1 = f1_score(train_df['job_tier'], [tier_base_class]*len(train_df), average='macro', zero_division=0)
    
    base_edu_acc = (train_df['education'] == edu_base_class).mean()
    base_edu_f1 = f1_score(train_df['education'], [edu_base_class]*len(train_df), average='macro', zero_division=0)
    
    base_house_acc = (train_df['housing'] == housing_base_class).mean()
    base_house_f1 = f1_score(train_df['housing'], [housing_base_class]*len(train_df), average='macro', zero_division=0)
    
    print("\n=== Cross-Validation Results Summary ===")
    print(f"1. Job Tier Prediction: Acc={np.mean(fold_tier_accs)*100:.2f}% (Base: {base_tier_acc*100:.2f}%) | Macro F1={np.mean(fold_tier_f1s):.4f} (Base: {base_tier_f1:.4f})")
    print(f"2. Education Prediction: Acc={np.mean(fold_edu_accs)*100:.2f}% (Base: {base_edu_acc*100:.2f}%) | Macro F1={np.mean(fold_edu_f1s):.4f} (Base: {base_edu_f1:.4f})")
    print(f"3. Housing Loan Predict: Acc={np.mean(fold_house_accs)*100:.2f}% (Base: {base_house_acc*100:.2f}%) | Macro F1={np.mean(fold_house_f1s):.4f} (Base: {base_house_f1:.4f})")
    
    # -------------------------------------------------------------
    # 3. DOWNSTREAM TERMP DEPOSIT CONVERSION BY PREDICTED TIER
    # -------------------------------------------------------------
    tier_conv = train_df.groupby('pred_tier')['y'].agg(
        total='count',
        yes=lambda s: (s == 'yes').sum(),
        rate=lambda s: (s == 'yes').mean()
    ).reset_index()
    overall_base_rate = (train_df['y'] == 'yes').mean()
    tier_conv['lift'] = tier_conv['rate'] / overall_base_rate
    print("\nDownstream Deposit Conversion by Predicted Job Tier:")
    print(tier_conv.to_string(index=False))

    # -------------------------------------------------------------
    # 4. GENERATE MULTI-PANEL DASHBOARD FIGURE
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)
    
    # PANEL A: Latent SVD 2 vs SVD 4 colored by Predicted Job Tier
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Get SVD coordinates on full train for plotting
    pipe_full = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])
    X_full_p = pipe_full.fit_transform(train_df[feature_cols])
    X_full_c = X_full_p - np.mean(X_full_p, axis=0)
    svd_full = TruncatedSVD(n_components=5, random_state=42)
    Z_full = svd_full.fit_transform(X_full_c)
    train_df['SVD_2'] = Z_full[:, 1]
    train_df['SVD_4'] = Z_full[:, 3]
    
    sample_sub = train_df.sample(n=3500, random_state=42)
    tier_colors = {
        'White-Collar': '#3182ce',
        'Blue-Collar/Services': '#dd6b20',
        'Life-Stage': '#38a169',
        'Independent/Biz': '#805ad5',
        'Other': '#a0aec0'
    }
    
    for t_name, col in tier_colors.items():
        sub_pts = sample_sub[sample_sub['pred_tier'] == t_name]
        if len(sub_pts) > 0:
            ax1.scatter(sub_pts['SVD_2'], sub_pts['SVD_4'], s=14, alpha=0.5, color=col, label=f"Pred: {t_name}")
            
    ax1.set_title("Panel A: SVD Decision Space for 4 Occupational Tiers\n(SVD 2: Channel/Inflation vs SVD 4: Demographic Life-Stage)", 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel("SVD Component 2 (Telephone vs Cellular / Inflation)", fontsize=11)
    ax1.set_ylabel("SVD Component 4 (Age, Married vs Single, Degree)", fontsize=11)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # PANEL B: Confusion Matrix of Predicted Job Tier
    ax2 = fig.add_subplot(gs[0, 1])
    eval_tiers = ['White-Collar', 'Blue-Collar/Services', 'Independent/Biz', 'Life-Stage']
    cm = confusion_matrix(train_df['job_tier'], train_df['pred_tier'], labels=eval_tiers)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im_cm = ax2.imshow(cm_norm, cmap='Blues', aspect='auto')
    ax2.set_title("Panel B: Out-of-Fold Confusion Matrix for Job Tiers\n(Row-normalized: diagonal shows Tier Recall)", 
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(eval_tiers)))
    ax2.set_yticks(range(len(eval_tiers)))
    ax2.set_xticklabels(eval_tiers, rotation=25, ha='right', fontsize=9.5)
    ax2.set_yticklabels(eval_tiers, fontsize=9.5)
    ax2.set_xlabel("Predicted Job Tier", fontsize=11)
    ax2.set_ylabel("Actual Job Tier", fontsize=11)
    plt.colorbar(im_cm, ax=ax2, fraction=0.046, pad=0.04)
    
    for r in range(len(eval_tiers)):
        for c in range(len(eval_tiers)):
            val = cm_norm[r, c]
            ax2.text(c, r, f"{val:.2f}", ha='center', va='center', 
                     color='white' if val > 0.4 else 'black', fontsize=9.5, fontweight='bold')
                     
    # PANEL C: Downstream Feature Prediction Accuracy & F1 Lift
    ax3 = fig.add_subplot(gs[1, 0])
    
    tasks = ['Job Tier', 'Education', 'Housing Loan']
    base_acc_vals = [base_tier_acc * 100, base_edu_acc * 100, base_house_acc * 100]
    clust_acc_vals = [np.mean(fold_tier_accs) * 100, np.mean(fold_edu_accs) * 100, np.mean(fold_house_accs) * 100]
    
    x = np.arange(len(tasks))
    w = 0.35
    ax3.bar(x - w/2, base_acc_vals, w, label='Majority Baseline', color='#cbd5e0')
    ax3.bar(x + w/2, clust_acc_vals, w, label='SVD (k=5) + Clusters (K=10)', color='#3182ce')
    ax3.set_ylabel("Held-Out 5-Fold Accuracy (%)", fontsize=11)
    ax3.set_title("Panel C: Downstream Prediction on Multiple Features\n(Predicting Tier, Education, and Housing via Cluster Personas)", 
                  fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(tasks, fontsize=11)
    ax3.legend(loc='lower right', fontsize=9.5)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    
    for i in range(len(tasks)):
        gain = clust_acc_vals[i] - base_acc_vals[i]
        ax3.text(x[i] + w/2, clust_acc_vals[i] + 0.8, f"+{gain:.2f}%", ha='center', fontweight='bold', color='#2b6cb0', fontsize=10)
        
    # PANEL D: Term Deposit Conversion Rate by Inferred Tier
    ax4 = fig.add_subplot(gs[1, 1])
    
    tier_plot_df = tier_conv[tier_conv['pred_tier'].isin(eval_tiers)].copy()
    tier_plot_df = tier_plot_df.sort_values(by='rate', ascending=False)
    
    bars = ax4.bar(tier_plot_df['pred_tier'], tier_plot_df['rate'] * 100, color='#38a169', width=0.5)
    ax4.axhline(overall_base_rate * 100, color='crimson', linestyle='--', label=f'Overall Base Rate ({overall_base_rate*100:.2f}%)')
    ax4.set_ylabel("Subscription Rate (%)", fontsize=11)
    ax4.set_title("Panel D: Downstream Conversion Rate by Predicted Job Tier\n(Assessing Lead Quality of Inferred Personas)", 
                  fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(tier_plot_df)))
    ax4.set_xticklabels(tier_plot_df['pred_tier'], rotation=20, ha='right', fontsize=9.5)
    ax4.legend(loc='upper right', fontsize=9.5)
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar, lift in zip(bars, tier_plot_df['lift']):
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{lift:.2f}x Lift", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#22543d')
        
    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, 'job_tier_pipeline_results.png')
    fig.savefig(dashboard_path, dpi=200)
    plt.close()
    print(f"Saved dashboard figure to {dashboard_path}")
    print("\n=== Execution Complete ===")

if __name__ == '__main__':
    main()
