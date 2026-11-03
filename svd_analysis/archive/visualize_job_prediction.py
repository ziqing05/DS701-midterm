import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report

def main():
    print("=== Generating Visual Representation of SVD (k=5) Clusters (K=10) Job Prediction ===")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'bank-additional-full.csv')
    
    df = pd.read_csv(data_path, sep=';')
    train_df = df.iloc[:24712].copy()
    
    target = 'job'
    num_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    cat_cols = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    feature_cols = num_cols + cat_cols
    
    # 5-Fold Cross-Validation for Out-of-Fold Predictions
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = pd.Series(index=train_df.index, dtype='object')
    oof_cluster = pd.Series(index=train_df.index, dtype='int')
    
    # Store full-dataset SVD coordinates and KMeans for visualization
    pipe = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])
    
    X_proc = pipe.fit_transform(train_df[feature_cols])
    X_mean = np.mean(X_proc, axis=0)
    X_centered = X_proc - X_mean
    
    svd = TruncatedSVD(n_components=5, random_state=42)
    Z = svd.fit_transform(X_centered)
    
    km = KMeans(n_clusters=10, random_state=42, n_init=10).fit(Z)
    cluster_labels = km.labels_
    centroids = km.cluster_centers_
    
    # Run 5-fold CV to get genuine held-out predictions for confusion matrix & metrics
    for train_idx, val_idx in kf.split(train_df):
        X_tr = train_df.iloc[train_idx][feature_cols]
        y_tr = train_df.iloc[train_idx][target]
        X_va = train_df.iloc[val_idx][feature_cols]
        
        p = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ])
        X_tr_p = p.fit_transform(X_tr)
        X_va_p = p.transform(X_va)
        
        m_tr = np.mean(X_tr_p, axis=0)
        X_tr_c = X_tr_p - m_tr
        X_va_c = X_va_p - m_tr
        
        s = TruncatedSVD(n_components=5, random_state=42)
        Z_tr = s.fit_transform(X_tr_c)
        Z_va = s.transform(X_va_c)
        
        k_model = KMeans(n_clusters=10, random_state=42, n_init=10).fit(Z_tr)
        
        # Profile modes
        c_mode = {}
        for c in range(10):
            mask = (k_model.labels_ == c)
            c_mode[c] = y_tr[mask].mode()[0] if mask.sum() > 0 else y_tr.mode()[0]
            
        va_cl = k_model.predict(Z_va)
        oof_cluster.iloc[val_idx] = va_cl
        oof_preds.iloc[val_idx] = [c_mode[c] for c in va_cl]

    train_df['pred_job'] = oof_preds
    train_df['cluster'] = cluster_labels
    train_df['SVD_1'] = Z[:, 0]
    train_df['SVD_2'] = Z[:, 1]
    train_df['SVD_3'] = Z[:, 2]
    train_df['SVD_4'] = Z[:, 3]
    train_df['SVD_5'] = Z[:, 4]
    
    # Determine dominant job per cluster in full model
    cluster_dominant_jobs = {}
    for c in range(10):
        sub = train_df[train_df['cluster'] == c]
        cluster_dominant_jobs[c] = sub['job'].mode()[0]
        
    print("Cluster Dominant Jobs:", cluster_dominant_jobs)
    
    # -------------------------------------------------------------
    # BUILD 4-PANEL DASHBOARD FIGURE
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)
    
    # -------------------------------------------------------------
    # PANEL 1: Latent Space Projection (SVD 2 vs SVD 4) with Clusters & Centroids
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Palette for 10 clusters
    cmap10 = plt.get_cmap('tab10')
    
    # Sample 3,000 points for readable scatter
    sample_vis = train_df.sample(n=3500, random_state=42)
    
    for c in range(10):
        c_sub = sample_vis[sample_vis['cluster'] == c]
        pred_label = cluster_dominant_jobs[c]
        ax1.scatter(c_sub['SVD_2'], c_sub['SVD_4'], s=16, alpha=0.45, 
                    color=cmap10(c), label=f"C{c} $\\rightarrow$ Pred: {pred_label}")
        
    # Plot Centroids
    ax1.scatter(centroids[:, 1], centroids[:, 3], c='black', s=160, marker='X', edgecolors='white', linewidths=2, zorder=10)
    for c in range(10):
        ax1.annotate(f"C{c}: {cluster_dominant_jobs[c]}", (centroids[c, 1], centroids[c, 3]), 
                     xytext=(7, 7), textcoords='offset points', fontweight='bold', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="black", alpha=0.85), zorder=11)
        
    ax1.set_title("Panel A: 10 SVD Clusters & Decision Centroids in Latent Space\n(SVD 2: Channel/Inflation vs SVD 4: Demographic Life-Stage)", 
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel("SVD Component 2 (Telephone vs Cellular / Inflation)", fontsize=11)
    ax1.set_ylabel("SVD Component 4 (Age, Married, Basic Schooling vs Single, Degree)", fontsize=11)
    ax1.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # -------------------------------------------------------------
    # PANEL 2: Cluster Job Composition / Purity (Stacked Bar Chart)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    crosstab_job = pd.crosstab(train_df['cluster'], train_df['job'], normalize='index') * 100
    top_display_jobs = ['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'entrepreneur', 'self-employed', 'housemaid', 'student']
    crosstab_display = crosstab_job[top_display_jobs].copy()
    crosstab_display['other'] = 100 - crosstab_display.sum(axis=1)
    
    bottom = np.zeros(10)
    job_colors = plt.get_cmap('tab20').colors
    
    for idx, j_name in enumerate(crosstab_display.columns):
        vals = crosstab_display[j_name].values
        ax2.bar(range(10), vals, bottom=bottom, label=j_name, color=job_colors[idx % len(job_colors)], width=0.65)
        bottom += vals
        
    ax2.set_ylim(0, 116)
    ax2.set_title("Panel B: Job Composition & Purity across 10 SVD Clusters\n(Dominant Job assigned as Cluster Prediction)", 
                  fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Cluster Index (C0 – C9)", fontsize=11)
    ax2.set_ylabel("Share of Cluster Members (%)", fontsize=11)
    ax2.set_xticks(range(10))
    ax2.set_xticklabels([f"C{c}\nPred:\n{cluster_dominant_jobs[c]}" for c in range(10)], fontsize=8.5)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.28, 1.02), fontsize=8.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Annotate dominant purity on top with clean margin
    for c in range(10):
        top_j = cluster_dominant_jobs[c]
        purity = crosstab_job.loc[c, top_j]
        ax2.text(c, 103, f"{purity:.1f}%\n({top_j[:4]}.)", ha='center', fontsize=8, fontweight='bold', color='#2b6cb0')
        
    # -------------------------------------------------------------
    # PANEL 3: Normalized Out-of-Fold Confusion Matrix
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    
    all_jobs_sorted = ['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'entrepreneur', 'self-employed', 'housemaid', 'unemployed', 'student', 'unknown']
    
    # Filter to jobs present in predictions or actual
    cm = confusion_matrix(train_df['job'], train_df['pred_job'], labels=all_jobs_sorted)
    # Row-normalize to show recall per actual class
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im_cm = ax3.imshow(cm_norm, cmap='Blues', aspect='auto')
    ax3.set_title("Panel C: Out-of-Fold Confusion Matrix (True Job vs Predicted Job)\n(Row-normalized: diagonal indicates Class Recall)", 
                  fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(all_jobs_sorted)))
    ax3.set_yticks(range(len(all_jobs_sorted)))
    ax3.set_xticklabels(all_jobs_sorted, rotation=45, ha='right', fontsize=9)
    ax3.set_yticklabels(all_jobs_sorted, fontsize=9)
    ax3.set_xlabel("Predicted Job from SVD Cluster Mode", fontsize=11)
    ax3.set_ylabel("Actual Client Job", fontsize=11)
    plt.colorbar(im_cm, ax=ax3, fraction=0.046, pad=0.04)
    
    # Text in confusion matrix cells
    for r in range(len(all_jobs_sorted)):
        for c_idx in range(len(all_jobs_sorted)):
            val = cm_norm[r, c_idx]
            if val > 0.05:
                ax3.text(c_idx, r, f"{val:.2f}", ha='center', va='center', 
                         color='white' if val > 0.35 else 'black', fontsize=8)
                
    # -------------------------------------------------------------
    # PANEL 4: Class-by-Class Recall: SVD Clusters vs Majority Baseline
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Extract recall per class
    # Majority baseline predicts 'admin.' for all -> Recall is 1.0 for admin, 0.0 for all others!
    cluster_recalls = np.diag(cm_norm)
    baseline_recalls = [1.0 if j == 'admin.' else 0.0 for j in all_jobs_sorted]
    
    x_pos = np.arange(len(all_jobs_sorted))
    bar_w = 0.38
    
    rect1 = ax4.bar(x_pos - bar_w/2, [b*100 for b in baseline_recalls], bar_w, label='Majority Baseline (All "admin.")', color='#cbd5e0')
    rect2 = ax4.bar(x_pos + bar_w/2, [c*100 for c in cluster_recalls], bar_w, label='SVD (k=5) + Clusters (K=10)', color='#3182ce')
    
    ax4.set_title("Panel D: Recall by Job Category: SVD Clusters vs Baseline\n(Shows multi-class identification vs trivial majority collapse)", 
                  fontsize=13, fontweight='bold')
    ax4.set_ylabel("Class Recall (%)", fontsize=11)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(all_jobs_sorted, rotation=45, ha='right', fontsize=9)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Annotate recalls
    for i, rec in enumerate(cluster_recalls):
        if rec > 0.01:
            ax4.text(x_pos[i] + bar_w/2, rec*100 + 1.5, f"{rec*100:.1f}%", ha='center', fontsize=8, fontweight='bold', color='#2b6cb0')
            
    plt.savefig(os.path.join(output_dir, 'svd_k5_clusters10_job_prediction.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved svd_analysis/svd_k5_clusters10_job_prediction.png successfully!")

if __name__ == '__main__':
    main()
