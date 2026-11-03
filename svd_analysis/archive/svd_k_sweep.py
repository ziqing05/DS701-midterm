import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, adjusted_mutual_info_score, silhouette_score

def main():
    print("=== Fine-Grained Sweep of SVD Components (k) vs Cluster Prediction Quality ===")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'bank-additional-full.csv')
    
    df = pd.read_csv(data_path, sep=';')
    train_df = df.iloc[:24712].copy()
    
    num_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    # Grid of SVD ranks to test
    # None represents Full Raw Preprocessed Features
    svd_k_values = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, None]
    
    # Grid of Cluster counts K
    cluster_K_values = [5, 10, 15, 20]
    
    target_features = ['job', 'education']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = []
    
    for target in target_features:
        print(f"\n=======================================================")
        print(f"Evaluating Target Feature: '{target}'")
        print(f"=======================================================")
        
        curr_num = [c for c in num_cols if c != target]
        curr_cat = [c for c in cat_cols if c != target]
        curr_feats = curr_num + curr_cat
        
        y_all = train_df[target]
        base_class = y_all.mode()[0]
        base_acc = (y_all == base_class).mean()
        base_f1 = f1_score(y_all, [base_class]*len(y_all), average='macro', zero_division=0)
        print(f"Majority Baseline: Predict '{base_class}' | Acc: {base_acc*100:.2f}% | Macro F1: {base_f1:.4f}\n")
        
        for k_svd in svd_k_values:
            k_name = f"k={k_svd}" if k_svd is not None else "Full (Raw)"
            print(f"Testing SVD {k_name:12s} ...", end="", flush=True)
            
            for n_clusters in cluster_K_values:
                fold_accs = []
                fold_f1s = []
                fold_amis = []
                fold_silhs = []
                
                for train_idx, val_idx in kf.split(train_df):
                    X_tr = train_df.iloc[train_idx][curr_feats]
                    y_tr = train_df.iloc[train_idx][target]
                    X_va = train_df.iloc[val_idx][curr_feats]
                    y_va = train_df.iloc[val_idx][target]
                    
                    pipe = ColumnTransformer([
                        ('num', StandardScaler(), curr_num),
                        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), curr_cat)
                    ])
                    X_tr_p = pipe.fit_transform(X_tr)
                    X_va_p = pipe.transform(X_va)
                    
                    # Center
                    mean_tr = np.mean(X_tr_p, axis=0)
                    X_tr_c = X_tr_p - mean_tr
                    X_va_c = X_va_p - mean_tr
                    
                    if k_svd is not None:
                        svd = TruncatedSVD(n_components=k_svd, random_state=42)
                        X_tr_in = svd.fit_transform(X_tr_c)
                        X_va_in = svd.transform(X_va_c)
                    else:
                        X_tr_in = X_tr_c
                        X_va_in = X_va_c
                        
                    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5).fit(X_tr_in)
                    train_cl = km.labels_
                    
                    # Compute cluster majority classes
                    c_modes = {}
                    for c in range(n_clusters):
                        m = (train_cl == c)
                        c_modes[c] = y_tr[m].mode()[0] if m.sum() > 0 else y_tr.mode()[0]
                        
                    val_cl = km.predict(X_va_in)
                    val_preds = [c_modes[c] for c in val_cl]
                    
                    acc = accuracy_score(y_va, val_preds)
                    f1 = f1_score(y_va, val_preds, average='macro', zero_division=0)
                    ami = adjusted_mutual_info_score(y_va, val_cl)
                    
                    fold_accs.append(acc)
                    fold_f1s.append(f1)
                    fold_amis.append(ami)
                    
                # Compute silhouette on sample for tractability
                sil_sample_idx = np.random.default_rng(42).choice(len(X_tr_in), size=2000, replace=False)
                sil = silhouette_score(X_tr_in[sil_sample_idx], train_cl[sil_sample_idx])
                
                m_acc = np.mean(fold_accs)
                s_acc = np.std(fold_accs)
                m_f1 = np.mean(fold_f1s)
                s_f1 = np.std(fold_f1s)
                m_ami = np.mean(fold_amis)
                
                all_results.append({
                    'Target': target,
                    'SVD_k': k_svd if k_svd is not None else X_tr_p.shape[1],
                    'SVD_Label': k_name,
                    'Clusters_K': n_clusters,
                    'Mean_Accuracy': m_acc,
                    'Std_Accuracy': s_acc,
                    'Accuracy_Gain': m_acc - base_acc,
                    'Mean_Macro_F1': m_f1,
                    'Std_Macro_F1': s_f1,
                    'F1_Lift': m_f1 - base_f1,
                    'AMI': m_ami,
                    'Silhouette': sil
                })
            print(" Done.")

    res_df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'svd_k_sweep_results.csv')
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved sweep results to {csv_path}")
    
    # -------------------------------------------------------------
    # Plotting: Accuracy & Macro F1 vs SVD k (for K=10 and K=15 clusters)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    svd_numeric_k = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
    
    for i, target in enumerate(['job', 'education']):
        sub = res_df[res_df['Target'] == target]
        base_acc = (train_df[target] == train_df[target].mode()[0]).mean() * 100
        base_f1 = f1_score(train_df[target], [train_df[target].mode()[0]]*len(train_df), average='macro', zero_division=0)
        
        # Plot Accuracy vs SVD k
        ax_acc = axes[i, 0]
        raw_row_k10 = sub[(sub['SVD_Label'] == 'Full (Raw)') & (sub['Clusters_K'] == 10)].iloc[0]
        raw_row_k15 = sub[(sub['SVD_Label'] == 'Full (Raw)') & (sub['Clusters_K'] == 15)].iloc[0]
        
        k10_data = sub[(sub['SVD_Label'] != 'Full (Raw)') & (sub['Clusters_K'] == 10)].sort_values('SVD_k')
        k15_data = sub[(sub['SVD_Label'] != 'Full (Raw)') & (sub['Clusters_K'] == 15)].sort_values('SVD_k')
        
        ax_acc.plot(k10_data['SVD_k'], k10_data['Mean_Accuracy']*100, 'o-', color='#3182ce', lw=2, label='Cluster K=10')
        ax_acc.fill_between(k10_data['SVD_k'], (k10_data['Mean_Accuracy'] - k10_data['Std_Accuracy'])*100, 
                            (k10_data['Mean_Accuracy'] + k10_data['Std_Accuracy'])*100, color='#3182ce', alpha=0.15)
        
        ax_acc.plot(k15_data['SVD_k'], k15_data['Mean_Accuracy']*100, 's--', color='#38a169', lw=2, label='Cluster K=15')
        
        ax_acc.axhline(base_acc, color='crimson', linestyle=':', lw=1.5, label=f'Majority Baseline ({base_acc:.2f}%)')
        ax_acc.axhline(raw_row_k10['Mean_Accuracy']*100, color='#4a5568', linestyle='-.', lw=1.5, label=f'Raw Features K=10 ({raw_row_k10["Mean_Accuracy"]*100:.2f}%)')
        
        ax_acc.set_title(f"Target '{target}': Held-Out Accuracy vs SVD Dimensions ($k$)", fontsize=13, fontweight='bold')
        ax_acc.set_xlabel('Number of SVD Components ($k$)', fontsize=11)
        ax_acc.set_ylabel('5-Fold CV Accuracy (%)', fontsize=11)
        ax_acc.legend(loc='lower right')
        ax_acc.grid(True, linestyle='--', alpha=0.5)
        
        # Plot Macro F1 vs SVD k
        ax_f1 = axes[i, 1]
        ax_f1.plot(k10_data['SVD_k'], k10_data['Mean_Macro_F1'], 'o-', color='#805ad5', lw=2, label='Cluster K=10')
        ax_f1.plot(k15_data['SVD_k'], k15_data['Mean_Macro_F1'], 's--', color='#dd6b20', lw=2, label='Cluster K=15')
        ax_f1.axhline(base_f1, color='crimson', linestyle=':', lw=1.5, label=f'Majority Baseline ({base_f1:.4f})')
        ax_f1.axhline(raw_row_k10['Mean_Macro_F1'], color='#4a5568', linestyle='-.', lw=1.5, label=f'Raw Features K=10 ({raw_row_k10["Mean_Macro_F1"]:.4f})')
        
        ax_f1.set_title(f"Target '{target}': Held-Out Macro F1 vs SVD Dimensions ($k$)", fontsize=13, fontweight='bold')
        ax_f1.set_xlabel('Number of SVD Components ($k$)', fontsize=11)
        ax_f1.set_ylabel('5-Fold CV Macro F1 Score', fontsize=11)
        ax_f1.legend(loc='lower right')
        ax_f1.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'svd_k_sweep_accuracy_f1.png')
    fig.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved sweep plot to {plot_path}")
    
    # -------------------------------------------------------------
    # Plotting Heatmaps of (SVD k x Cluster K) for Target 'job'
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sub_job = res_df[res_df['Target'] == 'job']
    pivot_acc = sub_job.pivot(index='SVD_Label', columns='Clusters_K', values='Mean_Accuracy') * 100
    pivot_f1 = sub_job.pivot(index='SVD_Label', columns='Clusters_K', values='Mean_Macro_F1')
    
    # Reindex to order by k
    order = [f"k={k}" for k in svd_numeric_k] + ["Full (Raw)"]
    pivot_acc = pivot_acc.reindex(order)
    pivot_f1 = pivot_f1.reindex(order)
    
    im1 = ax1.imshow(pivot_acc.values, cmap='Blues', aspect='auto')
    ax1.set_title("Held-Out Accuracy (%) for 'job'\n(Higher is better)", fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(cluster_K_values)))
    ax1.set_xticklabels([f"K={K}" for K in cluster_K_values])
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels(order)
    plt.colorbar(im1, ax=ax1)
    
    # Annotate values
    for r in range(len(order)):
        for c in range(len(cluster_K_values)):
            val = pivot_acc.values[r, c]
            ax1.text(c, r, f"{val:.1f}%", ha='center', va='center', color='black' if val < 29.5 else 'white', fontsize=9)
            
    im2 = ax2.imshow(pivot_f1.values, cmap='Viridis', aspect='auto')
    ax2.set_title("Held-Out Macro F1 for 'job'\n(Higher is better)", fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(cluster_K_values)))
    ax2.set_xticklabels([f"K={K}" for K in cluster_K_values])
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order)
    plt.colorbar(im2, ax=ax2)
    
    for r in range(len(order)):
        for c in range(len(cluster_K_values)):
            val = pivot_f1.values[r, c]
            ax2.text(c, r, f"{val:.3f}", ha='center', va='center', color='black' if val < 0.08 else 'white', fontsize=9)
            
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'svd_k_sweep_heatmap.png')
    fig.savefig(heatmap_path, dpi=200)
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")
    print("\n=== Sweep Script Complete ===")

if __name__ == '__main__':
    main()
