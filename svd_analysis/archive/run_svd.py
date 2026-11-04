import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, adjusted_mutual_info_score

def run_svd():
    print("=== DS701 Midterm: SVD Matrix Simplification & Analysis ===")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'bank-additional-full.csv')
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, sep=';')
    
    # Use Training partition according to validation plan (first 24,712 rows)
    train_df = df.iloc[:24712].copy()
    print(f"Training partition shape: {train_df.shape}")
    
    # Feature sets - strictly exclude 'duration' and 'y' from SVD matrix
    num_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    feature_cols = num_cols + cat_cols
    
    # Preprocessing: StandardScaler for numerical, OneHotEncoder for categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    
    X_train_raw = train_df[feature_cols]
    X_proc = preprocessor.fit_transform(X_train_raw)
    
    # Center the matrix so SVD corresponds to exact Principal Component axes (PCA)
    X_mean = np.mean(X_proc, axis=0)
    X_centered = X_proc - X_mean
    
    n_samples, n_features = X_centered.shape
    print(f"Preprocessed matrix dimensions: {n_samples} samples x {n_features} features")
    
    # Retrieve feature names
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_feature_names = list(num_cols) + list(cat_feature_names)
    
    # -------------------------------------------------------------
    # 1. Full SVD computation: X_c = U * S * V^T
    # -------------------------------------------------------------
    print("\nComputing full SVD on centered feature matrix...")
    # np.linalg.svd with full_matrices=False computes economy SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Variance calculations
    explained_variance = (s ** 2) / (n_samples - 1)
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print(f"Computed {len(s)} singular values.")
    for thresh in [0.50, 0.75, 0.85, 0.90, 0.95]:
        n_comp = np.argmax(cumulative_variance_ratio >= thresh) + 1
        print(f"Components needed for {int(thresh*100)}% variance: {n_comp} / {n_features} features ({n_comp/n_features*100:.1f}%)")
    
    # -------------------------------------------------------------
    # 2. Scree Plot & Cumulative Variance
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot (Singular values)
    ax1.plot(range(1, len(s) + 1), s, 'o-', color='#2b6cb0', markersize=4, lw=1.5)
    ax1.axvline(x=4, color='crimson', linestyle='--', label='Elbow cut (k=4)')
    ax1.axvline(x=10, color='orange', linestyle=':', label='K=10')
    ax1.set_title('Scree Plot of Singular Values ($\sigma_i$)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Singular Value Index $i$', fontsize=11)
    ax1.set_ylabel('Singular Value $\sigma_i$', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Cumulative Explained Variance
    ax2.plot(range(1, len(s) + 1), cumulative_variance_ratio * 100, 's-', color='#2f855a', markersize=4, lw=1.5)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50% variance')
    ax2.axhline(y=80, color='gray', linestyle=':', alpha=0.7, label='80% variance')
    ax2.axhline(y=90, color='gray', linestyle='-.', alpha=0.7, label='90% variance')
    ax2.set_title('Cumulative Explained Variance Ratio', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Number of Latent SVD Dimensions ($k$)', fontsize=11)
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    scree_path = os.path.join(output_dir, 'scree_and_variance.png')
    fig.savefig(scree_path, dpi=200)
    plt.close()
    print(f"Saved scree plot to {scree_path}")
    
    # -------------------------------------------------------------
    # 3. Component Loadings (Interpreting Latent Dimensions)
    # -------------------------------------------------------------
    print("\nAnalyzing Component Loadings for top 4 SVD axes...")
    components_df = pd.DataFrame(Vt[:4, :], columns=all_feature_names, index=[f'SVD_{i+1}' for i in range(4)])
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    component_titles = [
        "SVD Component 1: Macroeconomic & Contact Regime (Euribor, Emp.Var, Landline vs Cellular)",
        "SVD Component 2: Demographic Life-Stage & Career (Age, Married, Retired vs Single, Student)",
        "SVD Component 3: Socioeconomic Education/Job Class (University & White-Collar vs Basic & Blue-Collar)",
        "SVD Component 4: Marketing Seasonality & Campaign Effort (Month Timing & Campaign Contacts)"
    ]
    
    for i in range(4):
        comp = components_df.iloc[i]
        # Top 7 positive and top 7 negative loadings
        top_pos = comp.nlargest(7)
        top_neg = comp.nsmallest(7)
        top_loadings = pd.concat([top_neg, top_pos]).sort_values()
        
        colors = ['#c53030' if val < 0 else '#2b6cb0' for val in top_loadings.values]
        axes[i].barh(top_loadings.index, top_loadings.values, color=colors)
        axes[i].set_title(component_titles[i], fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Singular Vector Weight ($V_{ij}$)', fontsize=10)
        axes[i].grid(axis='x', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    loadings_path = os.path.join(output_dir, 'component_loadings.png')
    fig.savefig(loadings_path, dpi=200)
    plt.close()
    print(f"Saved component loadings plot to {loadings_path}")
    
    # -------------------------------------------------------------
    # 4. Latent Space Projections (U * S)
    # -------------------------------------------------------------
    print("\nGenerating 2D Latent Space Projections...")
    # Coordinates in SVD space: Z = X_c * V = U * S
    Z = np.dot(U[:, :4], np.diag(s[:4]))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: SVD1 vs SVD2 colored by Subscription (y)
    y_sub = train_df['y'].values
    idx_no = np.where(y_sub == 'no')[0]
    idx_yes = np.where(y_sub == 'yes')[0]
    
    # Plot 'no' first with lower alpha, 'yes' on top
    ax1.scatter(Z[idx_no, 0], Z[idx_no, 1], c='#a0aec0', s=8, alpha=0.3, label='No Subscription')
    ax1.scatter(Z[idx_yes, 0], Z[idx_yes, 1], c='#e53e3e', s=16, alpha=0.8, label='Subscribed (y=yes)')
    ax1.set_title('Latent SVD 1 vs SVD 2: Term Deposit Subscription (y)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('SVD 1: Macroeconomic & Contact Channel Regime', fontsize=11)
    ax1.set_ylabel('SVD 2: Demographic Life-Stage (Age / Marital)', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Plot 2: SVD1 vs SVD3 colored by Education Level
    edu_sub = train_df['education'].values
    edu_groups = {
        'University Degree': np.where(edu_sub == 'university.degree')[0],
        'High School': np.where(edu_sub == 'high.school')[0],
        'Basic Schooling': np.where(np.isin(edu_sub, ['basic.4y', 'basic.6y', 'basic.9y']))[0],
        'Professional Course': np.where(edu_sub == 'professional.course')[0]
    }
    edu_colors = {'University Degree': '#3182ce', 'High School': '#dd6b20', 'Basic Schooling': '#38a169', 'Professional Course': '#805ad5'}
    for name, idxs in edu_groups.items():
        ax2.scatter(Z[idxs, 0], Z[idxs, 2], s=10, alpha=0.4, label=name, color=edu_colors[name])
    ax2.set_title('Latent SVD 1 vs SVD 3: Education Level Separation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('SVD 1: Macroeconomic & Contact Channel Regime', fontsize=11)
    ax2.set_ylabel('SVD 3: Socioeconomic / Education Class', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    # Plot 3: SVD2 vs SVD3 colored by Job Type (Top 3 jobs)
    job_sub = train_df['job'].values
    job_groups = {
        'admin.': (np.where(job_sub == 'admin.')[0], '#2b6cb0'),
        'blue-collar': (np.where(job_sub == 'blue-collar')[0], '#d69e2e'),
        'technician': (np.where(job_sub == 'technician')[0], '#319795'),
        'retired': (np.where(job_sub == 'retired')[0], '#e53e3e'),
        'student': (np.where(job_sub == 'student')[0], '#9f7aea')
    }
    for name, (idxs, col) in job_groups.items():
        ax3.scatter(Z[idxs, 1], Z[idxs, 2], s=12, alpha=0.5, label=name, color=col)
    ax3.set_title('Latent SVD 2 vs SVD 3: Occupational Segments', fontsize=12, fontweight='bold')
    ax3.set_xlabel('SVD 2: Demographic Life-Stage (Age / Marital)', fontsize=11)
    ax3.set_ylabel('SVD 3: Socioeconomic / Education Class', fontsize=11)
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.4)
    
    # Plot 4: SVD1 vs SVD2 colored by Previous Marketing Outcome
    pout_sub = train_df['poutcome'].values
    pout_groups = {
        'nonexistent': (np.where(pout_sub == 'nonexistent')[0], '#cbd5e0'),
        'failure': (np.where(pout_sub == 'failure')[0], '#e53e3e'),
        'success': (np.where(pout_sub == 'success')[0], '#38a169')
    }
    for name, (idxs, col) in pout_groups.items():
        ax4.scatter(Z[idxs, 0], Z[idxs, 1], s=14, alpha=0.6, label=f'poutcome: {name}', color=col)
    ax4.set_title('Latent SVD 1 vs SVD 2: Previous Campaign Outcome', fontsize=12, fontweight='bold')
    ax4.set_xlabel('SVD 1: Macroeconomic & Contact Channel Regime', fontsize=11)
    ax4.set_ylabel('SVD 2: Demographic Life-Stage (Age / Marital)', fontsize=11)
    ax4.legend(loc='upper right')
    ax4.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    proj_path = os.path.join(output_dir, 'latent_projections.png')
    fig.savefig(proj_path, dpi=200)
    plt.close()
    print(f"Saved projections plot to {proj_path}")
    
    # -------------------------------------------------------------
    # 5. Low-Rank Matrix Reconstruction Analysis
    # -------------------------------------------------------------
    print("\nAnalyzing Low-Rank Matrix Reconstruction...")
    # Relative Frobenius reconstruction error = ||X - X_k||_F / ||X||_F = sqrt(sum_{i=k+1}^d s_i^2 / sum_{i=1}^d s_i^2)
    ranks = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]
    total_frob_sq = np.sum(s ** 2)
    recon_errors = []
    
    for k in ranks:
        res_sq = np.sum(s[k:] ** 2)
        rel_frob_err = np.sqrt(res_sq / total_frob_sq)
        recon_errors.append(rel_frob_err)
        print(f"  Rank k={k:2d}: Relative Frobenius Reconstruction Error = {rel_frob_err*100:5.2f}% (Energy preserved: {(1 - rel_frob_err**2)*100:5.2f}%)")
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks, [e * 100 for e in recon_errors], 'o-', color='#dd6b20', lw=2)
    ax.set_title('Matrix Reconstruction Error vs SVD Truncation Rank $k$', fontsize=13, fontweight='bold')
    ax.set_xlabel('Truncation Rank $k$', fontsize=11)
    ax.set_ylabel('Relative Frobenius Error (%)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    recon_path = os.path.join(output_dir, 'reconstruction_error.png')
    fig.savefig(recon_path, dpi=200)
    plt.close()
    print(f"Saved reconstruction error plot to {recon_path}")
    
    # -------------------------------------------------------------
    # 6. Downstream Evaluation: SVD Matrix Simplification for Feature Prediction
    # -------------------------------------------------------------
    print("\n--- Downstream 5-Fold CV: Predicting 'job' and 'education' using Low-Rank SVD ---")
    # Can we replace all 50+ features with just top k SVD components to predict job?
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    eval_records = []
    
    for target in ['job', 'education']:
        curr_num = [c for c in num_cols if c != target]
        curr_cat = [c for c in cat_cols if c != target]
        curr_feats = curr_num + curr_cat
        
        y_target = train_df[target]
        base_acc = (y_target == y_target.mode()[0]).mean()
        
        # Test full raw vs SVD ranks
        feature_configs = [
            ('Raw Features (Full)', None),
            ('SVD Rank k=5', 5),
            ('SVD Rank k=10', 10),
            ('SVD Rank k=15', 15),
            ('SVD Rank k=25', 25)
        ]
        
        for name, rank in feature_configs:
            fold_accs = []
            fold_f1s = []
            
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
                
                if rank is not None:
                    # Fit TruncatedSVD on train fold only!
                    svd = TruncatedSVD(n_components=rank, random_state=42)
                    X_tr_in = svd.fit_transform(X_tr_p)
                    X_va_in = svd.transform(X_va_p)
                else:
                    X_tr_in = X_tr_p
                    X_va_in = X_va_p
                    
                # Fit KMeans (K=10) on the input space
                km = KMeans(n_clusters=10, random_state=42, n_init=10).fit(X_tr_in)
                train_cl = km.labels_
                
                # Assign cluster mode
                c_modes = {}
                for c in range(10):
                    m = (train_cl == c)
                    c_modes[c] = y_tr[m].mode()[0] if m.sum() > 0 else y_tr.mode()[0]
                    
                val_cl = km.predict(X_va_in)
                val_preds = [c_modes[c] for c in val_cl]
                
                fold_accs.append(accuracy_score(y_va, val_preds))
                fold_f1s.append(f1_score(y_va, val_preds, average='macro', zero_division=0))
                
            mean_acc = np.mean(fold_accs)
            mean_f1 = np.mean(fold_f1s)
            gain = mean_acc - base_acc
            print(f"Target '{target}' | Config: {name:20s} | Acc: {mean_acc*100:5.2f}% | Gain: {gain*100:+5.2f}% | Macro F1: {mean_f1:.4f}")
            eval_records.append({
                'Target': target,
                'Config': name,
                'Rank': rank if rank is not None else X_tr_p.shape[1],
                'Mean_Accuracy': mean_acc,
                'Baseline_Accuracy': base_acc,
                'Accuracy_Gain': gain,
                'Macro_F1': mean_f1
            })
            
    eval_df = pd.DataFrame(eval_records)
    csv_path = os.path.join(output_dir, 'svd_downstream_eval.csv')
    eval_df.to_csv(csv_path, index=False)
    print(f"\nSaved downstream evaluation results to {csv_path}")

    # -------------------------------------------------------------
    # 7. Write Comprehensive Documentation in README.md
    # -------------------------------------------------------------
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# SVD Matrix Simplification & Latent Structure Analysis\n\n")
        f.write("## Overview\n")
        f.write("This analysis applies **Singular Value Decomposition (SVD)** to the Bank Marketing feature matrix ")
        f.write("(strictly excluding `duration` and `y` from decomposition) to test whether we can simplify the matrix ")
        f.write("and extract meaningful latent customer and economic representations.\n\n")
        f.write("## Key Visualizations & Artifacts\n")
        f.write("- **`scree_and_variance.png`**: Scree plot of singular values and cumulative explained variance.\n")
        f.write("- **`component_loadings.png`**: Feature loading profiles across the top 4 latent singular dimensions.\n")
        f.write("- **`latent_projections.png`**: 2D scatter plots of latent spaces colored by `y`, `education`, `job`, and `poutcome`.\n")
        f.write("- **`reconstruction_error.png`**: Relative Frobenius norm reconstruction error vs. truncation rank $k$.\n")
        f.write("- **`svd_downstream_eval.csv`**: 5-fold cross-validation performance comparing full raw feature matrices against low-rank SVD projections.\n\n")
        f.write("## Mathematical Formulations\n")
        f.write("The centered feature matrix $X_c \in \mathbb{R}^{n \\times d}$ is decomposed into:\n")
        f.write("$$X_c = U \Sigma V^T$$\n")
        f.write("where $U$ contains left singular vectors (sample coordinates), $\Sigma$ is diagonal singular values, and $V^T$ represents the orthonormal basis of feature loadings.\n")
        f.write("The low-rank projection is given by $Z_k = X_c V_k = U_k \Sigma_k$.\n")
    print(f"Generated documentation at {readme_path}")
    print("\n=== SVD Analysis Completed Successfully! ===")

if __name__ == '__main__':
    run_svd()
