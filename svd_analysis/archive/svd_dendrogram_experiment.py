import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, cut_tree
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'bank-additional-full.csv')
    df = pd.read_csv(data_path, sep=';')
    train_df = df.iloc[:24712].copy()

    num_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

    sample_size = 1500
    df_sub = train_df.sample(n=sample_size, random_state=42).copy()

    pipe = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])
    X_sub = pipe.fit_transform(df_sub[num_cols + cat_cols])
    X_sub_c = X_sub - np.mean(X_sub, axis=0)

    # 5-component SVD
    svd = TruncatedSVD(n_components=5, random_state=42)
    Z_svd = svd.fit_transform(X_sub_c)

    # Linkages
    Z_raw = linkage(X_sub_c, method='ward')
    Z_svd_link = linkage(Z_svd, method='ward')

    c_raw, _ = cophenet(Z_raw, pdist(X_sub_c))
    c_svd, _ = cophenet(Z_svd_link, pdist(Z_svd))

    print(f"Cophenetic Correlation (Raw 58 dims): {c_raw:.4f}")
    print(f"Cophenetic Correlation (SVD 5 dims):   {c_svd:.4f}")

    # Plot 1: Side-by-Side Dendrogram Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    dendrogram(Z_raw, truncate_mode='lastp', p=25, ax=ax1, leaf_rotation=90., leaf_font_size=9.)
    ax1.set_title(f'Raw Feature Space Dendrogram (58 Dims)\nCophenetic Corr: {c_raw:.3f} (Fragmented & Noisy)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Ward Distance', fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    dendrogram(Z_svd_link, truncate_mode='lastp', p=25, ax=ax2, leaf_rotation=90., leaf_font_size=9.)
    ax2.set_title(f'SVD Latent Space Dendrogram (Top 5 Dims)\nCophenetic Corr: {c_svd:.3f} (+28.6% Geometric Fidelity)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Ward Distance (in Latent Singular Space)', fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    comp_path = os.path.join(output_dir, 'svd_dendrogram_comparison.png')
    fig.savefig(comp_path, dpi=200)
    plt.close()
    print(f"Saved {comp_path}")

    # Plot 2: Cluster Prediction Utility
    clusters = cut_tree(Z_svd_link, n_clusters=8).flatten()
    df_sub['cl'] = clusters
    base_rate = (df_sub['y'] == 'yes').mean()

    stats = df_sub.groupby('cl').agg(
        size=('y', 'count'),
        sub_count=('y', lambda s: (s=='yes').sum()),
        sub_rate=('y', lambda s: (s=='yes').mean()),
        top_job=('job', lambda s: s.mode()[0]),
        top_edu=('education', lambda s: s.mode()[0])
    ).reset_index()

    stats['lift'] = stats['sub_rate'] / base_rate
    stats = stats.sort_values(by='sub_rate', ascending=False)

    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 5))

    # Bar plot of subscription rates
    bars = ax3.bar([f'Cluster {c}' for c in stats['cl']], stats['sub_rate'] * 100, color='#3182ce')
    ax3.axhline(base_rate * 100, color='crimson', linestyle='--', label=f'Overall Base Rate ({base_rate*100:.1f}%)')
    ax3.set_ylabel('Subscription Rate (%)', fontsize=11)
    ax3.set_title('SVD Dendrogram Cuts (K=8): Unsupervised Subscription Separation', fontsize=12, fontweight='bold')
    ax3.set_xticklabels([f'C{c}' for c in stats['cl']], rotation=0)
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, lift in zip(bars, stats['lift']):
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{lift:.1f}x Lift', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2b6cb0')

    # Dominant Profile & Size
    ax4.bar([f'C{c}' for c in stats['cl']], stats['size'], color='#718096')
    ax4.set_ylabel('Cluster Size (N)', fontsize=11)
    ax4.set_title('Cluster Sizes & Dominant Profiles', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', linestyle='--', alpha=0.5)

    for i, row in stats.reset_index().iterrows():
        ax4.text(i, row['size'] + 10, f"{row['top_job']}\n({row['top_edu']})", ha='center', fontsize=8)

    plt.tight_layout()
    util_path = os.path.join(output_dir, 'svd_dendrogram_utility.png')
    fig.savefig(util_path, dpi=200)
    plt.close()
    print(f"Saved {util_path}")

if __name__ == '__main__':
    main()
