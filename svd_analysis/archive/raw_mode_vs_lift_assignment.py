import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'bank-additional-full.csv')
    df = pd.read_csv(data_path, sep=';')
    train_df = df.iloc[:24712].copy()
    
    num_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    cat_cols = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    pipe = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])
    X_p = pipe.fit_transform(train_df[num_cols + cat_cols])
    X_c = X_p - np.mean(X_p, axis=0)
    
    svd = TruncatedSVD(n_components=5, random_state=42)
    Z = svd.fit_transform(X_c)
    
    km = KMeans(n_clusters=10, random_state=42, n_init=10).fit(Z)
    train_df['cl'] = km.labels_
    
    ct = pd.crosstab(train_df['cl'], train_df['job'], normalize='index')
    base_rates = train_df['job'].value_counts(normalize=True)
    lift_matrix = ct.div(base_rates, axis=1)
    
    raw_modes = [ct.loc[c].idxmax() for c in range(10)]
    lift_modes = [lift_matrix.loc[c].idxmax() for c in range(10)]
    lift_vals = [lift_matrix.loc[c, lift_modes[c]] for c in range(10)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw Mode Counts (Frequency of predicted classes)
    raw_counts = pd.Series(raw_modes).value_counts()
    ax1.bar(raw_counts.index, raw_counts.values, color=['#3182ce', '#dd6b20', '#38a169'], width=0.5)
    ax1.set_title("Standard Plurality / Raw Mode Assignment\n(Trapped by Global Base-Rate: Only 3 of 12 Classes Predicted)", 
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel("Number of Assigned Clusters (out of 10)", fontsize=11)
    ax1.set_xlabel("Predicted Job Class", fontsize=11)
    ax1.set_ylim(0, 7)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(raw_counts.values):
        ax1.text(i, v + 0.15, f"{v} clusters", ha='center', fontweight='bold', fontsize=10)
        
    # Plot 2: Relative Lift Assignment (Affinity to over-represented classes)
    lift_counts = pd.Series(lift_modes).value_counts()
    colors = plt.cm.tab10(np.linspace(0, 1, len(lift_counts)))
    ax2.bar(lift_counts.index, lift_counts.values, color=colors, width=0.5)
    ax2.set_title("Relative Prevalence Lift Assignment (Affinity / Over-representation)\n(Unlocks 6+ Classes by Normalizing for Baseline Rarity)", 
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel("Number of Assigned Clusters (out of 10)", fontsize=11)
    ax2.set_xlabel("Assigned Job Class via Maximum Lift", fontsize=11)
    ax2.set_xticklabels(lift_counts.index, rotation=25, ha='right', fontsize=9.5)
    ax2.set_ylim(0, 7)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(lift_counts.values):
        ax2.text(i, v + 0.15, f"{v} clusters", ha='center', fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'raw_mode_vs_lift_assignment.png')
    fig.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved {fig_path}")

if __name__ == '__main__':
    main()
