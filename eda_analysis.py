"""
Exploratory Data Analysis for LOB Price Prediction Competition
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
DATA_DIR = Path("./datasets")
OUTPUT_DIR = Path("./eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("LOB Price Prediction - Exploratory Data Analysis")
print("=" * 60)

# Load data
print("\n📂 Loading datasets...")
train_df = pd.read_parquet(DATA_DIR / "train.parquet")
valid_df = pd.read_parquet(DATA_DIR / "valid.parquet")

print(f"\n{'='*60}")
print("1. DATASET OVERVIEW")
print("="*60)

print(f"\n📊 Training Data Shape: {train_df.shape}")
print(f"📊 Validation Data Shape: {valid_df.shape}")

print(f"\n📋 Columns ({len(train_df.columns)}):")
print(train_df.columns.tolist())

print(f"\n📋 Data Types:")
print(train_df.dtypes.value_counts())

# Sequence analysis
print(f"\n{'='*60}")
print("2. SEQUENCE ANALYSIS")
print("="*60)

n_train_seqs = train_df['seq_ix'].nunique()
n_valid_seqs = valid_df['seq_ix'].nunique()
steps_per_seq = train_df.groupby('seq_ix')['step_in_seq'].max().mode()[0] + 1

print(f"\n🔢 Number of Training Sequences: {n_train_seqs:,}")
print(f"🔢 Number of Validation Sequences: {n_valid_seqs:,}")
print(f"🔢 Steps per Sequence: {steps_per_seq}")
print(f"🔢 Total Training Rows: {len(train_df):,}")
print(f"🔢 Total Validation Rows: {len(valid_df):,}")

# Prediction requirements
train_need_pred = train_df['need_prediction'].sum()
valid_need_pred = valid_df['need_prediction'].sum()
print(f"\n✅ Training Predictions Required: {train_need_pred:,} ({100*train_need_pred/len(train_df):.1f}%)")
print(f"✅ Validation Predictions Required: {valid_need_pred:,} ({100*valid_need_pred/len(valid_df):.1f}%)")

# Feature columns
feature_cols = train_df.columns[3:35].tolist()
target_cols = ['t0', 't1']
print(f"\n📊 Feature Columns ({len(feature_cols)}): {feature_cols}")
print(f"🎯 Target Columns: {target_cols}")

# Basic statistics
print(f"\n{'='*60}")
print("3. FEATURE STATISTICS")
print("="*60)

print("\n📈 Training Data Statistics:")
print(train_df[feature_cols + target_cols].describe().round(4).to_string())

# Check for missing values
print(f"\n{'='*60}")
print("4. DATA QUALITY")
print("="*60)

train_missing = train_df.isnull().sum().sum()
valid_missing = valid_df.isnull().sum().sum()
print(f"\n❓ Missing Values in Training: {train_missing}")
print(f"❓ Missing Values in Validation: {valid_missing}")

# Check for infinities
train_inf = np.isinf(train_df[feature_cols + target_cols].values).sum()
valid_inf = np.isinf(valid_df[feature_cols + target_cols].values).sum()
print(f"♾️ Infinite Values in Training: {train_inf}")
print(f"♾️ Infinite Values in Validation: {valid_inf}")

# Target analysis
print(f"\n{'='*60}")
print("5. TARGET ANALYSIS")
print("="*60)

for target in target_cols:
    print(f"\n🎯 {target}:")
    for name, df in [("Train", train_df), ("Valid", valid_df)]:
        vals = df[target].values
        print(f"  {name}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}")

# Target correlation
train_target_corr = train_df['t0'].corr(train_df['t1'])
print(f"\n📊 Correlation between t0 and t1 (Train): {train_target_corr:.4f}")

# Visualization 1: Target Distributions
print(f"\n{'='*60}")
print("6. GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# t0 distribution
axes[0, 0].hist(train_df['t0'], bins=100, alpha=0.7, label='Train', density=True)
axes[0, 0].hist(valid_df['t0'], bins=100, alpha=0.7, label='Valid', density=True)
axes[0, 0].set_xlabel('t0')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Distribution of t0')
axes[0, 0].legend()
axes[0, 0].set_xlim(-10, 10)

# t1 distribution
axes[0, 1].hist(train_df['t1'], bins=100, alpha=0.7, label='Train', density=True)
axes[0, 1].hist(valid_df['t1'], bins=100, alpha=0.7, label='Valid', density=True)
axes[0, 1].set_xlabel('t1')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Distribution of t1')
axes[0, 1].legend()
axes[0, 1].set_xlim(-10, 10)

# t0 vs t1 scatter (sample)
sample_idx = np.random.choice(len(train_df), min(10000, len(train_df)), replace=False)
axes[1, 0].scatter(train_df['t0'].iloc[sample_idx], train_df['t1'].iloc[sample_idx], 
                   alpha=0.1, s=1)
axes[1, 0].set_xlabel('t0')
axes[1, 0].set_ylabel('t1')
axes[1, 0].set_title(f't0 vs t1 Scatter (r={train_target_corr:.3f})')

# Target by step position
step_means = train_df.groupby('step_in_seq')[target_cols].mean()
axes[1, 1].plot(step_means.index, step_means['t0'], label='t0', alpha=0.7)
axes[1, 1].plot(step_means.index, step_means['t1'], label='t1', alpha=0.7)
axes[1, 1].axvline(x=99, color='red', linestyle='--', alpha=0.5, label='Prediction Start')
axes[1, 1].set_xlabel('Step in Sequence')
axes[1, 1].set_ylabel('Mean Target Value')
axes[1, 1].set_title('Mean Target by Step Position')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'target_distributions.png', dpi=150, bbox_inches='tight')
print("✅ Saved: target_distributions.png")
plt.close()

# Visualization 2: Feature Distributions
fig, axes = plt.subplots(4, 8, figsize=(20, 12))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    axes[i].hist(train_df[col], bins=50, alpha=0.7, density=True)
    axes[i].set_title(col, fontsize=9)
    axes[i].tick_params(labelsize=7)

plt.suptitle('Feature Distributions (Training Data)', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_distributions.png', dpi=150, bbox_inches='tight')
print("✅ Saved: feature_distributions.png")
plt.close()

# Visualization 3: Feature-Target Correlations
print("\n📊 Computing feature-target correlations...")
correlations = train_df[feature_cols + target_cols].corr()
feature_target_corr = correlations.loc[feature_cols, target_cols]

fig, axes = plt.subplots(1, 2, figsize=(14, 10))

# Correlation heatmap for t0
feature_target_corr_sorted = feature_target_corr.sort_values('t0', ascending=False)
sns.barplot(x=feature_target_corr_sorted['t0'], y=feature_target_corr_sorted.index, ax=axes[0])
axes[0].set_xlabel('Correlation with t0')
axes[0].set_title('Feature Correlations with t0')
axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Correlation heatmap for t1
feature_target_corr_sorted = feature_target_corr.sort_values('t1', ascending=False)
sns.barplot(x=feature_target_corr_sorted['t1'], y=feature_target_corr_sorted.index, ax=axes[1])
axes[1].set_xlabel('Correlation with t1')
axes[1].set_title('Feature Correlations with t1')
axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_target_correlations.png', dpi=150, bbox_inches='tight')
print("✅ Saved: feature_target_correlations.png")
plt.close()

# Visualization 4: Feature Correlation Heatmap
fig, ax = plt.subplots(figsize=(16, 14))
feature_corr = correlations.loc[feature_cols, feature_cols]
mask = np.triu(np.ones_like(feature_corr, dtype=bool))
sns.heatmap(feature_corr, mask=mask, cmap='RdBu_r', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.5},
            annot=False, ax=ax)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_correlation_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Saved: feature_correlation_matrix.png")
plt.close()

# Visualization 5: Sample Sequences
print("\n📊 Visualizing sample sequences...")
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# Pick 3 random sequences
sample_seqs = np.random.choice(train_df['seq_ix'].unique(), 3, replace=False)

for i, seq_ix in enumerate(sample_seqs):
    seq_data = train_df[train_df['seq_ix'] == seq_ix].sort_values('step_in_seq')
    
    # Plot features (subset)
    for col in ['p0', 'p6', 'v0', 'v6']:
        axes[i, 0].plot(seq_data['step_in_seq'], seq_data[col], alpha=0.7, label=col)
    axes[i, 0].set_xlabel('Step')
    axes[i, 0].set_ylabel('Feature Value')
    axes[i, 0].set_title(f'Sequence {seq_ix} - Key Features')
    axes[i, 0].legend(fontsize=8)
    axes[i, 0].axvline(x=99, color='red', linestyle='--', alpha=0.5)
    
    # Plot targets
    axes[i, 1].plot(seq_data['step_in_seq'], seq_data['t0'], alpha=0.7, label='t0')
    axes[i, 1].plot(seq_data['step_in_seq'], seq_data['t1'], alpha=0.7, label='t1')
    axes[i, 1].set_xlabel('Step')
    axes[i, 1].set_ylabel('Target Value')
    axes[i, 1].set_title(f'Sequence {seq_ix} - Targets')
    axes[i, 1].legend(fontsize=8)
    axes[i, 1].axvline(x=99, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sample_sequences.png', dpi=150, bbox_inches='tight')
print("✅ Saved: sample_sequences.png")
plt.close()

# Visualization 6: Target Volatility/Amplitude Analysis (Important for weighted metric)
print("\n📊 Analyzing target amplitude distribution (critical for weighted metric)...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Amplitude distributions
for i, target in enumerate(target_cols):
    abs_vals = np.abs(train_df[target])
    axes[0, i].hist(abs_vals, bins=100, alpha=0.7, density=True)
    axes[0, i].set_xlabel(f'|{target}|')
    axes[0, i].set_ylabel('Density')
    axes[0, i].set_title(f'Amplitude Distribution of {target}')
    
    # Log scale for better visibility
    axes[0, i].set_yscale('log')
    
    # Statistics
    mean_amp = abs_vals.mean()
    median_amp = abs_vals.median()
    axes[0, i].axvline(mean_amp, color='red', linestyle='--', label=f'Mean: {mean_amp:.3f}')
    axes[0, i].axvline(median_amp, color='green', linestyle='--', label=f'Median: {median_amp:.3f}')
    axes[0, i].legend()

# Weight distribution implications
weights_t0 = np.abs(train_df['t0'])
weights_t1 = np.abs(train_df['t1'])

# Cumulative weight contribution
for i, (target, weights) in enumerate([('t0', weights_t0), ('t1', weights_t1)]):
    sorted_weights = np.sort(weights)[::-1]
    cumsum = np.cumsum(sorted_weights) / sorted_weights.sum()
    x_pct = np.arange(len(cumsum)) / len(cumsum) * 100
    
    axes[1, i].plot(x_pct, cumsum * 100)
    axes[1, i].set_xlabel('% of Data Points (sorted by weight)')
    axes[1, i].set_ylabel('% of Total Weight')
    axes[1, i].set_title(f'{target}: Cumulative Weight Distribution')
    axes[1, i].axhline(80, color='red', linestyle='--', alpha=0.5)
    axes[1, i].axhline(50, color='orange', linestyle='--', alpha=0.5)
    
    # Find percentage of points contributing 80% of weight
    pct_80 = x_pct[np.argmax(cumsum >= 0.8)]
    axes[1, i].axvline(pct_80, color='red', linestyle='--', alpha=0.5)
    axes[1, i].annotate(f'{pct_80:.1f}% points → 80% weight', 
                        xy=(pct_80, 80), fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'target_amplitude_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved: target_amplitude_analysis.png")
plt.close()

# Print key findings
print(f"\n{'='*60}")
print("7. KEY FINDINGS SUMMARY")
print("="*60)

print("\n📌 Feature-Target Correlations (Top 5 for each target):")
for target in target_cols:
    top_corrs = feature_target_corr[target].abs().sort_values(ascending=False).head(5)
    print(f"\n  {target}:")
    for feat, corr in top_corrs.items():
        sign = '+' if feature_target_corr.loc[feat, target] > 0 else '-'
        print(f"    {feat}: {sign}{corr:.4f}")

print("\n📌 Target Statistics:")
for target in target_cols:
    vals = train_df[target]
    print(f"\n  {target}:")
    print(f"    Range: [{vals.min():.3f}, {vals.max():.3f}]")
    print(f"    Mean: {vals.mean():.4f}, Std: {vals.std():.4f}")
    print(f"    Skewness: {vals.skew():.4f}")
    print(f"    Kurtosis: {vals.kurtosis():.4f}")
    
    # Percentage in high-weight region
    high_weight_pct = (np.abs(vals) > vals.std()).mean() * 100
    print(f"    % with |value| > 1 std: {high_weight_pct:.1f}%")

print(f"\n{'='*60}")
print(f"✅ EDA Complete! Visualizations saved to: {OUTPUT_DIR}")
print("="*60)
