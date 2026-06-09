# ============================================================
#   TITANIC DATASET - EXPLORATORY DATA ANALYSIS (EDA)
#   Internship Project 1
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

OUTPUT_DIR = "/home/claude/"

# ============================================================
# STEP 1 — LOAD & INSPECT
# ============================================================
print("=" * 60)
print("  STEP 1: LOADING & INSPECTING DATA")
print("=" * 60)

df = sns.load_dataset('titanic')

print(f"\n📦 Dataset Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
print("\n📋 Column Data Types:")
print(df.dtypes.to_string())
print("\n❓ Missing Values per Column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0].to_string())
print("\n📊 Basic Statistics:")
print(df.describe().round(2).to_string())

# ============================================================
# STEP 2 — CLEAN & PREPROCESS
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2: CLEANING DATA")
print("=" * 60)

# Fix missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)

# Drop high-missingness column
df.drop(columns=['deck'], inplace=True)

# Create age buckets
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=['Child\n(0–12)', 'Teen\n(13–18)', 'Young Adult\n(19–35)',
            'Adult\n(36–60)', 'Senior\n(61+)']
)

print("✅ Filled missing 'age' with median:", round(df['age'].median(), 1))
print("✅ Filled missing 'embarked' with mode:", df['embarked'].mode()[0])
print("✅ Dropped 'deck' column (too many missing values)")
print("✅ Created 'age_group' column with 5 buckets")
print(f"\n🔍 Remaining missing values: {df.isnull().sum().sum()}")

# ============================================================
# STEP 3 — SURVIVAL ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("  STEP 3: SURVIVAL RATE ANALYSIS")
print("=" * 60)

overall = df['survived'].mean() * 100
print(f"\n🌍 Overall Survival Rate : {overall:.1f}%")

sex_surv = df.groupby('sex')['survived'].mean() * 100
print("\n👤 Survival Rate by Sex:")
for s, v in sex_surv.items():
    print(f"   {s.capitalize():<10}: {v:.1f}%")

class_surv = df.groupby('pclass')['survived'].mean() * 100
print("\n🎟️  Survival Rate by Passenger Class:")
for c, v in class_surv.items():
    print(f"   Class {c}    : {v:.1f}%")

age_surv = df.groupby('age_group', observed=False)['survived'].mean() * 100
print("\n🎂 Survival Rate by Age Group:")
for a, v in age_surv.items():
    print(f"   {str(a):<22}: {v:.1f}%")

combined = df.groupby(['sex', 'pclass'])['survived'].mean() * 100
print("\n🔀 Survival Rate by Sex × Class:")
print(combined.unstack().round(1).to_string())

# ============================================================
# STEP 4 — VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("  STEP 4: GENERATING VISUALIZATIONS")
print("=" * 60)

COLORS = {
    'survived':     '#2ecc71',
    'not_survived': '#e74c3c',
    'male':         '#3498db',
    'female':       '#e91e8c',
    'class':        ['#1abc9c', '#f39c12', '#e74c3c'],
    'age':          ['#9b59b6', '#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
}

# ── Figure 1: Overview Dashboard ────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Titanic EDA — Overview Dashboard', fontsize=16, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1a Overall survival count
ax1 = fig.add_subplot(gs[0, 0])
counts = df['survived'].value_counts()
bars = ax1.bar(['Did Not\nSurvive', 'Survived'], counts.values,
               color=[COLORS['not_survived'], COLORS['survived']], width=0.5, edgecolor='white')
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
             str(val), ha='center', fontweight='bold', fontsize=11)
ax1.set_title('Overall Survival Count')
ax1.set_ylabel('Number of Passengers')
ax1.set_ylim(0, max(counts.values) * 1.15)

# 1b Survival rate by sex
ax2 = fig.add_subplot(gs[0, 1])
sex_vals = sex_surv.reset_index()
bars2 = ax2.bar(sex_vals['sex'].str.capitalize(), sex_vals['survived'],
                color=[COLORS['male'], COLORS['female']], width=0.5, edgecolor='white')
for bar, val in zip(bars2, sex_vals['survived']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax2.set_title('Survival Rate by Sex')
ax2.set_ylabel('Survival Rate (%)')
ax2.set_ylim(0, 100)
ax2.axhline(overall, color='gray', linestyle='--', linewidth=1, label=f'Overall {overall:.1f}%')
ax2.legend(fontsize=9)

# 1c Survival rate by class
ax3 = fig.add_subplot(gs[0, 2])
class_vals = class_surv.reset_index()
bars3 = ax3.bar([f'Class {c}' for c in class_vals['pclass']], class_vals['survived'],
                color=COLORS['class'], width=0.5, edgecolor='white')
for bar, val in zip(bars3, class_vals['survived']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax3.set_title('Survival Rate by Passenger Class')
ax3.set_ylabel('Survival Rate (%)')
ax3.set_ylim(0, 100)
ax3.axhline(overall, color='gray', linestyle='--', linewidth=1, label=f'Overall {overall:.1f}%')
ax3.legend(fontsize=9)

# 1d Survival rate by age group
ax4 = fig.add_subplot(gs[1, 0])
age_vals = age_surv.reset_index()
bars4 = ax4.bar(age_vals['age_group'].astype(str), age_vals['survived'],
                color=COLORS['age'], width=0.6, edgecolor='white')
for bar, val in zip(bars4, age_vals['survived']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)
ax4.set_title('Survival Rate by Age Group')
ax4.set_ylabel('Survival Rate (%)')
ax4.set_ylim(0, 100)
ax4.axhline(overall, color='gray', linestyle='--', linewidth=1)

# 1e Embarkation survival
ax5 = fig.add_subplot(gs[1, 1])
emb_surv = df.groupby('embark_town')['survived'].mean() * 100
bars5 = ax5.bar(emb_surv.index, emb_surv.values,
                color=['#16a085', '#8e44ad', '#d35400'], width=0.5, edgecolor='white')
for bar, val in zip(bars5, emb_surv.values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax5.set_title('Survival Rate by Embarkation Town')
ax5.set_ylabel('Survival Rate (%)')
ax5.set_ylim(0, 100)

# 1f Passenger class distribution
ax6 = fig.add_subplot(gs[1, 2])
class_counts = df['pclass'].value_counts().sort_index()
wedges, texts, autotexts = ax6.pie(
    class_counts.values,
    labels=[f'Class {c}' for c in class_counts.index],
    autopct='%1.1f%%',
    colors=COLORS['class'],
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
ax6.set_title('Passenger Class Distribution')

plt.savefig(OUTPUT_DIR + 'fig1_overview_dashboard.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Saved: fig1_overview_dashboard.png")

# ── Figure 2: Age & Distribution Analysis ───────────────────
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Titanic EDA — Age & Distribution Analysis', fontsize=14, fontweight='bold')

# 2a Violin plot — age by survival
sns.violinplot(x='survived', y='age', data=df, palette=[COLORS['not_survived'], COLORS['survived']],
               inner='box', ax=axes[0])
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Did Not Survive', 'Survived'])
axes[0].set_title('Age Distribution by Survival\n(Violin Plot)')
axes[0].set_ylabel('Age')

# 2b Boxplot — age by class and survival
sns.boxplot(x='pclass', y='age', hue='survived', data=df,
            palette=[COLORS['not_survived'], COLORS['survived']], ax=axes[1])
axes[1].set_title('Age by Class & Survival\n(Box Plot)')
axes[1].set_xlabel('Passenger Class')
axes[1].set_ylabel('Age')
axes[1].legend(title='Survived', labels=['No', 'Yes'])

# 2c Age histogram
axes[2].hist(df[df['survived'] == 0]['age'], bins=25, alpha=0.7,
             color=COLORS['not_survived'], label='Did Not Survive', edgecolor='white')
axes[2].hist(df[df['survived'] == 1]['age'], bins=25, alpha=0.7,
             color=COLORS['survived'], label='Survived', edgecolor='white')
axes[2].set_title('Age Distribution Histogram')
axes[2].set_xlabel('Age')
axes[2].set_ylabel('Count')
axes[2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'fig2_age_distribution.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Saved: fig2_age_distribution.png")

# ── Figure 3: Heatmaps ──────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Titanic EDA — Heatmap Analysis', fontsize=14, fontweight='bold')

# 3a Correlation heatmap
corr = df.select_dtypes(include=np.number).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[0], linewidths=0.5, cbar_kws={'shrink': 0.8})
axes[0].set_title('Correlation Heatmap\n(Numeric Features)')

# 3b Survival pivot — sex × class
pivot = df.pivot_table(values='survived', index='sex', columns='pclass', aggfunc='mean') * 100
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
            ax=axes[1], linewidths=0.5, cbar_kws={'label': 'Survival Rate (%)', 'shrink': 0.8},
            vmin=0, vmax=100)
axes[1].set_title('Survival Rate (%) Heatmap\nSex × Passenger Class')
axes[1].set_xlabel('Passenger Class')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'fig3_heatmaps.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Saved: fig3_heatmaps.png")

# ── Figure 4: Grouped Analysis ──────────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle('Titanic EDA — Grouped Survival Analysis', fontsize=14, fontweight='bold')

# 4a Grouped bar — sex × class
sex_class = df.groupby(['pclass', 'sex'])['survived'].mean() * 100
sex_class = sex_class.reset_index()
x = np.arange(3)
width = 0.35
male_vals = sex_class[sex_class['sex'] == 'male']['survived'].values
female_vals = sex_class[sex_class['sex'] == 'female']['survived'].values
axes[0].bar(x - width/2, male_vals, width, label='Male', color=COLORS['male'], edgecolor='white')
axes[0].bar(x + width/2, female_vals, width, label='Female', color=COLORS['female'], edgecolor='white')
for i, (m, f) in enumerate(zip(male_vals, female_vals)):
    axes[0].text(i - width/2, m + 1, f'{m:.0f}%', ha='center', fontsize=9, fontweight='bold')
    axes[0].text(i + width/2, f + 1, f'{f:.0f}%', ha='center', fontsize=9, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
axes[0].set_ylabel('Survival Rate (%)')
axes[0].set_title('Survival Rate by Sex & Class')
axes[0].legend()
axes[0].set_ylim(0, 110)

# 4b Stacked bar — age group × survived
age_counts = df.groupby(['age_group', 'survived'], observed=False).size().unstack()
age_counts.columns = ['Did Not Survive', 'Survived']
age_counts.index = age_counts.index.astype(str)
age_counts.plot(kind='bar', stacked=True, ax=axes[1],
                color=[COLORS['not_survived'], COLORS['survived']],
                edgecolor='white', width=0.6)
axes[1].set_title('Passenger Count by Age Group\n(Stacked by Survival)')
axes[1].set_xlabel('Age Group')
axes[1].set_ylabel('Number of Passengers')
axes[1].legend(loc='upper right')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'fig4_grouped_analysis.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Saved: fig4_grouped_analysis.png")

# ============================================================
# STEP 5 — INSIGHT REPORT
# ============================================================
print("\n" + "=" * 60)
print("  STEP 5: INSIGHT REPORT")
print("=" * 60)

report = """
╔══════════════════════════════════════════════════════════╗
║         TITANIC EDA — KEY INSIGHTS REPORT               ║
╚══════════════════════════════════════════════════════════╝

Dataset: 891 passengers | 15 original features
Overall Survival Rate: 38.4%

• INSIGHT 1 — Gender was the strongest survival predictor.
  Female passengers had a 74.2% survival rate compared to only
  18.9% for males. This reflects the maritime rule of
  "women and children first" during evacuation.

• INSIGHT 2 — Socioeconomic class significantly impacted survival.
  1st class passengers survived at 62.9%, 2nd class at 47.3%,
  and 3rd class at just 24.2%. Wealthier passengers had cabins
  closer to the upper decks and easier lifeboat access.

• INSIGHT 3 — Children were prioritized in the evacuation.
  Passengers aged 0–12 had the highest survival rate (~59%),
  while Seniors (61+) had the lowest (~22%), likely due to
  physical limitations and lower priority in evacuation.

• INSIGHT 4 — The gender-class interaction was dramatic.
  1st class females survived at ~97%, while 3rd class males
  survived at only ~15% — a 82 percentage point gap, showing
  that both sex and class together determined fate.

• INSIGHT 5 — Missing data was a key data quality issue.
  ~20% of age values were missing (filled with median: 28.0),
  and the 'deck' column had 77% missing values and was dropped.
  Proper imputation was critical before any analysis.
"""
print(report)

print("=" * 60)
print("  ALL STEPS COMPLETE — 4 CHART FILES SAVED")
print("=" * 60)
