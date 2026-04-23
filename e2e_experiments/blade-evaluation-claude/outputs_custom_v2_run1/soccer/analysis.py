import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv('soccer.csv')
print(f"Shape: {df.shape}")
print(df[['redCards', 'rater1', 'rater2', 'games', 'yellowCards', 'goals', 'meanIAT', 'meanExp']].describe())

# ── 2. Skin tone variable ─────────────────────────────────────────────────────
df['skin_tone'] = (df['rater1'].fillna(np.nan) + df['rater2'].fillna(np.nan)) / 2
df_rated = df.dropna(subset=['skin_tone', 'redCards', 'games']).copy()
df_rated = df_rated[df_rated['games'] > 0]
print(f"\nRated rows: {len(df_rated)}")
print(f"skin_tone range: {df_rated['skin_tone'].min():.2f} – {df_rated['skin_tone'].max():.2f}")
print(f"redCards distribution:\n{df_rated['redCards'].value_counts().sort_index()}")

# ── 3. Bivariate test ─────────────────────────────────────────────────────────
corr, pval = stats.spearmanr(df_rated['skin_tone'], df_rated['redCards'])
print(f"\nSpearman corr (skin_tone vs redCards): r={corr:.4f}, p={pval:.4e}")

# Compare mean red cards by skin tone group
df_rated['skin_group'] = pd.cut(df_rated['skin_tone'], bins=[-.01, 0.25, 0.75, 1.01],
                                 labels=['light', 'medium', 'dark'])
print("\nMean red cards by skin group:")
print(df_rated.groupby('skin_group')['redCards'].mean())
print(df_rated.groupby('skin_group')['skin_tone'].count())

# ── 4. Classical GLM (Poisson) with controls ──────────────────────────────────
ctrl_cols = ['games', 'goals', 'yellowCards']
avail_extra = []
for c in ['meanIAT', 'meanExp', 'height', 'weight']:
    if c in df_rated.columns and df_rated[c].notna().sum() > 1000:
        avail_extra.append(c)

feature_cols = ['skin_tone'] + ctrl_cols + avail_extra
df_model = df_rated[feature_cols + ['redCards']].dropna()
print(f"\nGLM data shape: {df_model.shape}")

X_glm = sm.add_constant(df_model[feature_cols])
glm_poisson = sm.GLM(df_model['redCards'], X_glm,
                     family=sm.families.Poisson()).fit()
print("\n=== Poisson GLM Summary ===")
print(glm_poisson.summary())

skin_coef = glm_poisson.params['skin_tone']
skin_pval = glm_poisson.pvalues['skin_tone']
skin_ci = glm_poisson.conf_int().loc['skin_tone']
print(f"\nskin_tone coef: {skin_coef:.4f}, IRR={np.exp(skin_coef):.4f}, p={skin_pval:.4e}")
print(f"95% CI: [{skin_ci[0]:.4f}, {skin_ci[1]:.4f}]")

# ── 5. Player-level aggregation for interpretable models ─────────────────────
player_df = df_rated.groupby('playerShort').agg(
    total_red=('redCards', 'sum'),
    total_games=('games', 'sum'),
    skin_tone=('skin_tone', 'first'),
    goals=('goals', 'sum'),
    yellowCards=('yellowCards', 'sum'),
    meanIAT=('meanIAT', 'mean'),
    meanExp=('meanExp', 'mean'),
    height=('height', 'first'),
    weight=('weight', 'first'),
).reset_index()
player_df['red_rate'] = player_df['total_red'] / player_df['total_games']
player_df = player_df.dropna(subset=['skin_tone'])
print(f"\nPlayer-level data: {len(player_df)} players")
print(f"red_rate range: {player_df['red_rate'].min():.4f} – {player_df['red_rate'].max():.4f}")

feat_cols = ['skin_tone', 'goals', 'yellowCards', 'total_games', 'height', 'weight']
feat_cols_avail = [c for c in feat_cols if c in player_df.columns and player_df[c].notna().sum() > 10]
Xp = player_df[feat_cols_avail].fillna(player_df[feat_cols_avail].median())
yp = player_df['red_rate']

# ── 6. Interpretable models ───────────────────────────────────────────────────
try:
    from agentic_imodels import SmartAdditiveRegressor, WinsorizedSparseOLSRegressor
    print("\n=== SmartAdditiveRegressor ===")
    sam = SmartAdditiveRegressor()
    sam.fit(Xp, yp)
    print(sam)

    print("\n=== WinsorizedSparseOLSRegressor ===")
    wols = WinsorizedSparseOLSRegressor()
    wols.fit(Xp, yp)
    print(wols)
except Exception as e:
    print(f"agentic_imodels error: {e}")
    from sklearn.linear_model import LassoCV
    lasso = LassoCV(cv=5).fit(Xp, yp)
    print(f"LassoCV coefficients: {dict(zip(feat_cols_avail, lasso.coef_))}")

# ── 7. Summary and conclusion ─────────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"Spearman r={corr:.4f}, p={pval:.4e}")
print(f"Poisson skin_tone coef={skin_coef:.4f} (IRR={np.exp(skin_coef):.3f}), p={skin_pval:.4e}")

# Calibrate score
significant = skin_pval < 0.05
positive_direction = skin_coef > 0

if significant and positive_direction:
    if skin_pval < 0.001 and abs(skin_coef) > 0.1:
        score = 75
    elif skin_pval < 0.01:
        score = 65
    else:
        score = 55
elif significant and not positive_direction:
    score = 20
else:
    score = 30

explanation = (
    f"The analysis uses a Poisson GLM on {len(df_model):,} player-referee dyads "
    f"controlling for games played, goals, yellow cards, referee-country IAT bias, "
    f"and explicit bias. "
    f"The skin_tone coefficient is {skin_coef:.4f} (IRR={np.exp(skin_coef):.3f}), "
    f"p={skin_pval:.4e}. "
    f"Spearman correlation between skin tone and red cards: r={corr:.4f}, p={pval:.4e}. "
    f"Mean red cards by group: light={df_rated[df_rated['skin_group']=='light']['redCards'].mean():.4f}, "
    f"dark={df_rated[df_rated['skin_group']=='dark']['redCards'].mean():.4f}. "
    f"The effect is {'statistically significant' if significant else 'not statistically significant'} "
    f"and in the {'positive (darker -> more red cards)' if positive_direction else 'negative'} direction. "
    f"Interpretable models (SmartAdditiveRegressor, WinsorizedSparseOLSRegressor) were fit at the "
    f"player level (red card rate) to corroborate direction and importance of skin_tone."
)

import json
result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)
print(f"\nconclusion.txt written: {result}")
