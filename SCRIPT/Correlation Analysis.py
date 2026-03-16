'''This script calculates Pearson and Spearman correlation coefficients
between GDP per capita and CO2 emissions per capita for selected
case-study countries, as outlined in the analysis plan.'''

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os

warnings.filterwarnings("ignore")

COUNTRIES = ["USA", "China", "India", "Germany", "Brazil","South Africa"]

# CO2 data ends at 2022, so we cap there.
YEAR_START = 1800
YEAR_END   = 2022

OUTPUT_DIR = "/Users/sihewang/PycharmProjects/ds4002-Project-2-CO2vsGDP-Study/OUTPUT"

co2_raw = pd.read_csv("/Users/sihewang/PycharmProjects/ds4002-Project-2-CO2vsGDP-Study/DATA/co2_pcap_cons.csv")
gdp_raw = pd.read_csv("/Users/sihewang/PycharmProjects/ds4002-Project-2-CO2vsGDP-Study/DATA/gdp_pcap.csv")

def melt_dataset(df, value_name, year_end):
    # Melt wide-format Gapminder data into long format
    year_cols = [c for c in df.columns if c.isdigit() and int(c) <= year_end]
    long = df.melt(id_vars=["geo", "name"], value_vars=year_cols, var_name="year", value_name=value_name)
    long["year"] = long["year"].astype(int)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    return long

co2 = melt_dataset(co2_raw, "co2_per_capita", YEAR_END)
gdp = melt_dataset(gdp_raw, "gdp_per_capita", YEAR_END)

merged = pd.merge(co2, gdp, on=["geo", "name", "year"], how="inner")
merged.dropna(subset=["co2_per_capita", "gdp_per_capita"], inplace=True)

# Filter to selected countries
country_mask = merged["name"].str.lower().isin([c.lower() for c in COUNTRIES])
df = merged[country_mask].copy().sort_values(["name", "year"]).reset_index(drop=True)

print(f"Countries found : {sorted(df['name'].unique())}")
print(f"Year range      : {df['year'].min()} – {df['year'].max()}")
print(f"Total rows      : {len(df)}\n")

# START ANALYSIS
results = []

for country in sorted(df["name"].unique()):
    sub = df[df["name"] == country]
    # Pearson
    pearson_r, pearson_p = stats.pearsonr(sub["gdp_per_capita"], sub["co2_per_capita"])
    # Spearman
    spearman_r, spearman_p = stats.spearmanr(sub["gdp_per_capita"], sub["co2_per_capita"])

    results.append({
        "Country": country,
        "N_years": len(sub),
        "Pearson_r": round(pearson_r, 4),
        "Pearson_p": pearson_p,
        "Spearman_rho": round(spearman_r, 4),
        "Spearman_p": spearman_p,
    })

results_df = pd.DataFrame(results)

# Visualization starts
sns.set_theme(style="whitegrid", font_scale=1.1)
palette = sns.color_palette("Set2", n_colors=len(COUNTRIES))

# Scatter + regression per country
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, country in enumerate(sorted(df["name"].unique())):
    ax = axes[i]
    sub = df[df["name"] == country]
    ax.scatter(sub["gdp_per_capita"], sub["co2_per_capita"],
               alpha=0.5, s=18, color=palette[i])

    # OLS fit line
    z = np.polyfit(sub["gdp_per_capita"], sub["co2_per_capita"], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(sub["gdp_per_capita"].min(), sub["gdp_per_capita"].max(), 200)
    ax.plot(x_line, p_line(x_line), color="red", linewidth=1.5, linestyle="--")

    r_val = results_df.loc[results_df["Country"] == country, "Pearson_r"].values[0]
    rho_val = results_df.loc[results_df["Country"] == country, "Spearman_rho"].values[0]
    ax.set_title(f"{country}\nr={r_val}, ρ={rho_val}", fontsize=12)
    ax.set_xlabel("GDP per capita (2021 int'l $)")
    ax.set_ylabel("CO₂ per capita (tonnes)")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("GDP per Capita vs CO₂ Emissions per Capita : Scatter + OLS Fit",
             fontsize=15, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "scatter_regression.png"), dpi=200, bbox_inches="tight")
plt.close()

# Correlation heatmap
corr_matrix = df.pivot_table(index="year", columns="name",
                              values="co2_per_capita").corr()
fig2, ax2 = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax2)
ax2.set_title("Cross-Country CO₂ per Capita Correlation Matrix")
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "co2_correlation_heatmap.png"), dpi=200, bbox_inches="tight")
plt.close()

# Time-series overlay
fig3, (ax_gdp, ax_co2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

for i, country in enumerate(sorted(df["name"].unique())):
    sub = df[df["name"] == country]
    ax_gdp.plot(sub["year"], sub["gdp_per_capita"], label=country, color=palette[i])
    ax_co2.plot(sub["year"], sub["co2_per_capita"], label=country, color=palette[i])

ax_gdp.set_ylabel("GDP per capita (2021 int'l $)")
ax_gdp.set_title("GDP per Capita Over Time")
ax_gdp.legend(loc="upper left")

ax_co2.set_ylabel("CO₂ per capita (tonnes)")
ax_co2.set_xlabel("Year")
ax_co2.set_title("CO₂ Emissions per Capita Over Time")
ax_co2.legend(loc="upper left")

fig3.suptitle("Historical Trends — Selected Case-Study Countries", fontsize=14, y=1.01)
plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "time_series_overlay.png"), dpi=200, bbox_inches="tight")
plt.close()

results_df["Pearson_p_fmt"]  = results_df["Pearson_p"].apply(lambda x: f"{x:.2e}")
results_df["Spearman_p_fmt"] = results_df["Spearman_p"].apply(lambda x: f"{x:.2e}")

summary = results_df[["Country", "N_years", "Pearson_r", "Pearson_p_fmt","Spearman_rho", "Spearman_p_fmt"]]
summary.to_csv(os.path.join(OUTPUT_DIR, "correlation_results.csv"), index=False)