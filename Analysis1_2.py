import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "../ResearchData/Respiratory_Sample_data_randamized.csv"
TRAIN_YEARS = [2019, 2020, 2021, 2022]
TEST_YEAR = 2023

TOP_N_NATIONALITIES_MODEL = 25   # to keep GLM stable
TOP_N_NATIONALITIES_PLOT  = 15   # to keep plot readable

DAYS_IN_YEAR = 365
HORIZON_30D = 30
HORIZON_15D = 15

# ============================================================
# 1) LOAD + CLEAN
# ============================================================
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Target: treat_consluation is string-like; "-" means 0
    y = df["treat_consluation"].astype(str).str.strip().replace({"-": "0", "": "0"})
    df["y_annual"] = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    # Exposure: Earned days
    df["Earned_Days"] = pd.to_numeric(df["Earned_Days"], errors="coerce")
    df = df[df["Earned_Days"].notna() & (df["Earned_Days"] > 0)].copy()

    # Clean categorical fields
    df["AGEBAND"] = df["AGEBAND"].astype(str).str.strip()
    df = df[df["AGEBAND"].ne("Out of range")].copy()

    df["gender"] = df["gender"].astype(str).str.strip()
    df["nationality"] = df["nationality"].astype(str).str.strip()
    df["dependency"] = df["dependency"].astype(str).str.strip()

    return df[["treat_year", "AGEBAND", "gender", "nationality", "dependency",
               "y_annual", "Earned_Days"]].copy()

# ============================================================
# 2) BUCKETING (STABILITY)
# ============================================================
def add_buckets(df: pd.DataFrame, top_n_nat: int) -> pd.DataFrame:
    out = df.copy()

    # Nationality -> Top N + OTHER (based on total exposure)
    top_nat = (
        out.groupby("nationality")["Earned_Days"]
           .sum()
           .sort_values(ascending=False)
           .head(top_n_nat)
           .index
    )
    out["nationality_grp"] = np.where(out["nationality"].isin(top_nat), out["nationality"], "OTHER")

    # Dependency -> stable buckets
    dep = out["dependency"]
    out["dependency_grp"] = np.select(
        [dep.eq("MEMBER"), dep.isin(["SPOUSE", "CHILD"]), dep.eq("DEPENDENT")],
        ["MEMBER", "SPOUSE_CHILD", "DEPENDENT"],
        default="OTHER"
    )

    return out

# ============================================================
# 3) AGGREGATE TO RISK CELLS (FAST GLM)
# ============================================================
def aggregate_cells(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["treat_year", "AGEBAND", "gender", "nationality_grp", "dependency_grp"], as_index=False)
          .agg(
              y_annual=("y_annual", "sum"),
              Earned_Days=("Earned_Days", "sum")
          )
    )

# ============================================================
# 4) FIT POISSON GLM (BASELINE)
# ============================================================
def fit_poisson_glm(train_cells: pd.DataFrame):
    formula = "y_annual ~ C(AGEBAND) + C(gender) + C(nationality_grp) + C(dependency_grp)"
    model = smf.glm(
        formula=formula,
        data=train_cells,
        family=sm.families.Poisson(),
        offset=np.log(train_cells["Earned_Days"])
    ).fit(maxiter=100)
    return model

# ============================================================
# 5) PREDICT + SCALE TO 30D / 15D
# ============================================================
def add_predictions(df_cells: pd.DataFrame, model) -> pd.DataFrame:
    out = df_cells.copy()
    out["pred_annual"] = model.predict(out, offset=np.log(out["Earned_Days"]))
    out["pred_30d"] = out["pred_annual"] * (HORIZON_30D / DAYS_IN_YEAR)
    out["pred_15d"] = out["pred_annual"] * (HORIZON_15D / DAYS_IN_YEAR)
    return out

# ============================================================
# 6) ACTUAL VS PRED TABLES
# ============================================================
def actual_vs_pred_by(df_cells: pd.DataFrame, group_col: str) -> pd.DataFrame:
    t = (
        df_cells.groupby(group_col, as_index=False)
                .agg(
                    exposure_days=("Earned_Days", "sum"),
                    actual_annual=("y_annual", "sum"),
                    pred_annual=("pred_annual", "sum"),
                    pred_30d=("pred_30d", "sum"),
                    pred_15d=("pred_15d", "sum")
                )
    )
    t["pred_minus_actual"] = t["pred_annual"] - t["actual_annual"]
    t["pred_over_actual"] = t["pred_annual"] / np.maximum(t["actual_annual"], 1)
    return t

# ============================================================
# 7) PLOTTING HELPERS
# ============================================================
def plot_actual_vs_pred_bar(tbl: pd.DataFrame, x: str, title: str, top_n: int = None):
    t = tbl.copy()
    if top_n is not None:
        t = t.sort_values("actual_annual", ascending=False).head(top_n)
    else:
        t = t.sort_values(x)

    x_vals = t[x].astype(str).values
    idx = np.arange(len(t))
    w = 0.4

    plt.figure(figsize=(max(8, len(t)*0.45), 5))
    plt.bar(idx - w/2, t["actual_annual"].values, width=w, label="Actual (Annual)")
    plt.bar(idx + w/2, t["pred_annual"].values, width=w, label="Predicted (Annual)")
    plt.xticks(idx, x_vals, rotation=60, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_age_pred_horizons(age_tbl: pd.DataFrame):
    t = age_tbl.sort_values("AGEBAND").copy()
    x = np.arange(len(t))

    plt.figure(figsize=(max(8, len(t)*0.45), 5))
    plt.plot(x, t["pred_annual"].values, marker="o", label="Predicted Annual")
    plt.plot(x, t["pred_30d"].values, marker="o", label="Predicted 30-day (scaled)")
    plt.plot(x, t["pred_15d"].values, marker="o", label="Predicted 15-day (scaled)")
    plt.xticks(x, t["AGEBAND"].astype(str).values, rotation=60, ha="right")
    plt.ylabel("Expected count")
    plt.title("By Age Band: Predicted Annual vs 30-day vs 15-day (Scaled from Annual)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN RUN
# ============================================================
# Load + prep
df = load_and_clean(DATA_PATH)
df = add_buckets(df, top_n_nat=TOP_N_NATIONALITIES_MODEL)
cells = aggregate_cells(df)

# Train/Test split by year
train = cells[cells["treat_year"].isin(TRAIN_YEARS)].copy()
test  = cells[cells["treat_year"] == TEST_YEAR].copy()

# Fit GLM
glm = fit_poisson_glm(train)
print(glm.summary().tables[0])

# Predict on test year
test_pred = add_predictions(test, glm)

# Build intuitive output tables
age_tbl = actual_vs_pred_by(test_pred, "AGEBAND").sort_values("AGEBAND")
gender_tbl = actual_vs_pred_by(test_pred, "gender").sort_values("gender")
nat_tbl = actual_vs_pred_by(test_pred, "nationality_grp").sort_values("actual_annual", ascending=False)

print("\n--- 2023: Actual vs Predicted by Age Band (includes 30d/15d predictions) ---")
print(age_tbl.to_string(index=False))

print("\n--- 2023: Actual vs Predicted by Gender ---")
print(gender_tbl.to_string(index=False))

print(f"\n--- 2023: Actual vs Predicted by Nationality (Top {TOP_N_NATIONALITIES_PLOT} shown in plot) ---")
print(nat_tbl.head(25).to_string(index=False))

# Plots requested
plot_actual_vs_pred_bar(age_tbl, x="AGEBAND",
                        title="2023: Actual vs Predicted Annual Respiratory Incidence by Age Band")

plot_actual_vs_pred_bar(gender_tbl, x="gender",
                        title="2023: Actual vs Predicted Annual Respiratory Incidence by Gender")

plot_actual_vs_pred_bar(nat_tbl, x="nationality_grp",
                        title=f"2023: Actual vs Predicted Annual Respiratory Incidence by Nationality (Top {TOP_N_NATIONALITIES_PLOT})",
                        top_n=TOP_N_NATIONALITIES_PLOT)

# For each age: show annual vs 30-day vs 15-day predictions
plot_age_pred_horizons(age_tbl)
