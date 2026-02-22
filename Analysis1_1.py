import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================
# 1) Load
# =========================
path = '../ResearchData/Respiratory_Sample_data_randamized.csv'
df = pd.read_csv(path)

# =========================
# 2) Clean target + exposure
# =========================
# Target (annual count): treat_consluation is stored as strings with blanks and "-"
y_raw = df["treat_consluation"].astype(str).str.strip().replace({"-": "0", "": "0"})
df["y_annual"] = pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int)

# Exposure (days)
df["Earned_Days"] = pd.to_numeric(df["Earned_Days"], errors="coerce")
df = df[df["Earned_Days"].notna() & (df["Earned_Days"] > 0)].copy()

# Keep only columns needed for this GLM
df = df[["treat_year", "AGEBAND", "gender", "nationality", "dependency",
         "y_annual", "Earned_Days"]].copy()

# Drop odd ageband
df["AGEBAND"] = df["AGEBAND"].astype(str).str.strip()
df = df[df["AGEBAND"].ne("Out of range")].copy()

# =========================
# 3) Reduce cardinality (stability)
# =========================
# Nationality: keep top-N by exposure, rest -> OTHER
top_n = 25
top_nat = (
    df.groupby("nationality")["Earned_Days"]
      .sum()
      .sort_values(ascending=False)
      .head(top_n)
      .index
)
df["nationality_grp"] = np.where(df["nationality"].isin(top_nat), df["nationality"], "OTHER")

# Dependency: collapse to stable buckets
dep = df["dependency"].astype(str).str.strip()
df["dependency_grp"] = np.select(
    [
        dep.eq("MEMBER"),
        dep.isin(["SPOUSE", "CHILD"]),
        dep.eq("DEPENDENT")
    ],
    ["MEMBER", "SPOUSE_CHILD", "DEPENDENT"],
    default="OTHER"
)

# =========================
# 4) Aggregate to risk cells
#    (Massively speeds up GLM and improves stability)
# =========================
g = (
    df.groupby(["treat_year", "AGEBAND", "gender", "nationality_grp", "dependency_grp"], as_index=False)
      .agg(
          y_annual=("y_annual", "sum"),
          Earned_Days=("Earned_Days", "sum")
      )
)

# =========================
# 5) Train/Test split by year (example): train 2019-2022, test 2023
# =========================
train = g[g["treat_year"] <= 2022].copy()
test  = g[g["treat_year"] == 2023].copy()

# =========================
# 6) Fit basic Poisson GLM with exposure offset
# =========================
formula = "y_annual ~ C(AGEBAND) + C(gender) + C(nationality_grp) + C(dependency_grp)"
glm_pois = smf.glm(
    formula=formula,
    data=train,
    family=sm.families.Poisson(),
    offset=np.log(train["Earned_Days"])
).fit(maxiter=100)

print(glm_pois.summary())

# =========================
# 7) Predict annual mean frequency for 2023 cells
# =========================
test["mu_annual_pred"] = glm_pois.predict(test, offset=np.log(test["Earned_Days"]))

# Quick evaluation (cell-level)
mae  = np.mean(np.abs(test["y_annual"] - test["mu_annual_pred"]))
mape = np.mean(np.abs(test["y_annual"] - test["mu_annual_pred"]) / np.maximum(test["y_annual"], 1))
print("\nHoldout (2023) cell-level:")
print("MAE :", mae)
print("MAPE:", mape)

# Portfolio-level check
print("\nHoldout (2023) total actual vs predicted:")
print("Actual:", test["y_annual"].sum())
print("Pred  :", test["mu_annual_pred"].sum())

# =========================
# 8) Convert annual prediction -> 30-day and 15-day approximations
#    Assumption: homogeneous Poisson rate within the year (no seasonality yet)
# =========================
test["mu_30d_pred"] = test["mu_annual_pred"] * (30 / 365)
test["mu_15d_pred"] = test["mu_annual_pred"] * (15 / 365)

# Show top predicted risk cells
out = (test.sort_values("mu_annual_pred", ascending=False)
          .head(15)[["treat_year", "AGEBAND", "gender", "nationality_grp", "dependency_grp",
                     "Earned_Days", "y_annual", "mu_annual_pred", "mu_30d_pred", "mu_15d_pred"]])
print("\nTop predicted cells (2023):")
print(out.to_string(index=False))

# =========================
# 9) Optional: get Incidence Rate Ratios (IRR) for interpretation
# =========================
coef = glm_pois.params
irr = np.exp(coef)
irr_table = (pd.DataFrame({"coef": coef, "IRR": irr, "p_value": glm_pois.pvalues})
               .sort_values("IRR", ascending=False))
print("\nTop IRR effects:")
print(irr_table.head(20).to_string())
