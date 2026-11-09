import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.sample(2000, random_state=1).to_csv("data/demo_no_drift.csv", index=False)

d = df.sample(2000, random_state=2).copy()
d["tenure"] = (d["tenure"] * 0.4).round().astype(int)
d["MonthlyCharges"] = (d["MonthlyCharges"] * 1.3).round(2)
idx = d.sample(frac=0.8, random_state=3).index
d.loc[idx, "Contract"] = "Month-to-month"
d.to_csv("data/demo_drift.csv", index=False)
print("done")