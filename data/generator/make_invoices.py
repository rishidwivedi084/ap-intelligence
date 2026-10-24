import numpy as np
import pandas as pd

np.random.seed(42)
N_INVOICES = 50_000

vendors = pd.read_csv("data/vendors.csv")

# ---------- Step 1: vendors, dates, amounts, due dates ----------

# Some vendors send many invoices, some only a few
weights = np.random.dirichlet(np.ones(len(vendors)))
vendor_idx = np.random.choice(len(vendors), size=N_INVOICES, p=weights)
inv = vendors.iloc[vendor_idx].reset_index(drop=True)   # one vendor row per invoice

# Random invoice dates across 3 years
start = pd.Timestamp("2023-11-01")
days = np.random.randint(0, 3 * 365, size=N_INVOICES)

invoices = pd.DataFrame({
    "vendor_id": inv["vendor_id"],
    "invoice_date": start + pd.to_timedelta(days, unit="D"),
})

# Amount = vendor's typical amount times a bit of random variation
noise = np.random.lognormal(mean=0, sigma=0.3, size=N_INVOICES)
invoices["amount"] = (inv["typical_amount"] * noise).round(2)

invoices["payment_terms_days"] = inv["payment_terms"]
invoices["due_date"] = invoices["invoice_date"] + pd.to_timedelta(
    inv["payment_terms"], unit="D"
)

invoices["department"] = np.random.choice(
    ["Operations", "IT", "Marketing", "HR", "Finance"],
    size=N_INVOICES,
    p=[0.35, 0.20, 0.20, 0.10, 0.15],
)

# Bigger amounts need more approvals (1, 2 or 3 levels)
q60 = invoices["amount"].quantile(0.60)
q90 = invoices["amount"].quantile(0.90)
invoices["approver_level"] = np.where(
    invoices["amount"] > q90, 3,
    np.where(invoices["amount"] > q60, 2, 1),
)

# ---------- Step 2: late probability and paid_late ----------

late_prob = inv["late_habit"].copy()        # start from the vendor's habit

q85 = invoices["amount"].quantile(0.85)
late_prob += np.where(invoices["amount"] > q85, 0.10, 0)                    # big amounts
late_prob += np.where(invoices["invoice_date"].dt.day >= 25, 0.08, 0)       # month-end rush
late_prob += 0.04 * (invoices["approver_level"] - 1)                        # more approvals
late_prob += np.where(invoices["payment_terms_days"] == 15, 0.05, 0)        # short terms

late_prob = late_prob.clip(0.02, 0.95)
paid_late = np.random.random(N_INVOICES) < late_prob
invoices["paid_late"] = paid_late.astype(int)

# ---------- Step 3: paid_date ----------

late_days = np.random.randint(1, 31, size=N_INVOICES)     # 1 to 30 days after due date
early_days = np.random.randint(0, 11, size=N_INVOICES)    # 0 to 10 days before due date
offset = np.where(paid_late, late_days, -early_days)
invoices["paid_date"] = invoices["due_date"] + pd.to_timedelta(offset, unit="D")

# ---------- Step 4: IDs and placeholder labels ----------

invoices = invoices.sort_values("invoice_date").reset_index(drop=True)   # oldest first
invoices.insert(0, "invoice_id", np.arange(1, N_INVOICES + 1))
invoices.insert(2, "invoice_number", "INV-" + (100000 + invoices["invoice_id"]).astype(str))
invoices["is_duplicate"] = 0     # we will inject real duplicates later
invoices["is_anomaly"] = 0       # we will inject real anomalies later

invoices.to_csv("data/invoices.csv", index=False)

# ---------- Quick sanity checks ----------
print(invoices.head())
print("Shape:", invoices.shape)
print("Overall late rate:", round(invoices["paid_late"].mean(), 3))
print("Late rate, big vs small amounts:")
print(invoices.groupby(invoices["amount"] > q85)["paid_late"].mean())