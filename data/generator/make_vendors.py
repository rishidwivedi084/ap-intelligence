import numpy as np
import pandas as pd
from faker import Faker

N_VENDORS = 200
np.random.seed(42)   # same random numbers every run
fake = Faker()
Faker.seed(42)

vendors = []
# Step 2: IDs and names
vendor_ids = [f"V{i:03d}" for i in range(1, N_VENDORS + 1)]
names = [fake.company() for _ in range(N_VENDORS)]

# Step 3: the three random columns
typical_amount = np.random.lognormal(10, 0.5, size=N_VENDORS)
payment_terms = np.random.choice([15, 30, 45, 60], size=N_VENDORS)
late_habit = np.random.beta(2, 5, size=N_VENDORS)

# Step 4: the table
vendors = pd.DataFrame({
    "vendor_id": vendor_ids,
    "name": names,
    "typical_amount": typical_amount,
    "payment_terms": payment_terms,
    "late_habit": late_habit
})

# Step 5: save
vendors.to_csv("data/vendors.csv", index=False)
print(vendors.head())
