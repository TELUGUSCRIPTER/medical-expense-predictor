"""
Generate realistic Indian medical insurance data and append to existing dataset.
Covers all major Indian regions with India-appropriate demographics.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

# Indian regions (replacing US regions for new data)
INDIAN_REGIONS = [
    'north',      # Delhi, UP, Haryana, Punjab, HP, J&K, Uttarakhand
    'south',      # Tamil Nadu, Kerala, Karnataka, AP, Telangana
    'east',       # West Bengal, Bihar, Odisha, Jharkhand, Assam
    'west',       # Maharashtra, Gujarat, Rajasthan, Goa
    'central',    # MP, Chhattisgarh
    'northeast',  # Already exists — will map to NE India states
]

NUM_NEW_RECORDS = 500

# Generate ages (Indian demographic: younger population)
ages = np.random.choice(
    range(18, 66),
    size=NUM_NEW_RECORDS,
    p=None  # uniform, will adjust with weights below
)
# Skew younger (India has younger population)
age_weights = np.array([max(0.5, 1.0 - (a - 18) * 0.012) for a in range(18, 66)])
age_weights /= age_weights.sum()
ages = np.random.choice(range(18, 66), size=NUM_NEW_RECORDS, p=age_weights)

# Gender (roughly equal)
sexes = np.random.choice(['male', 'female'], size=NUM_NEW_RECORDS, p=[0.52, 0.48])

# BMI (Indian avg ~23-24, but urban areas higher)
bmis = np.round(np.random.normal(loc=25.5, scale=5.5, size=NUM_NEW_RECORDS), 2)
bmis = np.clip(bmis, 15, 50)

# Children (Indian families tend to have more)
children = np.random.choice(range(0, 6), size=NUM_NEW_RECORDS, p=[0.20, 0.25, 0.30, 0.15, 0.07, 0.03])

# Smoker status (~15% smoking rate in India)
smokers = np.random.choice(['yes', 'no'], size=NUM_NEW_RECORDS, p=[0.15, 0.85])

# Regions (weighted by population)
regions = np.random.choice(
    INDIAN_REGIONS,
    size=NUM_NEW_RECORDS,
    p=[0.24, 0.22, 0.18, 0.20, 0.10, 0.06]  # Population-weighted
)

# Generate charges based on realistic Indian factors
# Base charge in USD (will be stored same as original dataset)
charges = []
for i in range(NUM_NEW_RECORDS):
    # Base cost (Indian healthcare is cheaper than US)
    base = 1200 + (ages[i] - 18) * 120  # Age factor

    # BMI factor
    if bmis[i] > 35:
        base *= 1.45
    elif bmis[i] > 30:
        base *= 1.25
    elif bmis[i] > 25:
        base *= 1.10

    # Smoker factor (massive impact)
    if smokers[i] == 'yes':
        base *= 3.5 + np.random.uniform(0, 1.0)

    # Children factor
    base += children[i] * 350

    # Gender factor (slight variation)
    if sexes[i] == 'female' and ages[i] > 35:
        base *= 1.05

    # Region cost factor (metro cities cost more)
    region_multiplier = {
        'north': 1.12,   # Delhi NCR drives costs up
        'south': 1.08,   # Good healthcare infrastructure
        'west': 1.15,    # Mumbai, expensive
        'east': 0.92,    # Lower costs
        'central': 0.88, # Rural, lowest costs
        'northeast': 0.85, # Remote areas
    }
    base *= region_multiplier.get(regions[i], 1.0)

    # Add noise
    base *= np.random.uniform(0.85, 1.15)

    charges.append(round(base, 3))

# Create DataFrame
new_data = pd.DataFrame({
    'age': ages,
    'sex': sexes,
    'bmi': bmis,
    'children': children,
    'smoker': smokers,
    'region': regions,
    'charges': charges
})

# Load existing data
existing = pd.read_csv('dataset/insurance.csv')
print(f"Existing records: {len(existing)}")
print(f"Existing regions: {existing['region'].unique()}")

# Combine
combined = pd.concat([existing, new_data], ignore_index=True)
print(f"\nNew records added: {len(new_data)}")
print(f"Total records: {len(combined)}")
print(f"\nAll regions: {combined['region'].unique()}")
print(f"\nRegion distribution:")
print(combined['region'].value_counts())
print(f"\nNew data stats:")
print(new_data.describe())
print(f"\nSmoker breakdown (new data):")
print(new_data['smoker'].value_counts())

# Save
combined.to_csv('dataset/insurance.csv', index=False)
print(f"\n✅ Dataset saved! Total: {len(combined)} records")
