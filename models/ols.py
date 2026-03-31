import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
df_new = pd.read_csv("/home/hav/scratch/can_ai_dd/df_new.csv")

### OLS

# Define target and feature
y = df_new['log_wages'].values.reshape(-1, 1)   # target
X = df_new[['female']].values                    # feature(s)

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Print coefficients
print("OLS Coefficient for female:", model.coef_[0][0])
