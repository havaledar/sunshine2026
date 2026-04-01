import pandas as pd
import numpy as np
import statsmodels.api as sm  

# 1. Load Data
output_data = pd.read_pickle("/home/hav/scratch/sunshine/06_5_l_data_s.pkl")

data = output_data['data']
thresholds = output_data['thresholds']

cpi_lookup = thresholds.set_index('year')['Adjusted_Threshold']
gdp_lookup = thresholds.set_index('year')['GDPgrowth']

# Pre-filter datasets (Added .copy() to ensure they are isolated dataframes)
print("Filtering datasets...")
data_cpi = data[data['comp'] >= data['year'].map(cpi_lookup)].copy()
data_gdp = data[data['comp'] >= data['year'].map(gdp_lookup)].copy()

# --- THE FIX ---
# Patsy/Statsmodels crashes on Pandas 'Int64' or 'Float64' extension types.
# This loop converts them back to standard numpy float64 types.
print("Formatting data types for Statsmodels...")
for df in [data, data_cpi, data_gdp]:
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        # Check if the dtype is a capitalized Pandas nullable type
        if dtype_str in ['Int8', 'Int16', 'Int32', 'Int64', 'Float32', 'Float64']:
            df[col] = df[col].astype('float64')
        elif dtype_str == 'string':
            df[col] = df[col].astype(str)
# ---------------

# 2. Define the 7 Model Specifications
# formulas = [
#     'np.log(comp) ~ pgf',
#     'np.log(comp) ~ pgf + C(sector)',
#     'np.log(comp) ~ pgf + C(sector) + C(year)',
#     'np.log(comp) ~ pgf + C(sector) + C(year) + tenure',
#     'np.log(comp) ~ pgf + C(sector) + C(year) + tenure + inst_size',
#     'np.log(comp) ~ pgf + C(sector) + C(year) + tenure + inst_size + C(emp_id)',
#     'np.log(comp) ~ pgf + C(sector) + C(year) + tenure + inst_size + C(emp_id) + C(soc)'
# ]

formulas = [
    'np.log(comp) ~ pgf + C(sector) + C(year) + tenure + inst_size + C(emp_id) + C(soc)'
]

# The names of the variables we actually want to report in the LaTeX table
vars_to_keep = ['Intercept', 'pgf']

# 3. Loop through each formula, run the regressions, and save the results
# for i, formula in enumerate(formulas, start=1):
#     print(f"Running Specification {i}/{len(formulas)}...")
for i, formula in enumerate(formulas, start=7): 
    print(f"Running Specification {i}...")
    
    # Run OLS Models
    model_nom = sm.OLS.from_formula(formula, data=data).fit()
    model_cpi = sm.OLS.from_formula(formula, data=data_cpi).fit()
    model_gdp = sm.OLS.from_formula(formula, data=data_gdp).fit()

    # Extract ONLY the target coefficients and standard errors
    data_md = {
        'Coef. (Nominal)': model_nom.params[vars_to_keep].round(3),
        'Std.Err. (Nominal)': model_nom.bse[vars_to_keep].round(5),
        'Coef. (CPI)': model_cpi.params[vars_to_keep].round(3),
        'Std.Err. (CPI)': model_cpi.bse[vars_to_keep].round(5),
        'Coef. (GDP)': model_gdp.params[vars_to_keep].round(3),
        'Std.Err. (GDP)': model_gdp.bse[vars_to_keep].round(5)
    }

    # Convert to DataFrame
    results_md = pd.DataFrame(data_md)

    # Rename the index
    results_md.index = ['Intercept', 'Women']

    # Sort alphabetically
    results_md = results_md.sort_index()

    # Export and Save LaTeX Output
    latex_table = results_md.to_latex(index=True, header=True)

    # Save to a distinct text file in your scratch directory
    output_filepath = f"/home/hav/scratch/sunshine/regression_results_spec_{i}.tex"
    with open(output_filepath, "w") as file:
        file.write(latex_table)
        
    print(f"Saved: {output_filepath}\n")

print("All 7 specifications completed successfully.")
