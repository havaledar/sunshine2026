import pandas as pd
import numpy as np
import statsmodels.api as sm  

# 1. Load Data
output_data = pd.read_pickle("/home/hav/scratch/sunshine/06_5_l_data_s.pkl")

data = output_data['data']
thresholds = output_data['thresholds']

cpi_lookup = thresholds.set_index('year')['Adjusted_Threshold']
gdp_lookup = thresholds.set_index('year')['GDPgrowth']

# 2. Filter for probability and create the dummy variable
print("Filtering for pgf probabilities...")
mask = (data['pgf'] >= 0.95) | (data['pgf'] <= 0.05)
data = data[mask].copy()

# Assign 1 if >= 0.95, else 0 (since we already filtered out everything else)
data['women_dummy'] = np.where(data['pgf'] >= 0.95, 1, 0)

# Pre-filter datasets for CPI and GDP based on the filtered baseline data
print("Filtering CPI and GDP datasets...")
data_cpi = data[data['comp'] >= data['year'].map(cpi_lookup)].copy()
data_gdp = data[data['comp'] >= data['year'].map(gdp_lookup)].copy()

# 3. Format data types for Statsmodels
print("Formatting data types for Statsmodels...")
for df in [data, data_cpi, data_gdp]:
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if dtype_str in ['Int8', 'Int16', 'Int32', 'Int64', 'Float32', 'Float64']:
            df[col] = df[col].astype('float64')
        elif dtype_str == 'string':
            df[col] = df[col].astype(str)

# 4. Define the 9 Model Specifications
# (Added missing tildes '~', commas, and swapped 'pgf' for 'women_dummy')
formulas = [
    'np.log(comp) ~ women_dummy',
    'np.log(comp) ~ women_dummy + C(sector)',
    'np.log(comp) ~ women_dummy + C(year)',
    'np.log(comp) ~ women_dummy + C(sector) + C(year)',
    'np.log(comp) ~ women_dummy + C(sector) + C(year) + tenure',
    'np.log(comp) ~ women_dummy + C(sector) + C(year) + tenure + inst_size',
    'np.log(comp) ~ women_dummy + C(sector) + C(year) + tenure + inst_size + C(emp_id)',
    'np.log(comp) ~ women_dummy + C(sector) + C(year) + tenure + inst_size + C(soc)',
    'np.log(comp) ~ women_dummy + C(sector) + C(year) + tenure + inst_size + C(emp_id) + C(soc)'
]

vars_to_keep = ['Intercept', 'women_dummy']

# 5. Data structures to hold results
results = {
    'Nominal': {'N': 0, 'models': []},
    'CPI':     {'N': 0, 'models': []},
    'GDP':     {'N': 0, 'models': []}
}

# 6. Loop through each formula and collect results
for i, formula in enumerate(formulas, start=1):
    print(f"Running Specification {i}/{len(formulas)}...")
    
    model_nom = sm.OLS.from_formula(formula, data=data).fit()
    model_cpi = sm.OLS.from_formula(formula, data=data_cpi).fit()
    model_gdp = sm.OLS.from_formula(formula, data=data_gdp).fit()
    
    # Capture Number of Observations from the first run
    if i == 1:
        results['Nominal']['N'] = int(model_nom.nobs)
        results['CPI']['N'] = int(model_cpi.nobs)
        results['GDP']['N'] = int(model_gdp.nobs)

    # Store Coefficients and Standard Errors
    results['Nominal']['models'].append({
        'Int_coef': model_nom.params['Intercept'], 'Int_se': model_nom.bse['Intercept'],
        'W_coef': model_nom.params['women_dummy'], 'W_se': model_nom.bse['women_dummy']
    })
    
    results['CPI']['models'].append({
        'Int_coef': model_cpi.params['Intercept'], 'Int_se': model_cpi.bse['Intercept'],
        'W_coef': model_cpi.params['women_dummy'], 'W_se': model_cpi.bse['women_dummy']
    })
    
    results['GDP']['models'].append({
        'Int_coef': model_gdp.params['Intercept'], 'Int_se': model_gdp.bse['Intercept'],
        'W_coef': model_gdp.params['women_dummy'], 'W_se': model_gdp.bse['women_dummy']
    })

# 7. Generate the custom LaTeX string
print("Generating LaTeX table...")
n_specs = len(formulas)
col_alignment = "l" + "r" * n_specs

# Define the custom stacked headers for all 9 specifications
custom_headers = [
    r"\textbf{Baseline}",
    r"\textbf{Sector}",
    r"\textbf{Year}",
    r"\begin{tabular}[t]{@{}l@{}}\textbf{Sector} \\ \textbf{Year}\end{tabular}",
    r"\begin{tabular}[t]{@{}l@{}}\textbf{Sector} \\ \textbf{Year} \\ \textbf{Tenure}\end{tabular}",
    r"\begin{tabular}[t]{@{}l@{}}\textbf{Sector} \\ \textbf{Year} \\ \textbf{Tenure} \\ \textbf{Inst. Size}\end{tabular}",
    r"\begin{tabular}[t]{@{}l@{}}\textbf{Sector} \\ \textbf{Year} \\ \textbf{Tenure} \\ \textbf{Inst. Size} \\ \textbf{Employer}\end{tabular}",
    r"\begin{tabular}[t]{@{}l@{}}\textbf{Sector} \\ \textbf{Year} \\ \textbf{Tenure} \\ \textbf{Inst. Size} \\ \textbf{Position}\end{tabular}",
    r"\begin{tabular}[t]{@{}l@{}}\textbf{Sector} \\ \textbf{Year} \\ \textbf{Tenure} \\ \textbf{Inst. Size} \\ \textbf{Employer} \\ \textbf{Position}\end{tabular}"
]

header_row = " & " + " & ".join(custom_headers) + " \\\\"

latex_str = f"""\\begin{{table}}[h!]
\\centering
\\caption{{OLS Regression analysis for estimating the magnitude of the gender wage gap.}}
\\vspace{{0.2cm}}
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{{col_alignment}}}
\\toprule
{header_row}
\\midrule
"""

for group, group_label in [('Nominal', 'Nominal Compensation'), ('CPI', 'CPI Adjusted'), ('GDP', 'GDP Adjusted')]:
    N_formatted = f"{results[group]['N']:,}"
    latex_str += f"\n% Group {group}\n"
    latex_str += f"\\textbf{{{group_label}}} & \\multicolumn{{{n_specs}}}{{r}}{{\\textit{{N}} = {N_formatted}}} \\\\\n"
    latex_str += "\\midrule\n"
    
    # Intercept row
    int_coefs = " & ".join([f"{m['Int_coef']:.3f}" for m in results[group]['models']])
    latex_str += f"Intercept & {int_coefs} \\\\\n"
    int_ses = " & ".join([f"({m['Int_se']:.5f})" for m in results[group]['models']])
    latex_str += f" & {int_ses} \\\\\n"
    
    # Women row
    w_coefs = " & ".join([f"{m['W_coef']:.3f}" for m in results[group]['models']])
    latex_str += f"Women & {w_coefs} \\\\\n"
    w_ses = " & ".join([f"({m['W_se']:.5f})" for m in results[group]['models']])
    latex_str += f" & {w_ses} \\\\\n"
    latex_str += "\\midrule\n"

# Notice \multicolumn{10} because we have 1 label column + 9 model columns
latex_str += """\\bottomrule
\\multicolumn{10}{l}{\\footnotesize \\textit{Notes:}} \\\\
\\multicolumn{10}{l}{\\footnotesize $\\bullet$ Standard errors in parentheses.} \\\\
\\multicolumn{10}{l}{\\footnotesize $\\bullet$ $N$ represents the total number of observations per category.} \\\\
\\multicolumn{10}{l}{\\footnotesize $\\bullet$ The dependent variable is the natural logarithm of total compensation (salary plus taxable benefits).} 
\\end{tabular}
}
\\end{table}
"""

# 8. Save the output
output_filepath = "/home/hav/scratch/sunshine/master_regression_table.tex"
with open(output_filepath, "w") as file:
    file.write(latex_str)

# 9. Generate and save the CSV backup
print("Generating CSV backup...")
csv_rows = []

# Loop through our results dictionary and flatten it into rows
for group in ['Nominal', 'CPI', 'GDP']:
    n_obs = results[group]['N']
    for i, model in enumerate(results[group]['models'], start=1):
        csv_rows.append({
            'Adjustment_Type': group,
            'Specification': i,
            'Intercept_Coef': model['Int_coef'],
            'Intercept_SE': model['Int_se'],
            'Women_Dummy_Coef': model['W_coef'],
            'Women_Dummy_SE': model['W_se'],
            'Observations_N': n_obs
        })

# Convert to DataFrame and export
csv_df = pd.DataFrame(csv_rows)
csv_filepath = "/home/hav/scratch/sunshine/master_regression_table.csv"
csv_df.to_csv(csv_filepath, index=False)

print(f"CSV backup successfully saved to: {csv_filepath}")
print(f"Job complete. Master table successfully saved to: {output_filepath}")