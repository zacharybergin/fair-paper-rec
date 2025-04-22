import pandas as pd
import matplotlib.pyplot as plt

# excel file path for both binary and continuous
# -------------
file_path = 'fairness_outputs.xlsx'
# file_path = 'fairness_outputs_cont.xlsx'
excel_data = pd.ExcelFile(file_path)

# get all the sheets in a dataframe
sheet_names = excel_data.sheet_names
data_frames = {sheet: excel_data.parse(sheet) for sheet in sheet_names}

paper_counts = [10, 25, 50, 150, 250, 350]

# sort the lambda_values
lambda_values = sorted(set(value for df in data_frames.values() for value in df['lambda_fairness'].unique() if pd.notnull(value)))

# initialize dictionaries to hold each variable needed for all the lambda's
utilities_by_lambda = {lambda_val: [] for lambda_val in lambda_values}
protected_macro_by_lambda = {lambda_val: [] for lambda_val in lambda_values}
protected_micro_by_lambda = {lambda_val: [] for lambda_val in lambda_values}
f_measures_by_lambda = {lambda_val: [] for lambda_val in lambda_values}

# f_measure function
def f_measure(protected: float, utility: float, beta: float = 1.0) -> float:
    if protected <= 0 or utility <= 0 or beta <= 0:
        return 0.0

    beta_squared = beta ** 2
    numerator = (1 + beta_squared) * protected * utility
    denominator = (beta_squared * protected) + utility

    return numerator / denominator if denominator != 0 else 0.0

# populate the dictionaries with appropiate values from the excel
for lambda_val in lambda_values:
    for df in data_frames.values():
        if lambda_val in df['lambda_fairness'].values:
            row = df[df['lambda_fairness'] == lambda_val]
            utilities_by_lambda[lambda_val].append(row['Utility'].values[0])
            protected_macro_by_lambda[lambda_val].append(row['Protected_Macro'].values[0])
            protected_micro_by_lambda[lambda_val].append(row['Protected_Micro'].values[0])
        else:
            # in case we have to
            utilities_by_lambda[lambda_val].append(None)
            protected_macro_by_lambda[lambda_val].append(None)
            protected_micro_by_lambda[lambda_val].append(None)

# find the f-measures for each lambda value
beta_value = 1.0  # change beta

for lambda_val in lambda_values:
    for utility, protected_macro, protected_micro in zip(
        utilities_by_lambda[lambda_val], 
        protected_macro_by_lambda[lambda_val], 
        protected_micro_by_lambda[lambda_val]
    ):
        # use macro for f-measure but are able to switch to micro if needed

        # if utility is not None and protected_micro is not None:
        #     f_measures_by_lambda[lambda_val].append(f_measure(protected_micro, utility, beta=beta_value))
        # else:
        #     f_measures_by_lambda[lambda_val].append(None)

        if utility is not None and protected_macro is not None:
            f_measures_by_lambda[lambda_val].append(f_measure(protected_macro, utility, beta=beta_value))
        else:
            f_measures_by_lambda[lambda_val].append(None)

# only use 0 (baseline) and 2.5 (because it was the best) (the case for logits loss fn as well)
show_only_0_and_2_5 = False

# plottttting
plt.figure(figsize=(12, 8))
for lambda_val, f_measures in f_measures_by_lambda.items():
    if show_only_0_and_2_5 and lambda_val not in [0, 2.5]:
        continue
    label = "Baseline" if lambda_val == 0 else f"lambda_fairness = {lambda_val}"
    plt.plot(paper_counts, f_measures, marker='o', linestyle='-', label=label)

plt.xlabel("Number of Papers")
plt.ylabel("F Measure")
if file_path == 'fairness_outputs_cont.xlsx':
    plt.title(f"F-Measure Comparison for Fair Selection (λ Tuning, β = {beta_value}, Continuous Race)")
else:
    plt.title(f"F-Measure Comparison for Fair Selection (λ Tuning, β = {beta_value}, Boolean Race)")
plt.legend()
plt.grid()
plt.show()
