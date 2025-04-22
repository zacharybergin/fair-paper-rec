import pandas as pd
import matplotlib.pyplot as plt

# excel file path
file_path = 'fairness_outputs.xlsx'
# file_path = 'fairness_outputs_cont.xlsx'
excel_data = pd.ExcelFile(file_path)

# get all the sheets in a dataframe
sheet_names = excel_data.sheet_names
data_frames = {sheet: excel_data.parse(sheet) for sheet in sheet_names}

paper_counts = [10, 25, 50, 150, 250, 350]

# sort the lambda_values
lambda_values = sorted(set(value for df in data_frames.values() for value in df['lambda_fairness'].unique() if pd.notnull(value)))

utilities_by_lambda = {lambda_val: [] for lambda_val in lambda_values}

# get the utility from each lambda value that is found across multiple sheets
for lambda_val in lambda_values:
    for df in data_frames.values():
        if lambda_val in df['lambda_fairness'].values:
            utilities_by_lambda[lambda_val].append(df[df['lambda_fairness'] == lambda_val]['Utility'].values[0])
        else:
            utilities_by_lambda[lambda_val].append(None)

# option to display only lambda_fairness = 0 and lambda_fairness = 2.5
show_only_0_and_2_5 = False

# plottttting
plt.figure(figsize=(12, 8))
for lambda_val, utilities in utilities_by_lambda.items():
    if show_only_0_and_2_5 and lambda_val not in [0, 2.5]:
        continue
    label = "Baseline" if lambda_val == 0 else f"lambda_fairness = {lambda_val}"
    plt.plot(paper_counts, utilities, marker='o', linestyle='-', label=label)
    print(label, utilities)

plt.xlabel("Number of Papers")
plt.ylabel("Utility")
if file_path == "fairness_outputs_cont.xlsx":
    plt.title("Utility vs. Number of Papers (Continuous)")
else:
    plt.title("Utility vs. Number of Papers (Boolean)")
plt.legend()
plt.grid()
plt.show()
