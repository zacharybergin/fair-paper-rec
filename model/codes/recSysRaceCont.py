# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import helpersCont as hp

# Set random seeds for reproducibility
randomSeed = 42
torch.manual_seed(randomSeed)
np.random.seed(randomSeed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining the weights
weights = {
        'Graduate Student': 0.45,
        'Lecturer': 0.18,
        'Professor': 0.07,
        'Associate Professor': 0.12,
        'Researcher': 0.18
    }

# Loading the dataset needed
dataset = pd.read_csv('Datasets/added-set/final-asian_continuous.csv', index_col= 0)

overallPaperPool = pd.read_csv('Datasets/papers_final-asian-withRace.csv', index_col=0)
sigchiPaperPool = pd.read_csv('Datasets/sigchi_selected_papers.csv')
sigchiPaperPool.reset_index(drop=True, inplace=True)
sigcaiPaperTitle = sigchiPaperPool['Title'].to_list()
authorFeatures = pd.read_csv('Datasets/features_final-asian.csv')
authorFeatures['Authors'] = authorFeatures['Authors'].apply(hp.clean_author_name)

# Define the protected Group
PROTECTED_GROUP = 'race'
if PROTECTED_GROUP == 'race':
    protectedGroup = ['Race']
elif PROTECTED_GROUP == 'country':
    protectedGroup = ['Country']
elif PROTECTED_GROUP == 'both':
    protectedGroup = ['Country', 'Race']

# Prepare the data
train_loader, val_loader, input_features, target, protected_attribute = hp.prepare_data(dataset, protectedGroup, device)

# Model Definition
class FairMLPModel(nn.Module):
    def __init__(self, input_size):
        super(FairMLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x).squeeze()

# Initialize the model
input_size = input_features.shape[1]
model = FairMLPModel(input_size=input_size).to(device)

# Define optimizer and loss functions
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_prediction = nn.BCELoss()

# Hyperparameters
num_epochs = 50


def performanceMetrics(model):
    # 4. Obtain Predictions on the Full Dataset
    model.eval()
    with torch.no_grad():
        X_full_tensor = torch.tensor(input_features.values, dtype=torch.float32).to(device)
        y_full_pred = model(X_full_tensor).cpu().numpy()
        y_full_true = target.values

    # 5. Adjust Threshold to Select Exactly 351 Papers
    desired_acceptance_count = 351 
    y_full_binary = hp.adjust_threshold(y_full_pred, desired_acceptance_count)
    num_accepted = y_full_binary.sum()
    print(f"\nNumber of accepted papers: {num_accepted}")

    # 6. Evaluate Performance and Fairness Metrics on the Full Dataset

    # Compute performance metrics
    accuracy, precision, recall, f1 = hp.get_performance_metrics(y_full_true, y_full_binary)

    # Output results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    return 

results = []

overallMacros = hp.getAuthorDistributions(overallPaperPool, authorFeatures, PROTECTED_GROUP, 'macro')
overallMicros = hp.getAuthorDistributions(overallPaperPool, authorFeatures, PROTECTED_GROUP, 'micro')
overallUtility = hp.getOverallUtility(overallPaperPool, authorFeatures, weights)

sigchiMacros = hp.getAuthorDistributions(sigchiPaperPool, authorFeatures, PROTECTED_GROUP, 'macro')
sigchiMicros = hp.getAuthorDistributions(sigchiPaperPool, authorFeatures, PROTECTED_GROUP, 'micro')
sigchiUtility = hp.getOverallUtility(sigchiPaperPool, authorFeatures, weights)

# Add overall metrics to the results
results.append({
    'lambda_fairness': None,
    'Protected_Macro': overallMacros['protectedPercentage'],
    'NonProtected_Macro': overallMacros['nonProtectedPercentage'],
    'Total_Papers_Macro': overallMacros['totalPapers'],
    'Protected_Micro': overallMicros['protectedPercentage'],
    'NonProtected_Micro': overallMicros['nonProtectedPercentage'],
    'Total_Authors_Micro': overallMicros['totalAuthors'],
    'Utility': overallUtility,
    'Pool': 'Overall'
})

# Add sigchi metrics to the results
results.append({
    'lambda_fairness': None,
    'Protected_Macro': sigchiMacros['protectedPercentage'],
    'NonProtected_Macro': sigchiMacros['nonProtectedPercentage'],
    'Total_Papers_Macro': sigchiMacros['totalPapers'],
    'Protected_Micro': sigchiMicros['protectedPercentage'],
    'NonProtected_Micro': sigchiMicros['nonProtectedPercentage'],
    'Total_Authors_Micro': sigchiMicros['totalAuthors'],
    'Utility': sigchiUtility,
    'Pool': 'SIGCHI'
})



lambda_values= [0,1,2,2.5,3,5,10]
for lambda_fairness in lambda_values:
    print(f'Running the expirement for Protected Group: {PROTECTED_GROUP} with Lambda: {lambda_fairness}!')
    trained_model = hp.train_model(model, train_loader, val_loader, lambda_fairness, PROTECTED_GROUP)
    performanceMetrics(trained_model)
    selected_papers = hp.select_top_papers(model, input_features, device,dataset)
    selected_papers.to_csv(f'SelectedPapersListContinuous_{PROTECTED_GROUP}_{lambda_fairness}.csv')
    selectedMacros = hp.getAuthorDistributions(selected_papers, authorFeatures, PROTECTED_GROUP, 'macro')
    selectedMicros = hp.getAuthorDistributions(selected_papers, authorFeatures, PROTECTED_GROUP, 'micro')
    selectedUtility = hp.getOverallUtility(selected_papers, authorFeatures, weights)

    # Collect the data for this lambda value (only for selected papers)
    result = {
        'lambda_fairness': lambda_fairness,
        'Protected_Macro': selectedMacros['protectedPercentage'],
        'NonProtected_Macro': selectedMacros['nonProtectedPercentage'],
        'Total_Papers_Macro': selectedMacros['totalPapers'],
        'Protected_Micro': selectedMicros['protectedPercentage'],
        'NonProtected_Micro': selectedMicros['nonProtectedPercentage'],
        'Total_Authors_Micro': selectedMicros['totalAuthors'],
        'Utility': selectedUtility,
        'Pool': 'Selected'
    }

    # Add to the results list
    results.append(result)

# Convert the results list to a pandas DataFrame for display and analysis
df_results = pd.DataFrame(results)

# Display the dataframe
print(df_results)

# Optionally save to CSV
df_results.to_csv(f'fairness_experiment_results_cont_{PROTECTED_GROUP}_{randomSeed}_{lambda_fairness}.csv', index=False)


