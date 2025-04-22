import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def prepare_data(dataset, protected_group_column, device, test_size=0.2, random_state=42):
    """
    Prepares the dataset by splitting into training and validation sets and converts them to tensors.
    
    Arguments:
    - dataset: The input dataset.
    - protected_group_column: The column indicating the protected group.
    
    Returns:
    - train_loader, val_loader: DataLoader for training and validation.
    - input_features, target, protected_attribute: Dataset components for further evaluation.
    """
    # Ensure protected_group_columns is a list
    if not isinstance(protected_group_column, list):
        raise ValueError("protected_group_columns should be a list of column names.")
    # Label mapping and rating logic
    label_mapping = {1: 0, 2: 1, 3: 2}
    dataset['Label'] = dataset['Label'].replace(label_mapping)
    dataset['rating'] = dataset['Label'].apply(lambda x: 1 if x == 2 else 0)

    # Split input features and target
    # edited from 'rating' to 'Label'
    input_features = dataset.drop(columns=protected_group_column + ['Label', 'Title', 'Authors'])
    input_features = input_features.astype(float)
    target = dataset['rating'].astype(float)
    protected_attribute = dataset[protected_group_column].astype(float)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val, protected_train, protected_val = train_test_split(
        input_features, target, protected_attribute, test_size=test_size, random_state=random_state
    )

    # Convert data to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32).to(device),
        torch.tensor(y_train.values, dtype=torch.float32).to(device),
        torch.tensor(protected_train.values, dtype=torch.float32).to(device)
    )
    
    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32).to(device),
        torch.tensor(y_val.values, dtype=torch.float32).to(device),
        torch.tensor(protected_val.values, dtype=torch.float32).to(device)
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    return train_loader, val_loader, input_features, target, protected_attribute


# def statistical_parity_loss(y_pred, protectedGroup):
#     """
#     Computes the weighted statistical parity loss between protected and unprotected groups.
    
#     Parameters:
#     y_pred: Predicted probabilities for paper acceptance.
#     race: Binary tensor indicating group membership (1 for Protected, 0 for Unprotected).
#     w_protected: Weight for the protected group.
#     w_unprotected: Weight for the unprotected group.
    
#     Returns:
#     Weighted statistical parity loss.
#     """
#     # Ensure y_pred and protectedGroup are of the same shape (after squeezing)
#     protectedGroup = protectedGroup.squeeze()  # Remove extra dimensions
#     assert y_pred.shape == protectedGroup.shape, "y_pred and protectedGroup must have the same shape"
    
#     protected_mask = (protectedGroup == 1)
#     unprotected_mask = (protectedGroup == 0)

#     protected_count = protected_mask.sum().item()
#     unprotected_count = unprotected_mask.sum().item()

#     selection_rate_protected = y_pred[protected_mask].mean() if protected_count > 0 else torch.tensor(0.0, device=y_pred.device)
#     selection_rate_unprotected = y_pred[unprotected_mask].mean() if unprotected_count > 0 else torch.tensor(0.0, device=y_pred.device)

#     squared_diff_protected = (selection_rate_protected - selection_rate_unprotected) ** 2
#     squared_diff_unprotected = (selection_rate_unprotected - selection_rate_protected) ** 2

#     weighted_squared_diff = squared_diff_protected + squared_diff_unprotected

#     return weighted_squared_diff

def statistical_parity_loss(y_pred, protectedGroup):
    """
    Computes a fairness loss for continuous protected attributes.
    Uses the squared Pearson correlation between predictions and the protected attribute.
    """
    protectedGroup = protectedGroup.squeeze()
    y_pred = y_pred.squeeze()

    # Normalize
    y_pred_centered = y_pred - y_pred.mean()
    protected_centered = protectedGroup - protectedGroup.mean()

    covariance = torch.sum(y_pred_centered * protected_centered)
    y_pred_var = torch.sum(y_pred_centered ** 2)
    protected_var = torch.sum(protected_centered ** 2)

    if y_pred_var.item() == 0 or protected_var.item() == 0:
        return torch.tensor(0.0, device=y_pred.device)

    correlation = covariance / (torch.sqrt(y_pred_var) * torch.sqrt(protected_var))
    
    return correlation ** 2  # squared correlation as loss



def train_model(model, train_loader, val_loader, lambda_fairness, PROTECTED_GROUP, num_epochs=50, patience=10):
    """
    Trains the model with fairness and prediction loss and performs early stopping based on validation performance.
    
    Arguments:
    - model: The neural network model.
    - train_loader, val_loader: DataLoader for training and validation.
    - lambda_fairness: Weight for the fairness loss.
    - num_epochs: Number of epochs.
    - patience: Patience for early stopping.
    
    Returns:
    - The trained model.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_prediction = nn.BCELoss()
    best_val_loss = np.inf
    counter = 0
    wCountry = 0.68; wRace = 0.32
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels, protecGroup in train_loader:
            optimizer.zero_grad()
            y_pred = model(inputs)
            prediction_loss = criterion_prediction(y_pred, labels)

            if PROTECTED_GROUP == 'both':
                country = protecGroup[:, 0]
                fairCountry = statistical_parity_loss(y_pred, country)

                race = protecGroup[:, 1]
                fairRace = statistical_parity_loss(y_pred, race)

                total_loss = prediction_loss + lambda_fairness * (wCountry * fairCountry + wRace * fairRace)

            else: 
                fairness_loss = statistical_parity_loss(y_pred, protecGroup)
                total_loss = (0.5) * prediction_loss + lambda_fairness * fairness_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        # Validation
        model.eval()
        val_loss = evaluate_model(model, val_loader, criterion_prediction, lambda_fairness, PROTECTED_GROUP)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f'best_model_{lambda_fairness}.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Load the best model
    model.load_state_dict(torch.load(f'best_model_{lambda_fairness}.pth'))

    # to get rid of future warning that is triggered when using 'weights_only=False', use this line instead
    # model.load_state_dict(torch.load(f'best_model_{lambda_fairness}.pth', weights_only=True))

    return model


def evaluate_model(model, val_loader, criterion_prediction, lambda_fairness, PROTECTED_GROUP):
    """
    Evaluates the model on the validation set.
    
    Arguments:
    - model: The neural network model.
    - val_loader: DataLoader for validation data.
    - criterion_prediction: Loss function for prediction.
    - lambda_fairness: Weight for fairness loss.
    
    Returns:
    - Validation loss.
    """
    wCountry = 0.62; wRace = 0.38
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels, protecGroup in val_loader:
            y_pred = model(inputs)
            prediction_loss = criterion_prediction(y_pred, labels)
            if PROTECTED_GROUP == 'both':
                country = protecGroup[:, 0]
                fairCountry = statistical_parity_loss(y_pred, country)

                race = protecGroup[:, 1]
                fairRace = statistical_parity_loss(y_pred, race)

                val_loss = prediction_loss + lambda_fairness * (wCountry * fairCountry + wRace * fairRace)
            else:
                fairness_loss = statistical_parity_loss(y_pred, protecGroup)
                val_loss += (0.5) *  prediction_loss + lambda_fairness * fairness_loss

    return val_loss / len(val_loader)

def get_performance_metrics(y_true, y_pred):
    # Helper to evaluate performance metrics (accuracy, precision, recall, F1-score)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

def adjust_threshold(predictions, desired_acceptance_count):
    """
    Adjusts the threshold to accept exactly the desired number of papers.
    
    Arguments:
    - predictions: The predicted probabilities of paper acceptance.
    - desired_acceptance_count: The number of papers to be accepted.
    
    Returns:
    - Binary decisions based on the adjusted threshold.
    """
    sorted_indices = np.argsort(-predictions)
    sorted_probs = predictions[sorted_indices]

    threshold = sorted_probs[desired_acceptance_count - 1] if desired_acceptance_count <= len(predictions) else 0.0
    return (predictions >= threshold).astype(int)

#desired_acceptance_count was 323
#needed to change to 10, 25, 50, 150, 250, 350
def select_top_papers(model, input_features, device, df, desired_acceptance_count=10):
    """
    Selects the top N papers based on predicted probabilities and returns the titles and authors.

    Parameters:
    - y_full_pred: Predicted probabilities of paper acceptance.
    - df: The original DataFrame containing paper metadata (e.g., Title, Authors).
    - desired_acceptance_count: Number of papers to select (default is 323).

    Returns:
    - selected_titles_authors: DataFrame with the titles and authors of the selected papers.
    """
    # Get the predictions from the model
    X_full_tensor = torch.tensor(input_features.values, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        y_full_pred = model(X_full_tensor).cpu().numpy().flatten()
    # Sort predicted probabilities in descending order and get the top indices
    sorted_indices = np.argsort(-y_full_pred)
    selected_indices = sorted_indices[:desired_acceptance_count]

    # Extract the corresponding rows (titles and authors) from the DataFrame
    selected_papers = df.iloc[selected_indices]
    selected_titles_authors = selected_papers[['Title', 'Authors', 'Label', 'Race', 'Country']]

    return selected_titles_authors

def get_author_info(author, authorFeatures):
    author = author.lstrip()
    author_info = authorFeatures[
        authorFeatures['Authors'].str.contains(author, case=False, na=False)
    ]
    
    if author_info.empty:
        raise RuntimeError(f'No Author Info Returned from the Authors Dataset for {author}')
    
    return {
        'Gender': author_info['Gender'].iloc[0],
        'Race': author_info['Race'].iloc[0],
        'Country': author_info['Country'].iloc[0],
        'h_index': author_info['h_index'].iloc[0],
        'Career': author_info['Career'].iloc[0],
        'Label': author_info['Label'].iloc[0]
    }

def getAuthorRaceList(authorList, authorFeatures):
    authorRace = []
    for author in authorList:
        author_info = get_author_info(author, authorFeatures)  # Reuses the get_author_info helper function
        authorRace.append(author_info['Race'])
    return authorRace

def getAuthorCountryList(authorList, authorFeatures):
    authorCountry = []
    for author in authorList:
        author_info = get_author_info(author, authorFeatures)  # Reuses the get_author_info helper function
        authorCountry.append(author_info['Country'])
    return authorCountry

def getAuthorGenderList(authorList, authorFeatures):
    authorGender = []
    for author in authorList:
        author_info = get_author_info(author, authorFeatures)  # Reuses the get_author_info helper function
        authorGender.append(author_info['Gender'])
    return authorGender

def getAveragehIndex(authorList, authorFeatures, weights):
    hIndexSum = 0
    author_count = 0

    for author in authorList:
        author_info = get_author_info(author, authorFeatures)

        h_index = author_info['h_index']
        career_stage = author_info['Career']
        weight = weights.get(career_stage, 0)  # Default to 0 if career stage not found

        hIndexSum += h_index * weight
        author_count += 1

    # Return the average h-index
    return hIndexSum / author_count if author_count > 0 else 0


def clean_title(name):
    # Remove any character that is not alphabetic or a space
    cleaned_title = re.sub(r'[^A-Za-z\s]', '', name)
    # Optionally, you can also strip any extra spaces
    cleaned_title = cleaned_title.strip()
    return cleaned_title

def clean_author_name(name):
    # Remove special characters except for spaces and alphabets
    cleaned_name = re.sub(r'[^a-zA-Z\s\']', '', name)
    # Remove extra spaces
    #cleaned_name = ' '.join(cleaned_name.split())
    return cleaned_name.strip()

# Function to clean and convert the 'Authors' field into a list
def parse_authors(authors):
    if isinstance(authors, list):
        return authors  # If it's already a list, return it

    if isinstance(authors, str):
        # Remove unwanted characters like square brackets, and split by comma
        authors_cleaned = re.sub(r"[\[\]\'\"]", "", authors)  # Remove brackets and quotes
        author_list = [clean_author_name(author.strip()) for author in authors_cleaned.split(',')]  # Split and clean names
        return author_list

    return []  # If it's neither a string nor a list, return an empty list (or handle it differently)


def getMacroCounts(authorList, protectedGroupType, authorFeatures):
    protectedCount = 0
    nonProtectedCount = 0
    if protectedGroupType == 'race':
        authorRaceList = getAuthorRaceList(authorList, authorFeatures)
        if any(race in ['B_NL', 'HL'] for race in authorRaceList):
            protectedCount = 1
        # Check if any non-protected race is in the authorRaceList
        if any(race in ['W_NL', 'A'] for race in authorRaceList):
            nonProtectedCount = 1
    
    if protectedGroupType == 'country':
        authorCountryList = getAuthorCountryList(authorList, authorFeatures)
        if 'developed' in authorCountryList:
            nonProtectedCount = 1
        if 'under-developed' in authorCountryList:
            protectedCount = 1
    return protectedCount, nonProtectedCount

def getMicroCounts(authorList, protectedGroupType, authorFeatures):
    protectedCount = 0
    nonProtectedCount = 0
    
    if protectedGroupType == 'race':
        authorRaceList = getAuthorRaceList(authorList, authorFeatures)
        
        for race in authorRaceList:
            if race in ['B_NL', 'HL']:  
                protectedCount += 1
            elif race in ['W_NL', 'A']:  
                nonProtectedCount += 1
    
    if protectedGroupType == 'country':
        authorCountryList = getAuthorCountryList(authorList, authorFeatures)
        
        for country in authorCountryList:
            if country == 'under-developed':  
                protectedCount += 1
            elif country == 'developed':  # Non-protected countries
                nonProtectedCount += 1

    return protectedCount, nonProtectedCount


def normalize_to_100(num1, num2):
    total = num1 + num2
    
    normalized_num1 = (num1 / total) * 100
    normalized_num2 = (num2 / total) * 100
    
    return normalized_num1, normalized_num2

def getOverallUtility(dataset, authorFeatures, weights):
    totalUtility = 0
    totalPapers = len(dataset)
    for idx, row in dataset.iterrows():
        authorsName = row['Authors']
        authorList = parse_authors(authorsName)
        avgAuthorHindex =  getAveragehIndex(authorList, authorFeatures, weights)
        totalUtility +=  avgAuthorHindex
    
    return round(totalUtility/totalPapers,2)

def getAuthorDistributions(dataset, authorFeatures, protectedGroupType, type):
    totalPapers = len(dataset)
    totalAuthors = 0
    totalProtectedCount = 0
    totalNonProtectedCount = 0

    for idx, row in dataset.iterrows():
        authorsName = row['Authors']
        paperTitle = row['Title']
        authorList = parse_authors(authorsName)
        if type == 'macro':
            protectedCount, nonProtectedCount = getMacroCounts(authorList, protectedGroupType, authorFeatures)
        elif type == 'micro':
            protectedCount, nonProtectedCount = getMicroCounts(authorList, protectedGroupType, authorFeatures)
            totalProtectedCount += protectedCount
            totalNonProtectedCount += nonProtectedCount
            totalAuthors += len(authorList)

        totalProtectedCount += protectedCount
        totalNonProtectedCount += nonProtectedCount

    protectedPercentage = (totalProtectedCount / totalPapers * 100) if totalPapers > 0 else 0
    nonProtectedPercentage = (totalNonProtectedCount / totalPapers * 100) if totalPapers > 0 else 0
    normProtected, normNonProtected = normalize_to_100(protectedPercentage, nonProtectedPercentage)
    return {
        'protectedPercentage': round(normProtected,2),
        'nonProtectedPercentage': round(normNonProtected,2),
        'totalPapers': totalPapers,
        'totalAuthors': totalAuthors
    }


