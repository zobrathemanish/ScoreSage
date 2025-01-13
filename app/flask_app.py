from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Initialize Flask app
app = Flask(__name__)

# File path for pre-trained data
file_path = "C:\\Users\\harih\\OneDrive\\Desktop\\AI Final Project\\modelling_table (2).csv"
data = pd.read_csv(file_path)

# Drop unnecessary columns
drop_columns = [
    'Team 1',
    'Team 2',
    'Unnamed: 0',
    'H2H_Home_Total_Wins_Last_4',
    'H2H_Away_Total_Wins_Last_4',
    'H2H_Draws_Last_4'
]
data.drop(columns=drop_columns, inplace=True)

# Features and targets
X = data.drop(columns=["Team 1 Score", "Team 2 Score"])
y = np.where(data["Team 1 Score"] > data["Team 2 Score"], 0,   # Team 1 Win
             np.where(data["Team 1 Score"] == data["Team 2 Score"], 1,  # Draw
                      2))  # Team 2 Win

# Ordinal target labels for ordinal classification
def create_ordinal_labels(score):
    if score <= 0:
        return 0
    elif score <= 1:
        return 1
    elif score <= 2:
        return 2
    else:
        return 3

y_team1 = data["Team 1 Score"].apply(create_ordinal_labels)
y_team2 = data["Team 2 Score"].apply(create_ordinal_labels)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train stacking classifier
base_models = [
    ('win_model', LogisticRegression(penalty='l2', solver='liblinear', random_state=42)),
    ('draw_model', LogisticRegression(penalty='l2', solver='liblinear', random_state=42)),
    ('loss_model', LogisticRegression(penalty='l2', solver='liblinear', random_state=42))
]
meta_model = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
stacked_model.fit(X_scaled, y)

# Train ordinal classifiers
def train_ordinal_classifiers(X, y):
    thresholds = sorted(y.unique())[:-1]  # Exclude the highest class
    classifiers = []
    for threshold in thresholds:
        y_binary = (y > threshold).astype(int)
        classifier = LogisticRegression()
        classifier.fit(X, y_binary)
        classifiers.append(classifier)
    return classifiers

ordinal_classifiers_team1 = train_ordinal_classifiers(X_scaled, y_team1)
ordinal_classifiers_team2 = train_ordinal_classifiers(X_scaled, y_team2)

# Function to predict ordinal classes
def predict_ordinal_classes(classifiers, X):
    probabilities = [clf.predict_proba(X)[:, 1] for clf in classifiers]
    probabilities = np.column_stack(probabilities)
    predictions = (probabilities > 0.5).sum(axis=1)
    return predictions, probabilities

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Read uploaded file
    new_fixtures = pd.read_csv(file)
    nf_copy = new_fixtures.copy()

    # Retain only the features used during training
    try:
        new_fixtures = new_fixtures[X.columns]
    except KeyError as e:
        return f"Error: Missing or unexpected columns in the uploaded file: {e}"

    # Preprocess the data
    new_fixtures_imputed = pd.DataFrame(imputer.transform(new_fixtures), columns=new_fixtures.columns)
    new_fixtures_scaled = scaler.transform(new_fixtures_imputed)

    # Make predictions using stacked model
    predictions = stacked_model.predict(new_fixtures_scaled)
    probabilities = stacked_model.predict_proba(new_fixtures_scaled)

    # Map predictions to results
    prediction_labels = {0: 'Team 1 Wins', 1: 'Draw', 2: 'Team 2 Wins'}
    predictions_results = [prediction_labels[pred] for pred in predictions]

    # Generate ordinal predictions
    team1_ordinal_predictions, team1_ordinal_probs = predict_ordinal_classes(ordinal_classifiers_team1, new_fixtures_scaled)
    team2_ordinal_predictions, team2_ordinal_probs = predict_ordinal_classes(ordinal_classifiers_team2, new_fixtures_scaled)

    # Add predictions and probabilities to the original table
    nf_copy['Predictions'] = predictions_results
    nf_copy['Win Probability'] = probabilities[:, 0]
    nf_copy['Draw Probability'] = probabilities[:, 1]
    nf_copy['Loss Probability'] = probabilities[:, 2]
    nf_copy['Team_1_Ordinal_Prediction'] = team1_ordinal_predictions
    nf_copy['Team_2_Ordinal_Prediction'] = team2_ordinal_predictions

    # Add highest confidence score for ordinal predictions
    nf_copy['Team_1_Confidence'] = np.max(team1_ordinal_probs, axis=1)
    nf_copy['Team_2_Confidence'] = np.max(team2_ordinal_probs, axis=1)

    # Filter for the required columns
    selected_columns = [
        "Team 1", 
        "Team 2", 
        "Predictions", 
        "Win Probability", 
        "Draw Probability", 
        "Loss Probability", 
        "Team_1_Ordinal_Prediction", 
        "Team_2_Ordinal_Prediction", 
        "Team_1_Confidence", 
        "Team_2_Confidence"
    ]
    nf_filtered = nf_copy[selected_columns]

    # Render the results in a table
    return render_template(
        'results.html',
        tables=[nf_filtered.to_html(classes='data', index=False)],
        titles=nf_filtered.columns.values,
        zip=zip  # Pass the `zip` function explicitly
    )

if __name__ == '__main__':
    app.run(debug=True)
