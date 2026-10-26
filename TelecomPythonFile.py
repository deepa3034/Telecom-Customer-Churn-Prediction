import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data Loading
df=pd.read_csv(r'C:\Users\user\Downloads\archive (7)\WA_Fn-UseC_-Telco-Customer-Churn.csv')


# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display dataset dimensions
print("\nDataset Shape:")
print(df.shape)


# CODE 2: Dataset Structure and Data Quality

# 1. Display column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# 2. Display dataset information
print("\n--- Dataset Information ---")
df.info()

# 3. Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 4. Check duplicate records
print("\n--- Duplicate Records ---")
print(df.duplicated().sum())

# 5. Check data types
print("\n--- Data Types ---")
print(df.dtypes)

# 6. Check blank values in TotalCharges
print("\n--- Blank Values in TotalCharges ---")
print((df["TotalCharges"].str.strip() == "").sum())


# CODE 3: Missing Value Treatment

'''# Create a flag for originally missing TotalCharges
df["TotalCharges_missing"] = (
    df["TotalCharges"].astype(str).str.strip() == ""
).astype(int)

# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("\nMissing values before imputation:")
print(df["TotalCharges"].isnull().sum())

# Check tenure for records with missing TotalCharges
print("\nTenure of affected customers:")
print(df.loc[df["TotalCharges"].isnull(), "tenure"].value_counts())

# Fill missing TotalCharges with zero only when tenure is zero
condition = df["TotalCharges"].isnull() & (df["tenure"] == 0)

df.loc[condition, "TotalCharges"] = 0

# Verify remaining missing values
print("\nMissing values after imputation:")
print(df.isnull().sum())

# Check TotalCharges data type
print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)'''

# Save cleaned dataset
df.to_csv("telecom_churn_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully.")



# CODE 4: Correlation Analysis

# Ensure TotalCharges is numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Convert churn to numerical values
df["ChurnFlag"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Select numerical columns
numeric_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "ChurnFlag"
]

# Calculate Pearson correlation
correlation_matrix = df[numeric_columns].corr()

print("\n--- Correlation Matrix ---")
print(correlation_matrix.round(3))

print("\n--- Correlation with Churn ---")
print(
    correlation_matrix["ChurnFlag"]
    .drop("ChurnFlag")
    .sort_values()
    .round(3)
)

# Visualize correlation heatmap
plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Telecom Customer Churn - Correlation Heatmap")
plt.tight_layout()
plt.show()

# CODE 5: Detecting Churn Patterns and Risk Factors

# Create tenure groups
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months",
            "25-48 months", "49-72 months"]
)

# Function to calculate churn rate by customer group
def analyze_churn(column):
    result = df.groupby(
        column, observed=True
    )["ChurnFlag"].agg(
        Total_Customers="count",
        Churned_Customers="sum",
        Churn_Rate="mean"
    )

    result["Churn_Rate"] *= 100

    print("\n--- Churn Analysis:", column, "---")
    print(result.round(2).sort_values(
        "Churn_Rate", ascending=False
    ))

# Analyze important customer characteristics
analyze_churn("Contract")
analyze_churn("tenure_group")
analyze_churn("InternetService")
analyze_churn("PaymentMethod")
analyze_churn("TechSupport")
analyze_churn("OnlineSecurity")

# Compare numerical features by churn status
print("\n--- Average Values by Churn Status ---")

print(
    df.groupby("Churn")[
        ["tenure", "MonthlyCharges", "TotalCharges"]
    ].mean().round(2)
)



# CODE 6: Feature Selection for Machine Learning

# Columns we do not want to use as predictors
exclude_columns = [
    "customerID",
    "Churn",
    "ChurnFlag",
    "tenure_group",
    "gender",
    "SeniorCitizen"
]

# Select input features
X = df.drop(columns=exclude_columns, errors="ignore")

# Select target
y = df["ChurnFlag"]

# Check the selected data
print("\n--- MACHINE LEARNING FEATURE SELECTION ---")

print("\nNumber of customers:", len(X))
print("Number of features:", X.shape[1])

print("\nSelected features:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nFeature data types:")
print(X.dtypes)



# CODE 7: Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- TRAIN-TEST SPLIT ---")

print("Training customers:", X_train.shape[0])
print("Testing customers:", X_test.shape[0])

print("\nTraining churn distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting churn distribution:")
print(y_test.value_counts(normalize=True).round(3))


# CODE 8: Data Preprocessing

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Make sure TotalCharges is numeric
X_train = X_train.copy()
X_test = X_test.copy()

X_train["TotalCharges"] = pd.to_numeric(
    X_train["TotalCharges"], errors="coerce"
)

X_test["TotalCharges"] = pd.to_numeric(
    X_test["TotalCharges"], errors="coerce"
)

# Identify numerical and categorical columns
numerical_features = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

# Numerical preprocessing
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\n--- DATA PREPROCESSING ---")
print("\nNumerical features:", numerical_features)
print("\nCategorical features:", categorical_features)

print("\nPreprocessing pipeline created successfully!")


# CODE 9: Train and Compare Three ML Models

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Define the three models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Store results
model_results = []
trained_models = {}

# Train and evaluate each model
for name, model in models.items():

    # Combine preprocessing and ML model
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Train using training data only
    pipeline.fit(X_train, y_train)

    # Predict on test data
    y_pred = pipeline.predict(X_test)

    # Predict churn probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calculate performance metrics
    model_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

    # Save trained model
    trained_models[name] = pipeline

# Create comparison table
results_df = pd.DataFrame(model_results)

print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(results_df.round(3).to_string(index=False))

# Identify model with highest ROC-AUC
best_model_name = results_df.loc[
    results_df["ROC-AUC"].idxmax(), "Model"
]

print("\nHighest ROC-AUC model:", best_model_name)



# CODE 10: Logistic Regression Confusion Matrix

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Select our leading model
final_model = trained_models["Logistic Regression"]

# Predict on test customers
y_pred = final_model.predict(X_test)

# Print detailed classification report
print("\n--- LOGISTIC REGRESSION REPORT ---")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Stayed", "Churned"]
))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\n--- CONFUSION MATRIX ---")
print(cm)

# Display confusion matrix chart
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Churned"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Logistic Regression - Churn Prediction")
plt.tight_layout()
plt.show()



# CODE 11: Optimize Logistic Regression Threshold

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# Use cross-validation on training data only
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Obtain out-of-fold churn probabilities
cv_probabilities = cross_val_predict(
    final_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]

# Compare prediction thresholds
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

threshold_results = []

for threshold in thresholds:

    predictions = (
        cv_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_train, predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_train, predictions,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_train, predictions,
            zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

print("\n--- THRESHOLD COMPARISON ---")
print(threshold_df.round(3).to_string(index=False))

# Select threshold with highest F1 score
best_threshold = threshold_df.loc[
    threshold_df["F1 Score"].idxmax(),
    "Threshold"
]

print("\nSelected threshold:", best_threshold)


# CODE 12: Evaluate Optimized Churn Threshold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Use our already-trained Logistic Regression model
final_model = trained_models["Logistic Regression"]

# Get churn probabilities for test customers
test_probabilities = final_model.predict_proba(
    X_test
)[:, 1]

# Apply the threshold selected using training data
optimized_predictions = (
    test_probabilities >= best_threshold
).astype(int)

# Calculate test performance
print("\n--- OPTIMIZED MODEL TEST RESULTS ---")
print("Selected threshold:", best_threshold)

print(
    "Accuracy:",
    round(accuracy_score(y_test, optimized_predictions), 3)
)

print(
    "Precision:",
    round(precision_score(y_test, optimized_predictions), 3)
)

print(
    "Recall:",
    round(recall_score(y_test, optimized_predictions), 3)
)

print(
    "F1 Score:",
    round(f1_score(y_test, optimized_predictions), 3)
)

# Confusion matrix
optimized_cm = confusion_matrix(
    y_test,
    optimized_predictions
)

print("\n--- OPTIMIZED CONFUSION MATRIX ---")
print(optimized_cm)

# Display chart
display = ConfusionMatrixDisplay(
    confusion_matrix=optimized_cm,
    display_labels=["Stayed", "Churned"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Logistic Regression - Optimized Threshold")
plt.tight_layout()
plt.show()



# CODE 13: Feature Importance Analysis

# Select our trained Logistic Regression model
final_model = trained_models["Logistic Regression"]

# Extract preprocessing and classification steps
fitted_preprocessor = final_model.named_steps["preprocessor"]
logistic_model = final_model.named_steps["classifier"]

# Get feature names after preprocessing
feature_names = fitted_preprocessor.get_feature_names_out()

# Get Logistic Regression coefficients
coefficients = logistic_model.coef_[0]

# Create a DataFrame
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Calculate absolute coefficient values
feature_importance["Absolute_Importance"] = (
    feature_importance["Coefficient"].abs()
)

# Sort features by absolute coefficient
feature_importance = feature_importance.sort_values(
    by="Absolute_Importance",
    ascending=False
)

# Display top 15 features
print("\n--- TOP 15 IMPORTANT FEATURES ---")

print(
    feature_importance[
        ["Feature", "Coefficient"]
    ].head(15).round(3).to_string(index=False)
)

# Select top 15 features for visualization
top_features = feature_importance.head(15).copy()

# Sort for horizontal bar chart
top_features = top_features.sort_values("Coefficient")

# Create chart
plt.figure(figsize=(11, 7))

colors = [
    "tomato" if value > 0 else "steelblue"
    for value in top_features["Coefficient"]
]

plt.barh(
    top_features["Feature"],
    top_features["Coefficient"],
    color=colors
)

plt.axvline(0, color="black", linewidth=1)

plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Customer Features")

plt.title(
    "Top 15 Features Influencing Customer Churn Predictions"
)

plt.tight_layout()
plt.show()

# CODE 14: Individual Customer Churn Predictions

# Use our trained Logistic Regression model
final_model = trained_models["Logistic Regression"]

# Select five customers from the test dataset
sample_customers = X_test.head(5).copy()

# Predict churn probabilities
churn_probabilities = final_model.predict_proba(
    sample_customers
)[:, 1]

# Apply our selected 30% threshold
churn_predictions = (
    churn_probabilities >= best_threshold
).astype(int)

# Create prediction results
prediction_results = pd.DataFrame({
    "Tenure": sample_customers["tenure"].values,
    "Contract": sample_customers["Contract"].values,
    "Monthly Charges": sample_customers["MonthlyCharges"].values,
    "Churn Probability (%)": (
        churn_probabilities * 100
    ).round(2),
    "Predicted Churn": [
        "Yes" if pred == 1 else "No"
        for pred in churn_predictions
    ]
})

# Add risk classification
prediction_results["Risk Classification"] = [
    "At Risk" if pred == 1 else "Not Flagged"
    for pred in churn_predictions
]

print("\n--- INDIVIDUAL CUSTOMER CHURN PREDICTIONS ---")

print(prediction_results.to_string(index=False))


# CODE 15: Save the Trained Machine Learning Model

import joblib
from pathlib import Path

# Select our final Logistic Regression model
final_model = trained_models["Logistic Regression"]

# Get the folder containing our Python script
project_folder = Path(__file__).resolve().parent

# Save the trained model
model_path = project_folder / "telecom_churn_model.pkl"

joblib.dump(final_model, model_path)

# Save our selected prediction threshold
threshold_path = project_folder / "churn_threshold.pkl"

joblib.dump(best_threshold, threshold_path)

# Save the input feature names
features_path = project_folder / "model_features.pkl"

joblib.dump(X_train.columns.tolist(), features_path)

print("\n--- MODEL SAVED SUCCESSFULLY ---")

print("Model saved at:", model_path)
print("Threshold saved at:", threshold_path)
print("Features saved at:", features_path)





