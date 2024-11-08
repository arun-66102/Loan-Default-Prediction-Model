# DATA PRE-PROCESSING


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = 'D:\quiz_assignment\Loan_default.csv'  # Adjust path if needed
loan_data = pd.read_csv(file_path)

# Drop unique identifier
loan_data = loan_data.drop('LoanID', axis=1)

# Encode categorical features using one-hot encoding or label encoding
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus',
                    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

# Use one-hot encoding for categorical variables
loan_data = pd.get_dummies(loan_data, columns=categorical_cols, drop_first=True)

# Split features and target
X = loan_data.drop('Default', axis=1)
y = loan_data['Default']

# Address class imbalance in target variable using SMOTE
sm = SMOTE(random_state=42)

# Before applying SMOTE, handle NaN values in 'y'
# Remove rows with NaN in 'Default' column
loan_data = loan_data.dropna(subset=['Default'])

# Split features and target AFTER handling NaNs
X = loan_data.drop('Default', axis=1)
y = loan_data['Default']

X_res, y_res = sm.fit_resample(X, y)

# Standardize numerical features
numerical_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore',
                  'MonthsEmployed', 'NumCreditLines', 'InterestRate',
                  'LoanTerm', 'DTIRatio']

scaler = StandardScaler()
X_res[numerical_cols] = scaler.fit_transform(X_res[numerical_cols])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Check the processed data
print("Processed Data Shape:", X_train.shape, X_test.shape)
print("Class Distribution in Training Set:", y_train.value_counts())

"""# LOGISTIC REGRESSION"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Initialize and train the Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report for Logistic Regression:\n", classification_report(y_test, y_pred_lr))

"""# RANDOM FOREST CLASSIFIER"""

from sklearn.ensemble import RandomForestClassifier

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report for Random Forest:\n", classification_report(y_test, y_pred_rf))

"""# SUPPORT VECTOR MACHINE (SVM)"""

from sklearn.svm import SVC

# Initialize and train the SVM model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_svm = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report for SVM:\n", classification_report(y_test, y_pred_svm))

"""# DECISION TREE CLASSIFIER"""

from sklearn.tree import DecisionTreeClassifier

# Initialize and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report for Decision Tree:\n", classification_report(y_test, y_pred_dt))

"""# NEURAL NETWORK"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score

# Initialize the neural network model
nn_model = Sequential()

# Input layer (size of the input data)
nn_model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))

# Hidden layers
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dropout(0.5))  # Dropout to prevent overfitting

# Output layer
nn_model.add(Dense(1, activation='sigmoid'))  # Binary classification (loan default or not)

# Compile the model
nn_model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# Early stopping to prevent overfitting (stop training when the validation accuracy stops improving)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the neural network model
history = nn_model.fit(X_train, y_train,
                       epochs=50,
                       batch_size=32,
                       validation_data=(X_test, y_test),
                       callbacks=[early_stopping],
                       verbose=1)

# Predict and evaluate the model
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype('int32')  # Sigmoid output to binary

# Evaluate and print the performance
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
print("Classification Report for Neural Network:\n", classification_report(y_test, y_pred_nn))

"""# K-NEAREST NEIGHBORS (KNN)"""

from sklearn.neighbors import KNeighborsClassifier

# Initialize and train the KNN model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_knn = knn_model.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report for KNN:\n", classification_report(y_test, y_pred_knn))

"""# NAIVE BAYES CLASSIFIER"""

from sklearn.naive_bayes import GaussianNB

# Initialize and train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report for Naive Bayes:\n", classification_report(y_test, y_pred_nb))
