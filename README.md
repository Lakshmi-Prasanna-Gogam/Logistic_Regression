# Logistic_Regression

This project implements a binary classification model using Logistic Regression to predict whether a tumor is benign (0) or malignant (1) based on labeled features from a breast cancer dataset.

Regression is a supervised machine learning technique used for predicting continuous or categorical outcomes. Logistic Regression, a special case of regression, is used here for binary classification tasks.

# Objective
To build a binary classifier using Logistic Regression that can accurately classify tumors as malignant or benign.

# Tools & Libraries Used
Python

Pandas – for data manipulation

NumPy – for numerical operations

Matplotlib & Seaborn – for data visualization

Scikit-learn – for machine learning modeling and evaluation


# Data Preprocessing

Handle missing values:

Used fillna(0) to replace missing values.

Drop unnecessary column:

Unnamed: 32 was dropped due to being irrelevant/noisy.

Convert categorical to numerical:

diagnosis column was label-encoded using LabelEncoder.

Feature-target split:

X: All columns except diagnosis

y: diagnosis column

# Exploratory Data Analysis (EDA)
Summary statistics (dataset.describe()) revealed no major outliers.

Checked class distribution and correlations.

Visualizations (not shown in code snippet) are recommended for better insights.

# Model Building
# Step 1: Train-test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 2: Model Training

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train, y_train.values.ravel())

# Model Evaluation
Predictions

predictions = model.predict(x_test)

# Metrics


Precision	

Recall	

ROC-AUC	

Confusion Matrix

# Model Parameters
# Coefficients:

Vector of weights for each feature used in prediction.

# Intercept:

Bias term used in the decision function.

# Sigmoid Function (Logistic Function) and Threshold
