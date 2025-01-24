# Building a Machine Learning Pipeline
# Description: Creating a machine learning pipeline that handles data preprocessing, feature selection, model training, hyperparameter tuning, and evaluation.
# Key Libraries: scikit-learn, numpy, pandas, joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Sample Data
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Hyperparameter Tuning
params = {'model__n_estimators': [10, 50, 100], 'model__max_depth': [2, 5, 10]}
grid = GridSearchCV(pipeline, param_grid=params, cv=5)
grid.fit(X_train, y_train)

# Evaluation
y_pred = grid.best_estimator_.predict(X_test)
print(f"Best Parameters: {grid.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
