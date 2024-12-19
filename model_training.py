import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump

# Load and preprocess dataset
forest_fire_data = pd.read_csv('forest_fire_prevention_dataset.csv')
forest_fire_data = forest_fire_data.drop(columns=['Date'])
forest_fire_data['Fire Risk'] = forest_fire_data['Fire Risk'].apply(lambda x: 1 if x == 'High' else 0)
forest_fire_data_encoded = pd.get_dummies(forest_fire_data, columns=['Region', 'Vegetation Type'], drop_first=True)

X_ff = forest_fire_data_encoded.drop(columns=['Fire Risk'])
y_ff = forest_fire_data_encoded['Fire Risk']
X_ff_train, X_ff_test, y_ff_train, y_ff_test = train_test_split(X_ff, y_ff, test_size=0.2, random_state=42)

# Logistic Regression
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_ff_train, y_ff_train)
dump(log_reg_model, 'log_reg_model.joblib')

# Random Forest with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt']
}
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_ff_train, y_ff_train)
best_rf_model = grid_search.best_estimator_
dump(best_rf_model, 'best_rf_model.joblib')
