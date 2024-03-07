import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('feature_data.csv')

# Encode categorical feature X_0
le = LabelEncoder()
df['X_0'] = le.fit_transform(df['X_0'])

# Save the fitted LabelEncoder
joblib.dump(le, 'label_encoder.joblib')

# Features and target
X = df[['X_0', 'X_1', 'X_2']]
y = df['class']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model (example: RandomForest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'feature_model.joblib')
