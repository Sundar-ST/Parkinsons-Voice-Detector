import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- Configuration ---
DATA_FILE = 'parkinsons.csv'  # Use the renamed file
MODEL_NAME = 'parkinsons_model.pkl'
SCALER_NAME = 'feature_scaler.pkl'

# --- 1. Load Data ---
try:
    # Use the correct separator based on the file content
    df = pd.read_csv(DATA_FILE, sep=',')
    print(f"Dataset loaded: {df.shape[0]} samples.")
except FileNotFoundError:
    print(f"ERROR: {DATA_FILE} not found. Please ensure the file is named exactly {DATA_FILE}")
    exit()

# --- 2. Define X and y ---
# Drop 'name' and 'status' (the target)
X = df.drop(['name', 'status'], axis=1)
y = df['status']

# --- 3. Scale Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- 5. Train Model (Random Forest) ---
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- 6. Evaluate and Save ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy*100:.2f}%")

joblib.dump(model, MODEL_NAME)
joblib.dump(scaler, SCALER_NAME)
print(f"\nModel and Scaler successfully saved: {MODEL_NAME}, {SCALER_NAME}")