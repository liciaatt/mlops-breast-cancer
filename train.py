from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Chargement du dataset
print("Chargement du dataset...")
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Données d'entraînement : {X_train.shape[0]} exemples")
print(f"Données de test : {X_test.shape[0]} exemples")

# 3. Entraînement du modèle
print("\nEntraînement du modèle...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Évaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy : {accuracy:.4f}")
print("\nRapport détaillé :")
print(classification_report(y_test, predictions,
      target_names=["malignant", "benign"]))

# 5. Sauvegarde du modèle
joblib.dump(model, "model.pkl")
print("Modèle sauvegardé dans model.pkl ✅")