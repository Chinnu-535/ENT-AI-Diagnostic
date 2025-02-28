import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load features and labels
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy")

# Load trained SVM model and scaler
svm_model = joblib.load("ent_disease_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Normalize test data
test_features = scaler.transform(test_features)

# Predict on test set
y_pred = svm_model.predict(test_features)

# Compute accuracy
accuracy = accuracy_score(test_labels, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:\n", classification_report(test_labels, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Visualizing Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=["Chronic Otitis", "Earwax Plug", "Myringosclerosis", "Normal"],
            yticklabels=["Chronic Otitis", "Earwax Plug", "Myringosclerosis", "Normal"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for ENT Disease Detection")
plt.show()

# Best Performing Class:
# Class 1 (Myringosclerosis) has the highest precision (0.97) and f1-score (0.89)

# Classes That Need Improvement:
# Class 0 (Earwax Plug) has the lowest precision (0.70), meaning it's sometimes misclassified.
# Class 2 (Normal Ear) has low recall (0.78), meaning some normal ears are misclassified as other diseases.