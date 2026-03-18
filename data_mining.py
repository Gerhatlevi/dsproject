import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def logisctic_regression(file_path):
    df = pd.read_csv(file_path)

    X = df.drop(columns=['outcome'])
    y = df['outcome']

    model = LogisticRegression()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X)

    accuracies = cross_val_score(model, X_scaled_all, y, cv=cv, scoring='accuracy')
    recalls = cross_val_score(model, X_scaled_all, y, cv=cv, scoring='recall')

    print(f"\n=== Cross-Validation Results: {file_path} ===")
    print(f"Average Accuracy: {accuracies.mean():.2%} (+/- {accuracies.std():.2%})")
    print(f"Average Recall: {recalls.mean():.2%} (+/- {recalls.std():.2%})")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("--- Model's Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"Precision: {precision_score(y_test, y_pred):.2%}")
    print(f"Recall: {recall_score(y_test, y_pred):.2%}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nDetailed Report (Precision/Recall):")
    print(classification_report(y_test, y_pred))

#logisctic_regression("./final_data/110_cleaned.csv")
#logisctic_regression("./final_data/enriched_data_cleaned.csv")
#logisctic_regression("./final_data/375_cleaned.csv")