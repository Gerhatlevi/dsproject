import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def perform_cross_validation(X, y, model):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    recall = cross_val_score(pipe, X, y, cv=cv, scoring='recall')
    return accuracy.mean(), recall.mean()

def get_model_results(X, y, model_type='logreg'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    if model_type == 'logreg':
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train)
        X_test_final = scaler.transform(X_test)
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    else: 
        X_train_final, X_test_final = X_train, X_test
        model = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')

    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)

    importance = None
    if model_type == "logreg":
        importance = pd.Series(np.abs(model.coef_[0]), index=X.columns)
    else:
        importance = None

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Model_Obj": model,
        "Matrix": confusion_matrix(y_test, y_pred),
        "Importance": importance
    }

datasets = {
    "110 (3 biomarkers)": "./final_data/110_cleaned.csv",
    "Enriched (3 biomarkers)": "./final_data/enriched_data_cleaned.csv",
    "375 (74 biomarkers)": "./final_data/375_cleaned.csv"
}

results_list = []

for name, path in datasets.items():
    df = pd.read_csv(path)
    X = df.drop(columns=["outcome"])
    y = df["outcome"]

    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_cv_acc, lr_cv_rec = perform_cross_validation(X, y, lr_model)
    lr_res = get_model_results(X, y, "logreg")

    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
    dt_cv_acc, dt_cv_rec = perform_cross_validation(X, y, dt_model)
    dt_res = get_model_results(X, y, "tree")
    
    results_list.append({
        "Dataset": name,
        "LR_CV_Recall": lr_cv_rec,
        "DT_CV_Recall": dt_cv_rec,
        "LR_Test_Recall": lr_res["Recall"],
        "DT_Test_Recall": dt_res["Recall"],
        "LR_Matrix": lr_res["Matrix"],
        "DT_Matrix": dt_res["Matrix"],
        "375_Tree_Model": dt_res["Model_Obj"] if "375" in name else None,
        "Features": X.columns.tolist(),
        "Importance": lr_res["Importance"] if "375" in name else None
    })

res_df = pd.DataFrame(results_list)
ax = res_df.set_index('Dataset')[['LR_CV_Recall', 'DT_CV_Recall', 'LR_Test_Recall', 'DT_Test_Recall']].plot(
    kind='bar', figsize=(14, 7), colormap='Paired'
)
plt.title("Model Recall Comparison Across Datasets")
plt.ylabel("Recall")
plt.legend(["LogReg CV", "Tree CV", "LogReg Test", "Tree Test"], loc='lower right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, res in enumerate(results_list):
    disp = ConfusionMatrixDisplay(confusion_matrix=res["LR_Matrix"], 
                                  display_labels=['Discharged', 'Deceased'])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f"LogReg: {res['Dataset']}\n(Recall: {res['LR_Test_Recall']:.2%})")
plt.tight_layout()
plt.show()

tree_item = next(item for item in results_list if "375" in item["Dataset"])
plt.figure(figsize=(25, 12))
plot_tree(tree_item["375_Tree_Model"], 
          feature_names=tree_item["Features"], 
          class_names=['Discharged', 'Deceased'], 
          filled=True, rounded=True, fontsize=10)
plt.title("375 Biomarkers - Decision Tree Visualization")
plt.show()

for res in results_list:
    if "375" in res["Dataset"]:
        importance_series = res["Importance"].abs().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        importance_series.plot(kind='barh', color='teal')
        
        plt.title(f"Top 10 Predictors - {res['Dataset']}")
        plt.xlabel("Absolute Coefficient Value (Importance)")
        plt.ylabel("Biomarker Name")
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
