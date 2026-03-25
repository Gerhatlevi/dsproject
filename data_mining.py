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
import seaborn as sns

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
        model = DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced')

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

if __name__ == "__main__":

    datasets = {
        "3 biomarkers (basic)": "./final_data/110_cleaned.csv",
        "3 biomarkers (enriched)": "./final_data/enriched_data_cleaned.csv",
        "74 biomarkers": "./final_data/375_cleaned.csv"
    }

    results_list = []

    for name, path in datasets.items():
        df = pd.read_csv(path)
        X = df.drop(columns=["outcome"])
        y = df["outcome"]

        lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        lr_cv_acc, lr_cv_rec = perform_cross_validation(X, y, lr_model)
        lr_res = get_model_results(X, y, "logreg")

        dt_model = DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced')
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
            "74_Tree_Model": dt_res["Model_Obj"] if "74" in name else None,
            "Features": X.columns.tolist(),
            "Importance": lr_res["Importance"] if "74" in name else None
        })

    custom_colors = ['#5D6D7E', '#A93226', '#2C3E50', '#7B241C']
    res_df = pd.DataFrame(results_list)
    plot_df = res_df.set_index('Dataset')[['LR_CV_Recall', 'DT_CV_Recall', 'LR_Test_Recall', 'DT_Test_Recall']]
    ax = plot_df.plot(
        kind='bar', 
        figsize=(14, 7), 
        color=custom_colors,
        width=0.8, 
        edgecolor='white',
        linewidth=1
    )
    plt.title("Model Recall Comparison Across Datasets", fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.xticks(rotation=0, fontsize=11)
    plt.legend(
        ["LogReg (CV)", "Decision Tree (CV)", "LogReg (Test)", "Decision Tree (Test)"], 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False,
        fontsize=11
    )
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, res in enumerate(results_list):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=res["LR_Matrix"], 
            display_labels=['Discharged', 'Deceased']
        )
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
        axes[i].set_title(f"LogReg: {res['Dataset']}\n(Recall: {res['LR_Test_Recall']:.2%})",
                      fontsize=11, fontweight='bold', pad=15)
        if i > 0:
            axes[i].set_ylabel('')
            axes[i].set_xlabel('')
            axes[i].set_yticklabels([])
        else:
            axes[i].set_ylabel('True Label', fontsize=12, fontweight='600')
            axes[i].set_xlabel('Predicted Label', fontsize=12, fontweight='600')
    plt.subplots_adjust(wspace=0.1, top=0.85)
    plt.show()

    tree_item = next(item for item in results_list if "74" in item["Dataset"])
    clean_features = [f.replace('Lactate dehydrogenase', 'LDH')
                   .replace('Hypersensitive c-reactive protein', 'CRP')
                   .replace('Quantification of Treponema pallidum antibodies', 'T. Pallidum Ab')
                   .replace('Interleukin 2 receptor', 'IL-2R') 
                  for f in tree_item["Features"]]
    plt.figure(figsize=(24, 10), dpi=100)
    plot_tree(tree_item["74_Tree_Model"], 
          feature_names=clean_features, 
          class_names=['Discharged', 'Deceased'], 
          filled=True, 
          rounded=True, 
          fontsize=9,
          precision=2,
          impurity=False,
          proportion=False,
          node_ids=False, 
          label='none')
    plt.tight_layout()
    plt.show()

    res_74 = next((item for item in results_list if "74" in item["Dataset"]), None)
    if res_74 is not None and res_74["Importance"] is not None:
        importance_series = res_74["Importance"].sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(12, 8))
        sns.set_style("white")
        navy_blue = '#003366'
        ax = importance_series.plot(kind='barh', color=navy_blue, width=0.75)
        plt.title(f"Most Important Biomarkers", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Impact on Mortality Prediction", fontsize=12, labelpad=10)
        plt.ylabel("")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
