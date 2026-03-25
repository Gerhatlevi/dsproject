

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

from data_mining import get_model_results, perform_cross_validation


df_375 = pd.read_csv("./final_data/375_cleaned.csv")

top_3_features = ["PH value", "glucose", "aspartate aminotransferase"]
X_top3 = df_375[top_3_features]
y_top3 = df_375["outcome"]

model_top3 = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
cv_acc_top3, cv_rec_top3 = perform_cross_validation(X_top3, y_top3, model_top3)

res_top3 = get_model_results(X_top3, y_top3, "logreg")

fig, ax = plt.subplots(figsize=(10, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=res_top3["Matrix"], 
                              display_labels=['Discharged', 'Deceased'])
disp.plot(cmap='Blues', ax=ax, colorbar=False)
stats_text = (
    f"  Cross-Validation  \n"
    f"────────────────────\n"
    f"Accuracy: {cv_acc_top3:>8.2%}\n"
    f"Recall:   {cv_rec_top3:>8.2%}\n"
    f"\n"
    f"      Test Set      \n"
    f"────────────────────\n"
    f"Recall:   {res_top3['Recall']:>8.2%}"
)

plt.text(1.3, 0.5, stats_text, 
         transform=ax.transAxes,
         fontsize=11, 
         family='monospace',
         verticalalignment='center', 
         bbox=dict(boxstyle='round,pad=1', 
                   facecolor='#f9f9f9', 
                   edgecolor='#cccccc', 
                   alpha=1))
plt.title("Statistical Top 3 Predictors (PH, Glucose, AST)\nPerformance Summary", 
          fontsize=14, fontweight='bold', pad=20)

plt.ylabel('True Label', fontsize=12, fontweight='600')
plt.xlabel('Predicted Label', fontsize=12, fontweight='600')
plt.subplots_adjust(right=0.7) 

plt.show()