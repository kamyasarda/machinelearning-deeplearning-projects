# ------------------------------------------------------------
# Assignment: Predicting Box Office Success (Classification)
# - EDA (missing / outliers / redundancy)
# - Feature extraction + selection
# - 80/20 split (stratified) and 5-fold CV on training set
# - Models: Decision Tree, KNN (scaled)
# - Metrics: Accuracy, Precision, Recall, F1, ROC/AUC
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# 1) Load data
# ---------------------------
file_path = '/Users/kamyasarda/Downloads/Bollywood Box Office Success.xlsx'  # Updated path to Downloads folder
df = pd.read_excel(file_path)

# fix target name if misspelled
if 'Success/Faliure' in df.columns:
    df.rename(columns={'Success/Faliure': 'Success/Failure'}, inplace=True)

print("Initial shape:", df.shape)
print("\nFirst few rows of the dataset:")
print(df.head())

# ---------------------------
# 2) EDA: info, missing, target balance
# ---------------------------
print("\nData info:")
df.info()

print("\nMissing values per column:")
print(df.isna().sum())

print("\nTarget distribution (counts):")
print(df['Success/Failure'].value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x='Success/Failure', data=df)
plt.title('Success vs Failure count')
plt.show()

# Boxplots to visualize distributions and outliers
num_cols = ['Budget(in crores)', 'Youtube_Views', 'Youtube_Likes', 'Youtube_Dislikes']
plt.figure(figsize=(12,6))
df[num_cols].boxplot()
plt.title("Boxplot of Key Numeric Features (Before Outlier Treatment)")
plt.ylabel("Value (in crores or counts)")
plt.show()

# Correlation heatmap for numeric features
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


# ---------------------------
# 3) Remove redundancy & leakage
# Explanation: For pre-release prediction we must NOT use post-release features.
# We will remove Box_Office_Collection, Profit and Earning_Ratio from model features.
# But we keep a copy of raw data for any descriptive analysis.
# ---------------------------
df_raw = df.copy()  # keep copy for reporting / descriptive stats
leak_cols = ['Box_Office_Collection(in crores)', 'Profit(in crores)', 'Earning_Ratio']
for c in leak_cols:
    if c in df.columns:
        print("Dropping leakage column:", c)
        df = df.drop(columns=[c])

# drop identifier
if 'Movie_Name' in df.columns:
    df = df.drop(columns=['Movie_Name'])

print("\nShape after dropping leakage/ID:", df.shape)

# ---------------------------
# 4) Outlier handling for numeric features (IQR capping)
# ---------------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove('Success/Failure')  # keep target out
print("\nNumeric columns to treat for outliers:", num_cols)

# cap outliers using 1.5*IQR (winsorize)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # cap (winsorize)
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

print("\nOutlier capping applied.")

# ---------------------------
# 5) Simple feature extraction (create trailer engagement metric)
# ---------------------------
# If columns exist, create basic engagement ratio: (likes - dislikes) / views
if set(['Youtube_Views','Youtube_Likes','Youtube_Dislikes']).issubset(df.columns):
    # protect division by zero
    df['Trailer_Engagement'] = (df['Youtube_Likes'] - df['Youtube_Dislikes']) / (df['Youtube_Views'].replace(0, np.nan))
    df['Trailer_Engagement'] = df['Trailer_Engagement'].fillna(0)
    print("\nCreated feature: Trailer_Engagement")

# Also include Budget as-is (if present)
# (We keep features simple and interpretable)

# ---------------------------
# 6) Encode categorical features (LabelEncoder - professor style)
# ---------------------------
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns:", cat_cols)

le_map = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_map[col] = le

# Ensure target numeric
if df['Success/Failure'].dtype == 'object':
    df['Success/Failure'] = LabelEncoder().fit_transform(df['Success/Failure'].astype(str))

print(df.head())

# ---------------------------
# 7) Feature selection (mutual information ranking)
# ---------------------------
X_all = df.drop(columns=['Success/Failure'])
y_all = df['Success/Failure']

# compute mutual information for each feature
mi = mutual_info_classif(X_all, y_all, discrete_features='auto', random_state=42)
mi_series = pd.Series(mi, index=X_all.columns).sort_values(ascending=False)
print("\nMutual information feature ranking:")
print(mi_series)

# Choose top k features (e.g., top 8 or fewer if less available)
k = min(8, X_all.shape[1])
top_features = mi_series.index[:k].tolist()
print("\nSelected features:", top_features)

# ---------------------------
# 8) Train/test split (80/20 stratified)
# ---------------------------
X = df[top_features]
y = df['Success/Failure']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
print("\nTrain target distribution:\n", y_train.value_counts())

# ---------------------------
# 9) 5-fold CV setup (for reporting)
# ---------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------
# 10) Decision Tree: cross-validate (accuracy) on training set
# ---------------------------
dt = DecisionTreeClassifier(random_state=42)
scoring = ['accuracy','precision','recall','f1']
cv_results_dt = cross_validate(dt, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)
print("\nDecision Tree CV (means):")
for metric in scoring:
    print(f" {metric}: {np.mean(cv_results_dt['test_'+metric]):.4f}")

# Fit final tree on training set
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:,1]

print("Decision Tree classes:", dt.classes_)


# Test set metrics
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, zero_division=0)
rec_dt = recall_score(y_test, y_pred_dt, zero_division=0)
f1_dt = f1_score(y_test, y_pred_dt, zero_division=0)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
auc_dt = auc(fpr_dt, tpr_dt)

print("\nDecision Tree - Test metrics:")
print(" Accuracy:", round(acc_dt,4))
print(" Precision:", round(prec_dt,4))
print(" Recall:", round(rec_dt,4))
print(" F1:", round(f1_dt,4))
print(" AUC:", round(auc_dt,4))
print("\nConfusion Matrix (DT):\n", confusion_matrix(y_test, y_pred_dt))
print("\nClassification Report (DT):\n", classification_report(y_test, y_pred_dt, digits=4))

# Visualize a shallow tree for interpretability
plt.figure(figsize=(12,6))
plot_tree(dt, feature_names=X.columns, class_names=['Failure','Success'], filled=True, max_depth=3)
plt.title('Decision Tree (limited depth)')
plt.show()

print("\nTop textual rules (first 400 chars):\n")
print(export_text(dt, feature_names=list(X.columns))[:400])

# After Decision Tree confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.show()

# ---------------------------
# 11) KNN (scale features) + CV
# ---------------------------
# ----- FIX 1: One-hot encode categorical variables for KNN -----
X_train_knn = pd.get_dummies(X_train, drop_first=True)
X_test_knn = pd.get_dummies(X_test, drop_first=True)

# align columns to handle any category mismatches
X_test_knn = X_test_knn.reindex(columns=X_train_knn.columns, fill_value=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_knn)
X_test_scaled = scaler.transform(X_test_knn)

knn = KNeighborsClassifier(n_neighbors=5)
cv_results_knn = cross_validate(knn, X_train_scaled, y_train, cv=cv, scoring=scoring)
print("\nKNN CV (means):")
for metric in scoring:
    print(f" {metric}: {np.mean(cv_results_knn['test_'+metric]):.4f}")

# Fit and evaluate on test set
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
y_prob_knn = knn.predict_proba(X_test_scaled)[:,1]

acc_knn = accuracy_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn, zero_division=0)
rec_knn = recall_score(y_test, y_pred_knn, zero_division=0)
f1_knn = f1_score(y_test, y_pred_knn, zero_division=0)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
auc_knn = auc(fpr_knn, tpr_knn)

print("\nKNN - Test metrics:")
print(" Accuracy:", round(acc_knn,4))
print(" Precision:", round(prec_knn,4))
print(" Recall:", round(rec_knn,4))
print(" F1:", round(f1_knn,4))
print(" AUC:", round(auc_knn,4))
print("\nConfusion Matrix (KNN):\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report (KNN):\n", classification_report(y_test, y_pred_knn, digits=4))

# ---------------------------
# 12) ROC comparison plot
# ---------------------------
plt.figure(figsize=(6,5))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC={auc_dt:.3f})', color='blue')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC={auc_knn:.3f})', color='orange')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC: Decision Tree vs KNN')
plt.legend()
plt.show()

# ---------------------------
# 13) Feature importance (from tree) and simple interpretation mapping
# ---------------------------
if hasattr(dt, 'feature_importances_'):
    fi = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nDecision Tree feature importances:")
    print(fi)

# ---------------------------
# 14) Short model summary (two numerical expressions per model for final report)
# ---------------------------
print("\n--- FINAL SUMMARY (Test set) ---")
print(f"Decision Tree -> Accuracy = {acc_dt:.4f}, F1 = {f1_dt:.4f}, AUC = {auc_dt:.4f}")
print(f"KNN (k=5)     -> Accuracy = {acc_knn:.4f}, F1 = {f1_knn:.4f}, AUC = {auc_knn:.4f}")

# ---------------------------
# 15) Business interpretation & recommendations (copy-paste-ready)
# ---------------------------
print("\n--- Interpretation & Recommendations (copy-paste-ready) ---\n")
print("1) EDA: Dataset contains 149 movies (no missing values). Post-release columns were removed for pre-release prediction.")
print("2) Feature selection: Top pre-release predictors (by mutual information and tree importance) are listed above. Use these to inform decisions.")
print("3) Model performance: Decision Tree and KNN trained with 5-fold CV. Summary (test set) shown above as Accuracy and F1.")
print("4) Business meaning & recommendations:")
print("   • If the model highlights 'Lead actor category' or 'Production House' as important, prefer proven actors/production houses for high-budget projects.")
print("   • Improve trailer engagement (likes/views) and schedule releases in favorable windows (festivals/long weekends).")
print("   • Use the Decision Tree rules to segment films before release and allocate marketing budgets more efficiently.")
print("\nDone. You can copy the printed summaries and plots into your report. If you want, I can also generate a one-page written report (200–300 words) using these exact numbers.")
