
import os
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Define the base path for file operations
base_path = r'/data/xiaoyingzhen/PPHSVM'

# Load the data
train_data = pd.read_csv(os.path.join(base_path, 'yxtrainPPH.csv'))
test_data = pd.read_csv(os.path.join(base_path, 'yxvalPPH.csv'))

# Extract features and labels
train_x = train_data.iloc[:, 39:]
train_y = train_data.iloc[:, 1]
test_x = test_data.iloc[:, 39:]
test_y = test_data.iloc[:, 1]

# Standardize the features
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Compute correlation of each feature with the target
correlations = pd.DataFrame(train_x).apply(lambda x: x.corr(train_y))

# Select top features based on absolute correlation
top_features_index = correlations.abs().nlargest(420).index  # Select top 420 features
train_x = train_x[:, top_features_index]
test_x = test_x[:, top_features_index]

# Train Lasso model for feature selection
alpha_value = 0.17
lasso = Lasso(alpha=alpha_value, max_iter=5000, tol=5e-4)  
lasso.fit(train_x, train_y)

# Select features with non-zero coefficients
selected_features_index = [i for i, coef in enumerate(lasso.coef_) if coef != 0]
train_x_selected = train_x[:, selected_features_index]
test_x_selected = test_x[:, selected_features_index]

# Extract non-zero coefficients info after retraining
non_zero_feature_names = [str(x) for x in selected_features_index]
non_zero_feature_coef = lasso.coef_[lasso.coef_ != 0]

# Create a DataFrame of selected features and their coefficients
selected_features_df = pd.DataFrame({
    'feature_name': non_zero_feature_names,
    'coefficient': non_zero_feature_coef
})

# Define the path to save the CSV file
selected_features_csv_path = os.path.join(base_path, 'non_zero_featureslr.csv')

# Save the selected feature names and coefficients to CSV
selected_features_df.to_csv(selected_features_csv_path, index=False)

# Train SVM model on selected features
svm_model = SVC(
    C=1.0,
    kernel='linear',
    probability=True,  # Enable probability estimates
    class_weight={0:1, 1:2},
    random_state=None
)
svm_model.fit(train_x_selected, train_y)

# Save the SVM model
svm_model_path = os.path.join(base_path, 'PASsvm_model.joblib')
dump(svm_model, svm_model_path)

# Predict probabilities using the SVM model
probabilities_train = svm_model.predict_proba(train_x_selected)[:, 1]
probabilities_test = svm_model.predict_proba(test_x_selected)[:, 1]

# Compute AUC
auc_train = roc_auc_score(train_y, probabilities_train)
auc_test = roc_auc_score(test_y, probabilities_test)

# Plot ROC curve
fpr_train, tpr_train, _ = roc_curve(train_y, probabilities_train)
fpr_test, tpr_test, _ = roc_curve(test_y, probabilities_test)

plt.figure()
plt.plot(fpr_train, tpr_train, label=f'Train AUC = {auc_train:.4f}')
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc_test:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVM')
plt.legend(loc='lower right')
plt.show()

# Compute accuracy
acc_train = accuracy_score(train_y, svm_model.predict(train_x_selected))
acc_test = accuracy_score(test_y, svm_model.predict(test_x_selected))

# Save predictions with probabilities to CSV
train_output = pd.DataFrame({
    'sample_name': train_data.iloc[:, 0],
    'True Class': train_y,
    'Predicted Class': svm_model.predict(train_x_selected),
    'Probability Class 1': probabilities_train
})
test_output = pd.DataFrame({
    'sample_name': test_data.iloc[:, 0],
    'True Class': test_y,
    'Predicted Class': svm_model.predict(test_x_selected),
    'Probability Class 1': probabilities_test
})

train_output.to_csv(os.path.join(base_path, 'PASsvm_train_predictions.csv'), index=False)
test_output.to_csv(os.path.join(base_path, 'PASsvm_test_predictions.csv'), index=False)

# Output results
print(f"Train AUC: {auc_train:.4f}, Train Accuracy: {acc_train:.4f}")
print(f"Test AUC: {auc_test:.4f}, Test Accuracy: {acc_test:.4f}")
