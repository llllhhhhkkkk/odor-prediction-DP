import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, auc as sklearn_auc
import matplotlib.pyplot as plt


def read_excel_file(file_name):
    try:
        df = pd.read_excel(file_name)
        return df.to_numpy()
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return None


def process_predictions(file_names, quantile=0.95):
    all_predictions = []


    with ThreadPoolExecutor() as executor:
        results = list(executor.map(read_excel_file, file_names))

    for result in results:
        if result is not None:
            all_predictions.append(result)

    if not all_predictions:
        raise ValueError("No valid predictions found.")

    model_predictions_test = np.stack(all_predictions)

    final_predictions = np.zeros_like(model_predictions_test[0])

    for row_idx in range(final_predictions.shape[0]):
        for col_idx in range(final_predictions.shape[1]):
            preds = model_predictions_test[:, row_idx, col_idx]
            mean = np.mean(preds)
            std = np.std(preds)
            coefficient_of_variation = std / mean


            dynamic_threshold = np.quantile(model_predictions_test[:, :, col_idx].std(axis=0) /
                                            model_predictions_test[:, :, col_idx].mean(axis=0), quantile)

            if coefficient_of_variation > dynamic_threshold:
                final_predictions[row_idx, col_idx] = preds.min()
            else:
                final_predictions[row_idx, col_idx] = preds.max()

    return final_predictions

fold = 5
folds_results = []
micro_averages = []

mean_fpr = np.linspace(0, 1, 100)
tprs = []
mean_auc = 0.0

for k in range(fold):

    file_names = [f'./ensemble_cv_exp/test_pred_fold{k + 1}/preds_test_{i + 1}.xlsx' for i in range(10)]
    quantile = 0.98


    final_predictions = process_predictions(file_names, quantile)


    test_dataset_y = pd.read_excel(f'./ensemble_cv_exp/test_pred_fold{k + 1}/test_true.xlsx').to_numpy()
    roc_auc = roc_auc_score(test_dataset_y, final_predictions, average="macro")

    threshold = 0.65
    y_pred = (final_predictions >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(test_dataset_y, y_pred, average=None, zero_division=0)

    # for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
    #     print(f'Label {i}: Precision = {p:.2f}, Recall = {r:.2f}, F1-score = {f:.2f}')


    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(test_dataset_y, y_pred,
                                                                                 average='macro',
                                                                                 zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(test_dataset_y, y_pred,
                                                                                 average='micro',
                                                                                 zero_division=0)

    print(f'Macro-average: Precision = {precision_macro:.3f}, Recall = {recall_macro:.3f}, F1-score = {f1_macro:.3f}')
    print(f'Micro-average: Precision = {precision_micro:.3f}, Recall = {recall_micro:.3f}, F1-score = {f1_micro:.3f}')
    mi_average = [precision_micro, recall_micro, f1_micro]
    print("ROC AUC 分数:", roc_auc)

    folds_results.append(roc_auc)
    micro_averages.append(mi_average)

    # Calculate ROC curve and AUC for each class
    fpr, tpr, _ = roc_curve(test_dataset_y.ravel(), final_predictions.ravel())
    roc_auc_value = sklearn_auc(fpr, tpr)
    mean_auc += roc_auc_value
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

cv_mean_result = np.mean(folds_results)
cv_mean_micro_averages = np.mean(micro_averages, axis=0)

mean_auc /= fold
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_roc_auc = sklearn_auc(mean_fpr, mean_tpr)

# Plot the mean ROC curve
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f)' % mean_roc_auc, lw=2, alpha=.8)
# Plot the ROC curve for each fold
# for i, color in enumerate(colors[:fold]):
#     plt.plot(mean_fpr, tprs[i], lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i+1, folds_results[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for All Folds')
plt.legend(loc="lower right")
plt.savefig("./ensemble_cv_exp/AUC-ROC/ensemble_max_min.png")

with open("ensemble_cv_exp/final_score_max_min.txt", 'w+') as f:
    f.write(f"folds_results = {folds_results}\n")
    f.write(f"cv_mean_result = {cv_mean_result}\n")
    f.write(f"folds_micro_averages = {micro_averages}\n")
    f.write(f"cv_mean_micro_averages = {cv_mean_micro_averages}\n")
round(cv_mean_result, 4)

