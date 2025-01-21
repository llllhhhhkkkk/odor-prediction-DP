import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, roc_curve, auc as sklearn_auc

folds = 5
n_models = 10
folds_results = []
micro_averages = []

# Initialize variables to calculate the mean ROC
mean_fpr = np.linspace(0, 1, 100)
tprs = []
mean_auc = 0.0
colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'purple']

for k in range(folds):
    train_data_true = np.load(f'./ensemble_cv_exp/stacking/result/fold_{k + 1}/train_true_label.npy')
    test_data_true = np.load(f'./ensemble_cv_exp/stacking/result/fold_{k + 1}/test_true_label.npy')
    test_data_preds = []
    train_data_preds = []
    
    for i in range(n_models):
        test_data_preds.append(np.load(f'./ensemble_cv_exp/stacking/result/fold_{k + 1}/test_preds_{i + 1}.npy'))
    test_data_preds = np.array(test_data_preds)
    test_stacked_features = test_data_preds.transpose((1, 0, 2)).reshape(test_data_preds.shape[1], -1)
    print(test_stacked_features.shape)
    
    for i in range(n_models):   
        train_data_preds.append(np.load(f'./ensemble_cv_exp/stacking/result/fold_{k + 1}/train_preds_{i + 1}.npy'))
    train_data_preds = np.array(train_data_preds)
    train_stacked_features = train_data_preds.transpose((1, 0, 2)).reshape(train_data_preds.shape[1], -1)
    print(train_stacked_features.shape)
    
    n_labels = 138
    models = []
    for i in range(n_labels):
        model = LogisticRegression(max_iter=3000)
        model.fit(train_stacked_features, train_data_true[:, i])
        models.append(model)
        

    # y_pred = np.zeros_like(test_data_true)
    #
    # for i in range(n_labels):
    #     y_pred[:, i] = models[i].predict(test_stacked_features)
    # df_predictions = pd.DataFrame(y_pred, columns=[f'Label_{i}' for i in range(1, 139)])
    # df_predictions.to_csv('predictions.csv', index=False)
    # overall_accuracy = np.mean([accuracy_score(test_data_true[:, i], y_pred[:, i]) for i in range(n_labels)])
    # print("Overall Accuracy:", overall_accuracy)
    
    y_pred_probs = np.zeros((test_data_true.shape[0], n_labels))
    
    for i in range(n_labels):

        y_pred_probs[:, i] = models[i].predict_proba(test_stacked_features)[:, 1]

    threshold = 0.65
    y_pred = (y_pred_probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(test_data_true, y_pred, average=None, zero_division=0)

    # for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
    #     print(f'Label {i}: Precision = {p:.2f}, Recall = {r:.2f}, F1-score = {f:.2f}')


    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(test_data_true, y_pred,
                                                                                 average='macro',
                                                                                 zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(test_data_true, y_pred,
                                                                                 average='micro',
                                                                                 zero_division=0)

    print(f'Macro-average: Precision = {precision_macro:.3f}, Recall = {recall_macro:.3f}, F1-score = {f1_macro:.3f}')
    print(f'Micro-average: Precision = {precision_micro:.3f}, Recall = {recall_micro:.3f}, F1-score = {f1_micro:.3f}')
    mi_average = [precision_micro, recall_micro, f1_micro]
    fold_result =  roc_auc_score(test_data_true, y_pred_probs, average="macro")
    folds_results.append(fold_result)
    micro_averages.append(mi_average)

    # Calculate ROC curve and AUC for each class
    fpr, tpr, _ = roc_curve(test_data_true.ravel(), y_pred_probs.ravel())
    roc_auc_value = sklearn_auc(fpr, tpr)
    mean_auc += roc_auc_value
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

cv_mean_result = np.mean(folds_results)
cv_mean_micro_averages = np.mean(micro_averages, axis=0)

mean_auc /= folds
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
plt.savefig("./ensemble_cv_exp/AUC-ROC/ensemble_stacking_logistic.png")

with open("./ensemble_cv_exp/final_score_stacking_logistic.txt", 'w+') as f:
    f.write(f"folds_results = {folds_results}\n")
    f.write(f"cv_mean_result = {cv_mean_result}\n")
    f.write(f"folds_micro_averages = {micro_averages}\n")
    f.write(f"cv_mean_micro_averages = {cv_mean_micro_averages}\n")
round(cv_mean_result, 4)