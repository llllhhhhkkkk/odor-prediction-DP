import numpy as np
from torch.optim import Adam
import torch
import pandas as pd
from torch import nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, auc as sklearn_auc
import matplotlib.pyplot as plt


# k-fold ensemble cv
n_models = 10
nb_epoch = 62
folds_results = []
micro_averages = []
mean_fpr = np.linspace(0, 1, 100)
tprs = []
mean_auc = 0.0
folds = 5
for fold in range(folds):
    model_predictions = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        predictions_df = pd.read_excel(f'./ensemble_cv_exp/train_pred_fold{fold +1}/preds_train_{i + 1}.xlsx', header=None)
        predictions = predictions_df.values
        model_predictions.append(predictions)

    model_predictions_test = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        predictions_df = pd.read_excel(f'./ensemble_cv_exp/test_pred_fold{fold +1}/preds_test_{i + 1}.xlsx', header=None)
        predictions = predictions_df.values
        model_predictions_test.append(predictions)

    train_dataset_y = pd.read_excel(f'./ensemble_cv_exp/train_pred_fold{fold +1}/train_true.xlsx', header=None)
    train_dataset_y = train_dataset_y.values

    test_dataset_y = pd.read_excel(f'./ensemble_cv_exp/test_pred_fold{fold +1}/test_true.xlsx', header=None)
    test_dataset_y = test_dataset_y.values

    model_predictions = [torch.tensor(pred, dtype=torch.float32) for pred in model_predictions]
    print(model_predictions[0].shape)
    model_predictions_test = [torch.tensor(pred, dtype=torch.float32) for pred in model_predictions_test]
    print(model_predictions_test[0].shape)
    train_dataset_y = torch.tensor(train_dataset_y, dtype=torch.float32)
    print(train_dataset_y.shape)
    weights = torch.ones(10, requires_grad=True, dtype=torch.float32) / 10

    def ensemble_model(predictions, weights):
        weighted_predictions = torch.stack(predictions) * weights[:, None, None]
        return weighted_predictions.sum(0)

    loss_fn = nn.MSELoss()

    weights = torch.nn.Parameter(torch.ones(10, dtype=torch.float32) / 10)


    optimizer = Adam([weights], lr=0.01)


    for epoch in range(100):
        optimizer.zero_grad()
        ensemble_pred = ensemble_model(model_predictions, weights)
        loss = loss_fn(ensemble_pred, train_dataset_y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


    weights = torch.nn.functional.softmax(weights, dim=0)
    print("Optimized weights:", weights)

    stacked_predictions = torch.stack(model_predictions_test)
    weighted_average = torch.einsum('i,ijk->jk', weights, stacked_predictions).detach().numpy()

    threshold = 0.65
    y_pred = (weighted_average >= threshold).astype(int)

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
    roc_auc = roc_auc_score(test_dataset_y, weighted_average, average="macro")

    folds_results.append(roc_auc)
    micro_averages.append(mi_average)

    # Calculate ROC curve and AUC for each class
    fpr, tpr, _ = roc_curve(test_dataset_y.ravel(), weighted_average.ravel())
    roc_auc_value = sklearn_auc(fpr, tpr)
    mean_auc += roc_auc_value
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0


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
plt.savefig("./ensemble_cv_exp/AUC-ROC/ensemble_weight_mean.png")

cv_mean_result = np.mean(folds_results)
cv_mean_micro_averages = np.mean(micro_averages, axis=0)
with open("./ensemble_cv_exp/final_score_weight_mean.txt", 'w+') as f:
    f.write(f"folds_results = {folds_results}\n")
    f.write(f"cv_mean_result = {cv_mean_result}\n")
    f.write(f"folds_micro_averages = {micro_averages}\n")
    f.write(f"cv_mean_micro_averages = {cv_mean_micro_averages}\n")
round(cv_mean_result, 4)