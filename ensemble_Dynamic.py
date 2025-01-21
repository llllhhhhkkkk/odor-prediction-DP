import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import deepchem as dc
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, roc_curve, auc as sklearn_auc
import matplotlib.pyplot as plt

n_models = 10
k = 5
directories = [''] * 2 * k
for fold in range(k):
    directories[2 * fold] = f'./ensemble_cv_exp/fold_{fold + 1}/train_data'
    directories[2 * fold + 1] = f'./ensemble_cv_exp/fold_{fold + 1}/cv_data'

fold = 5
folds_results = []
micro_averages = []

# Initialize variables to calculate the mean ROC
mean_fpr = np.linspace(0, 1, 100)
tprs = []
mean_auc = 0.0
colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'purple']

for k in range(fold):
    train_dataset = dc.data.DiskDataset(directories[2 * k])
    test_dataset = dc.data.DiskDataset(directories[2 * k + 1])
    graph_tests = test_dataset.X
    graph_trains = train_dataset.X

    test_true_label = np.load(f'./ensemble_cv_exp/stacking/result/fold_{k + 1}/test_true_label.npy')
    train_true_label = np.load(f'./ensemble_cv_exp/stacking/result/fold_{k + 1}/train_true_label.npy')
    final_preds = np.zeros_like(test_true_label)
    for i, graph_test in enumerate(graph_tests):
        node_features_test = graph_test.node_features
        edge_features_test = graph_test.edge_features

        if edge_features_test.size == 0:
            edge_features_test = np.zeros((1, edge_features_test.shape[1]))

        graph_feature_test = np.concatenate([node_features_test.mean(axis=0), edge_features_test.mean(axis=0)])
        best_sim = 0
        train_sim_index = 0
        for j, graph_train in enumerate(graph_trains):
            node_features_train = graph_train.node_features
            edge_features_train = graph_train.edge_features

            if edge_features_train.size == 0:
                edge_features_train = np.zeros((1, edge_features_train.shape[1]))

            graph_feature_train = np.concatenate([node_features_train.mean(axis=0), edge_features_train.mean(axis=0)])

            features = np.array([graph_feature_test, graph_feature_train])

            kernel_matrix = rbf_kernel(features, gamma=0.8)

            if kernel_matrix[0][1] > best_sim:
                best_sim = kernel_matrix[0][1]
                train_sim_index = j

        max_score = 0
        max_model = 0
        for z in range(n_models):
            result = []
            model_train_preds_proba = np.load(
                f'./ensemble_cv_exp/stacking/result/fold_{k + 1}/train_preds_{z + 1}.npy')[train_sim_index, :]
            y_true = train_true_label[train_sim_index, :]

            auc = roc_auc_score(y_true, model_train_preds_proba, average='macro', multi_class='ovr')
            average_precision = average_precision_score(y_true, model_train_preds_proba, average='macro')
            result.append(auc)
            result.append(average_precision)

            if np.mean(result) > max_score:
                max_model = z + 1
                max_score = np.mean(result)
        final_preds[i, :] = np.load(f'./ensemble_cv_exp/stacking/result/fold_{k + 1}/test_preds_{max_model}.npy')[i]

    threshold = 0.65
    y_pred = (final_preds >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(test_true_label, y_pred, average=None, zero_division=0)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(test_true_label, y_pred,
                                                                                 average='macro',
                                                                                 zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(test_true_label, y_pred,
                                                                                 average='micro',
                                                                                 zero_division=0)

    mi_average = [precision_micro, recall_micro, f1_micro]
    roc_auc = roc_auc_score(test_true_label, final_preds, average='macro', multi_class='ovr')
    folds_results.append(roc_auc)
    micro_averages.append(mi_average)

    # Calculate ROC curve and AUC for each class
    fpr, tpr, _ = roc_curve(test_true_label.ravel(), final_preds.ravel())
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
plt.savefig("./ensemble_cv_exp/AUC-ROC/ensemble_Dynamic.png")

cv_mean_result = np.mean(folds_results)
cv_mean_micro_averages = np.mean(micro_averages, axis=0)
with open("./ensemble_cv_exp/final_score_Dynamic.txt", 'w+') as f:
    f.write(f"folds_results = {folds_results}\n")
    f.write(f"cv_mean_result = {cv_mean_result}\n")
    f.write(f"folds_micro_averages = {micro_averages}\n")
    f.write(f"cv_mean_micro_averages = {cv_mean_micro_averages}\n")
