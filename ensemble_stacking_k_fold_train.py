import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.utils.data_utils import get_class_imbalance_ratio, IterativeStratifiedSplitter
from open_pom_SCENE.openpom.models.mpnn_pom import MPNNPOMModel as SCENE_MPNNPOMModel
from open_pom_SchNet.openpom.models.mpnn_pom import MPNNPOMModel as SCHNET_MPNNPOMModel
from open_pom_R_GCN.openpom.models.mpnn_pom import MPNNPOMModel as R_GCN_MPNNPOMModel
from open_pom_PNAConv.openpom.models.mpnn_pom import MPNNPOMModel as PNAConv_MPNNPOMModel
from open_pom_main.openpom.models.mpnn_pom import MPNNPOMModel as NNCONV_MPNNPOMModel
from open_pom_GMMConv.openpom.models.mpnn_pom import MPNNPOMModel as GMMCONV_MPNNPOMModel
from open_pom_GatedGraphConv.openpom.models.mpnn_pom import MPNNPOMModel as GATEDGRAPHCONV_MPNNPOMModel
from open_pom_GatedGCNConv.openpom.models.mpnn_pom import MPNNPOMModel as GATEDGCNCONV_MPNNPOMModel
from open_pom_EGNN.openpom.models.mpnn_pom import MPNNPOMModel as EGNN_MPNNPOMModel
from open_pom_DGNConv.openpom.models.mpnn_pom import MPNNPOMModel as DGNCONV_MPNNPOMModel
from datetime import datetime
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
import pandas as pd
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression

TASKS = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
    'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
    'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
    'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
    'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
    'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
    'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
    'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
    'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
    'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
    'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
    'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
    'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
    'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
    'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
    'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
    'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
    'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
    'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
    'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]

print("No of tasks: ", len(TASKS))
n_tasks = len(TASKS)

input_file = 'curated_GS_LF_merged_4983.csv'  # or new downloaded file path

featurizer = GraphFeaturizer()

smiles_field = 'nonStereoSMILES'

loader = dc.data.CSVLoader(tasks=TASKS,
                           feature_field=smiles_field,
                           featurizer=featurizer)
dataset = loader.create_dataset(inputs=[input_file])
n_tasks = len(dataset.tasks)

# get k folds list
k = 5
splitter = IterativeStratifiedSplitter(order=2)
directories = [''] * 2 * k
for fold in range(k):
    directories[2 * fold] = f'./ensemble_cv_exp/fold_{fold + 1}/train_data'
    directories[2 * fold + 1] = f'./ensemble_cv_exp/fold_{fold + 1}/cv_data'
folds_list = splitter.k_fold_split(dataset=dataset, k=k, directories=directories)


def benchmark_ensemble_stacking_kfold(fold, dataset, test_dataset, model_list, nb_epoch):
    train_ratios = get_class_imbalance_ratio(dataset)
    assert len(train_ratios) == n_tasks

    # learning_rate = 0.001
    learning_rate = dc.models.optimizers.ExponentialDecay(initial_rate=0.001, decay_rate=0.5, decay_steps=32 * 20,
                                                          staircase=True)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

    dir = []
    n_models = len(model_list)
    for i in range(n_models):
        train_dir = f"./ensemble_cv_exp/fold_{fold + 1}/train_data/train{i + 1}"
        valid_dir = f"./ensemble_cv_exp/fold_{fold + 1}/train_data/valid{i + 1}"
        dir.append(train_dir)
        dir.append(valid_dir)
    splitter = dc.splits.RandomSplitter()
    k_fold_datasets = splitter.k_fold_split(dataset, k=n_models, directories=dir)

    for i in tqdm(model_list):
        print(f"model {i + 1} training now.")

        df_test_true = pd.DataFrame(test_dataset.y)
        folder_path = f'./ensemble_cv_exp/fold{fold + 1}/model{i + 1}/result_test'
        # 检查文件夹是否存在，如果不存在则创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        df_test_true.to_excel(f'./ensemble_cv_exp/fold{fold + 1}/model{i + 1}/result_test/true_test.xlsx', index=False,
                              header=False)

        for j, fold_dataset in enumerate(k_fold_datasets):
            print(f"fold {j + 1} training now.")
            train_dataset = fold_dataset[0]
            valid_dataset = fold_dataset[1]
            if i == 0:
                model = SCENE_MPNNPOMModel(n_tasks=n_tasks,
                                           batch_size=128,
                                           learning_rate=learning_rate,
                                           class_imbalance_ratio=train_ratios,
                                           loss_aggr_type='sum',
                                           node_out_feats=100,
                                           layer_dims=[100, 100, 100, 100, 100, 100],
                                           num_heads=10,
                                           edge_hidden_feats=75,
                                           edge_out_feats=100,
                                           num_step_message_passing=5,
                                           mpnn_residual=True,
                                           message_aggregator_type='sum',
                                           mode='classification',
                                           number_atom_features=GraphConvConstants.ATOM_FDIM,
                                           number_bond_features=GraphConvConstants.BOND_FDIM,
                                           n_classes=1,
                                           readout_type='set2set',
                                           num_step_set2set=3,
                                           num_layer_set2set=2,
                                           ffn_hidden_list=[392, 392],
                                           ffn_embeddings=256,
                                           ffn_activation='relu',
                                           ffn_dropout_p=0.12,
                                           ffn_dropout_at_input_no_act=False,
                                           weight_decay=1e-5,
                                           self_loop=False,
                                           optimizer_name='adam',
                                           log_frequency=32,
                                           model_dir=f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/experiments_{j + 1}',
                                           device_name='cuda')
            elif i == 1:
                model = SCHNET_MPNNPOMModel(n_tasks=n_tasks,
                                            batch_size=128,
                                            learning_rate=learning_rate,
                                            class_imbalance_ratio=train_ratios,
                                            loss_aggr_type='sum',
                                            node_out_feats=100,
                                            edge_hidden_feats=75,
                                            edge_out_feats=100,
                                            num_step_message_passing=5,
                                            mpnn_residual=True,
                                            message_aggregator_type='sum',
                                            mode='classification',
                                            number_atom_features=GraphConvConstants.ATOM_FDIM,
                                            number_bond_features=GraphConvConstants.BOND_FDIM,
                                            n_classes=1,
                                            readout_type='set2set',
                                            num_step_set2set=3,
                                            num_layer_set2set=2,
                                            ffn_hidden_list=[392, 392],
                                            ffn_embeddings=256,
                                            ffn_activation='relu',
                                            ffn_dropout_p=0.12,
                                            ffn_dropout_at_input_no_act=False,
                                            weight_decay=1e-5,
                                            self_loop=False,
                                            optimizer_name='adam',
                                            log_frequency=32,
                                            model_dir=f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/experiments_{j + 1}',
                                            device_name='cuda')
            elif i == 2:
                model = R_GCN_MPNNPOMModel(n_tasks=n_tasks,
                                           batch_size=128,
                                           learning_rate=learning_rate,
                                           class_imbalance_ratio=train_ratios,
                                           loss_aggr_type='sum',
                                           node_out_feats=100,
                                           edge_hidden_feats=75,
                                           edge_out_feats=100,
                                           num_step_message_passing=5,
                                           mpnn_residual=True,
                                           message_aggregator_type='sum',
                                           mode='classification',
                                           number_atom_features=GraphConvConstants.ATOM_FDIM,
                                           number_bond_features=GraphConvConstants.BOND_FDIM,
                                           n_classes=1,
                                           readout_type='set2set',
                                           num_step_set2set=3,
                                           num_layer_set2set=2,
                                           ffn_hidden_list=[392, 392],
                                           ffn_embeddings=256,
                                           ffn_activation='relu',
                                           ffn_dropout_p=0.12,
                                           ffn_dropout_at_input_no_act=False,
                                           weight_decay=1e-5,
                                           self_loop=False,
                                           optimizer_name='adam',
                                           log_frequency=32,
                                           model_dir=f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/experiments_{j + 1}',
                                           device_name='cuda')

            elif i == 3:
                model = PNAConv_MPNNPOMModel(n_tasks=n_tasks,
                                             batch_size=128,
                                             learning_rate=learning_rate,
                                             class_imbalance_ratio=train_ratios,
                                             loss_aggr_type='sum',
                                             node_out_feats=100,
                                             edge_hidden_feats=75,
                                             edge_out_feats=100,
                                             num_step_message_passing=5,
                                             mpnn_residual=True,
                                             message_aggregator_type=['mean', 'max', 'sum'],
                                             scalers=['identity', 'amplification'],
                                             delta=2.5,
                                             num_towers=5,
                                             mode='classification',
                                             number_atom_features=GraphConvConstants.ATOM_FDIM,
                                             number_bond_features=GraphConvConstants.BOND_FDIM,
                                             n_classes=1,
                                             readout_type='set2set',
                                             num_step_set2set=3,
                                             num_layer_set2set=2,
                                             ffn_hidden_list=[392, 392],
                                             ffn_embeddings=256,
                                             ffn_activation='relu',
                                             ffn_dropout_p=0.12,
                                             ffn_dropout_at_input_no_act=False,
                                             weight_decay=1e-5,
                                             self_loop=False,
                                             optimizer_name='adam',
                                             log_frequency=32,
                                             model_dir=f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/experiments_{j + 1}',
                                             device_name='cuda')

            elif i == 4:
                model = NNCONV_MPNNPOMModel(n_tasks=n_tasks,
                                            batch_size=128,
                                            learning_rate=learning_rate,
                                            class_imbalance_ratio=train_ratios,
                                            loss_aggr_type='sum',
                                            node_out_feats=100,
                                            edge_hidden_feats=75,
                                            edge_out_feats=100,
                                            num_step_message_passing=5,
                                            mpnn_residual=True,
                                            message_aggregator_type='sum',
                                            mode='classification',
                                            number_atom_features=GraphConvConstants.ATOM_FDIM,
                                            number_bond_features=GraphConvConstants.BOND_FDIM,
                                            n_classes=1,
                                            readout_type='set2set',
                                            num_step_set2set=3,
                                            num_layer_set2set=2,
                                            ffn_hidden_list=[392, 392],
                                            ffn_embeddings=256,
                                            ffn_activation='relu',
                                            ffn_dropout_p=0.12,
                                            ffn_dropout_at_input_no_act=False,
                                            weight_decay=1e-5,
                                            self_loop=False,
                                            optimizer_name='adam',
                                            log_frequency=32,
                                            model_dir=f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/experiments_{j + 1}',
                                            device_name='cuda')

            elif i == 5:
                model = GMMCONV_MPNNPOMModel(n_tasks=n_tasks,
                                             batch_size=128,
                                             learning_rate=learning_rate,
                                             class_imbalance_ratio=train_ratios,
                                             loss_aggr_type='sum',
                                             node_out_feats=100,
                                             edge_hidden_feats=75,
                                             edge_out_feats=100,
                                             num_step_message_passing=5,
                                             mpnn_residual=True,
                                             message_aggregator_type='sum',
                                             mode='classification',
                                             number_atom_features=GraphConvConstants.ATOM_FDIM,
                                             number_bond_features=GraphConvConstants.BOND_FDIM,
                                             n_classes=1,
                                             readout_type='set2set',
                                             num_step_set2set=3,
                                             num_layer_set2set=2,
                                             ffn_hidden_list=[392, 392],
                                             ffn_embeddings=256,
                                             ffn_activation='relu',
                                             ffn_dropout_p=0.12,
                                             ffn_dropout_at_input_no_act=False,
                                             weight_decay=1e-5,
                                             self_loop=False,
                                             optimizer_name='adam',
                                             log_frequency=32,
                                             model_dir=f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/experiments_{j + 1}',
                                             device_name='cuda')

            elif i == 6:
                model = GATEDGRAPHCONV_MPNNPOMModel(n_tasks=n_tasks,
                                                    batch_size=128,
                                                    learning_rate=learning_rate,
                                                    class_imbalance_ratio=train_ratios,
                                                    loss_aggr_type='sum',
                                                    node_out_feats=100,
                                                    edge_hidden_feats=75,
                                                    edge_out_feats=100,
                                                    num_step_message_passing=5,
                                                    mpnn_residual=True,
                                                    message_aggregator_type='sum',
                                                    mode='classification',
                                                    number_atom_features=GraphConvConstants.ATOM_FDIM,
                                                    number_bond_features=GraphConvConstants.BOND_FDIM,
                                                    n_classes=1,
                                                    readout_type='set2set',
                                                    num_step_set2set=3,
                                                    num_layer_set2set=2,
                                                    ffn_hidden_list=[392, 392],
                                                    ffn_embeddings=256,
                                                    ffn_activation='relu',
                                                    ffn_dropout_p=0.12,
                                                    ffn_dropout_at_input_no_act=False,
                                                    weight_decay=1e-5,
                                                    self_loop=False,
                                                    optimizer_name='adam',
                                                    log_frequency=32,
                                                    model_dir=f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/experiments_{j + 1}',
                                                    device_name='cuda')
            elif i == 7:
                model = GATEDGCNCONV_MPNNPOMModel(n_tasks=n_tasks,
                                                  batch_size=128,
                                                  learning_rate=learning_rate,
                                                  class_imbalance_ratio=train_ratios,
                                                  loss_aggr_type='sum',
                                                  node_out_feats=100,
                                                  edge_hidden_feats=75,
                                                  edge_out_feats=100,
                                                  num_step_message_passing=5,
                                                  mpnn_residual=True,
                                                  message_aggregator_type='sum',
                                                  mode='classification',
                                                  number_atom_features=GraphConvConstants.ATOM_FDIM,
                                                  number_bond_features=GraphConvConstants.BOND_FDIM,
                                                  n_classes=1,
                                                  readout_type='set2set',
                                                  num_step_set2set=3,
                                                  num_layer_set2set=2,
                                                  ffn_hidden_list=[392, 392],
                                                  ffn_embeddings=256,
                                                  ffn_activation='relu',
                                                  ffn_dropout_p=0.12,
                                                  ffn_dropout_at_input_no_act=False,
                                                  weight_decay=1e-5,
                                                  self_loop=False,
                                                  optimizer_name='adam',
                                                  log_frequency=32,
                                                  model_dir=f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/experiments_{j + 1}',
                                                  device_name='cuda')

            elif i == 8:
                model = EGNN_MPNNPOMModel(n_tasks=n_tasks,
                                          batch_size=128,
                                          learning_rate=learning_rate,
                                          class_imbalance_ratio=train_ratios,
                                          loss_aggr_type='sum',
                                          node_out_feats=100,
                                          edge_hidden_feats=75,
                                          edge_out_feats=100,
                                          num_step_message_passing=5,
                                          mpnn_residual=True,
                                          message_aggregator_type='sum',
                                          mode='classification',
                                          number_atom_features=GraphConvConstants.ATOM_FDIM,
                                          number_bond_features=GraphConvConstants.BOND_FDIM,
                                          n_classes=1,
                                          readout_type='set2set',
                                          num_step_set2set=3,
                                          num_layer_set2set=2,
                                          ffn_hidden_list=[392, 392],
                                          ffn_embeddings=256,
                                          ffn_activation='relu',
                                          ffn_dropout_p=0.12,
                                          ffn_dropout_at_input_no_act=False,
                                          weight_decay=1e-5,
                                          self_loop=False,
                                          optimizer_name='adam',
                                          log_frequency=32,
                                          model_dir=f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/experiments_{j + 1}',
                                          device_name='cuda')

            elif i == 9:
                model = DGNCONV_MPNNPOMModel(n_tasks=n_tasks,
                                             batch_size=128,
                                             learning_rate=learning_rate,
                                             class_imbalance_ratio=train_ratios,
                                             loss_aggr_type='sum',
                                             node_out_feats=100,
                                             edge_hidden_feats=75,
                                             edge_out_feats=100,
                                             num_step_message_passing=5,
                                             mpnn_residual=True,
                                             message_aggregator_type=['dir1-av', 'dir1-dx', 'sum'],
                                             scalers=['identity', 'amplification'],
                                             delta=2.5,
                                             num_towers=5,
                                             mode='classification',
                                             number_atom_features=GraphConvConstants.ATOM_FDIM,
                                             number_bond_features=GraphConvConstants.BOND_FDIM,
                                             n_classes=1,
                                             readout_type='set2set',
                                             num_step_set2set=3,
                                             num_layer_set2set=2,
                                             ffn_hidden_list=[392, 392],
                                             ffn_embeddings=256,
                                             ffn_activation='relu',
                                             ffn_dropout_p=0.12,
                                             ffn_dropout_at_input_no_act=False,
                                             weight_decay=1e-5,
                                             self_loop=False,
                                             optimizer_name='adam',
                                             log_frequency=32,
                                             model_dir=f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/experiments_{j + 1}',
                                             device_name='cuda')

            start_time = datetime.now()

            # fit model
            loss = model.fit(
                train_dataset,
                nb_epoch=nb_epoch,
                max_checkpoints_to_keep=1,
                deterministic=False,
                restore=False)
            end_time = datetime.now()

            train_scores = model.evaluate(train_dataset, [metric])['roc_auc_score']
            print(
                f"loss = {loss}; train_scores = {train_scores};time_taken = {str(end_time - start_time)}")
            model.save_checkpoint()

            # predict on test set
            preds_test = model.predict(test_dataset)
            df_preds_test = pd.DataFrame(preds_test)
            folder_path = f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/result_test'

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            df_preds_test.to_excel(f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/result_test/pred_test_{j + 1}.xlsx',
                                   index=False, header=False)

            # predict on validation set
            val_preds = model.predict(valid_dataset)
            df_preds_val = pd.DataFrame(val_preds)
            folder_path = f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/result_vail'

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            df_preds_val.to_excel(f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/result_vail/pred_vail_{j + 1}.xlsx',
                                  index=False, header=False)

            # save true labels
            df_true_val = pd.DataFrame(valid_dataset.y)
            folder_path = f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/result_vail'

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            df_true_val.to_excel(f'./ensemble_cv_exp/fold_{fold + 1}/model{i + 1}/result_vail/true_vail_{j + 1}.xlsx',
                                 index=False, header=False)

            # delete model
            del model
            torch.cuda.empty_cache()


model_list = [0, 1, 2, 3, 4]
nb_epoch = 62
for fold in tqdm(range(k)):
    print(f"Fold {fold + 1} ensemble starting now.")
    train_dataset = dc.data.DiskDataset(directories[2 * fold])
    test_dataset = dc.data.DiskDataset(directories[2 * fold + 1])
    print("train_dataset: ", len(train_dataset))
    print("test_dataset: ", len(test_dataset))
    benchmark_ensemble_stacking_kfold(fold=fold,
                                      dataset=train_dataset,
                                      test_dataset=test_dataset,
                                      model_list=model_list,
                                      nb_epoch=nb_epoch)