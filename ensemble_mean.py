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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, precision_score, \
    recall_score, f1_score, roc_curve, auc as sklearn_auc
import pandas as pd
import os
import torch.optim as optim
import matplotlib.pyplot as plt


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


def benchmark_ensemble(fold, train_dataset, test_dataset, n_models, nb_epoch, mean_auc = 0.0):
    train_ratios = get_class_imbalance_ratio(train_dataset)
    assert len(train_ratios) == n_tasks

    # learning_rate = 0.001
    learning_rate = dc.models.optimizers.ExponentialDecay(initial_rate=0.001, decay_rate=0.5, decay_steps=32 * 20,
                                                          staircase=True)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

    for i in tqdm(range(n_models)):
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
                                       model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                        model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
                                        device_name='cuda')
        elif i == 2:
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
                                        model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                         model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
                                         device_name='cuda')

        elif i == 4:
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
                                       model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                         model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                                model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                              model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                      model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                         model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
        test_scores = model.evaluate(test_dataset, [metric])['roc_auc_score']
        print(
            f"loss = {loss}; train_scores = {train_scores}; test_scores = {test_scores}; time_taken = {str(end_time - start_time)}")
        model.save_checkpoint()  # saves final checkpoint => `checkpoint2.pt`
        del model
        torch.cuda.empty_cache()

    # Get test score from the ensemble
    list_preds = []
    for i in range(n_models):
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
                                       model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                        model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
                                        device_name='cuda')
        elif i == 2:
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
                                        model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                         model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
                                         device_name='cuda')

        elif i == 4:
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
                                       model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                         model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                                model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                              model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                      model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
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
                                         model_dir=f'./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}',
                                         device_name='cuda')

        model.restore(f"./ensemble_cv_exp/mean/ensemble_fold_{fold + 1}/experiments_{i + 1}/checkpoint2.pt")
        test_scores = model.evaluate(test_dataset, [metric])['roc_auc_score']
        print("test_score: ", test_scores)

        preds_train = model.predict(train_dataset)
        df_preds_train = pd.DataFrame(preds_train)
        folder_path = f'./ensemble_cv_exp/mean/train_pred_fold{fold + 1}'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        df_preds_train.to_excel(f'./ensemble_cv_exp/mean/train_pred_fold{fold + 1}/preds_train_{i + 1}.xlsx', index=False, header=False)

        preds_test = model.predict(test_dataset)
        df_preds_test = pd.DataFrame(preds_test)
        # 
        folder_path = f'./ensemble_cv_exp/mean/test_pred_fold{fold + 1}'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # 
        df_preds_test.to_excel(f'./ensemble_cv_exp/mean/test_pred_fold{fold + 1}/preds_test_{i + 1}.xlsx', index=False, header=False)
        list_preds.append(preds_test)
        
    df_preds_test_true = pd.DataFrame(test_dataset.y)
    df_preds_test_true.to_excel(f'./ensemble_cv_exp/mean/test_pred_fold{fold + 1}/test_true.xlsx', index=False, header=False)
    
    preds_arr = np.asarray(list_preds)
    ensemble_preds = np.mean(preds_arr, axis=0)

    threshold = 0.65
    y_pred = (ensemble_preds >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(test_dataset.y, y_pred, average=None, zero_division=0)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(test_dataset.y, y_pred,
                                                                                 average='macro',
                                                                                 zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(test_dataset.y, y_pred,
                                                                                 average='micro',
                                                                                 zero_division=0)

    mi_average = [precision_micro, recall_micro, f1_micro]
    roc_auc = roc_auc_score(test_dataset.y, ensemble_preds, average='macro', multi_class='ovr')
    folds_results.append(roc_auc)
    micro_averages.append(mi_average)

    # Calculate ROC curve and AUC for each class
    fpr, tpr, _ = roc_curve(test_dataset.y.ravel(), ensemble_preds.ravel())
    roc_auc_value = sklearn_auc(fpr, tpr)
    mean_auc += roc_auc_value
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

    return roc_auc_score(test_dataset.y, ensemble_preds, average="macro")

# k-fold ensemble cv
n_models = 10
nb_epoch = 62
folds_results = []
micro_averages = []
mean_fpr = np.linspace(0, 1, 100)
tprs = []
mean_auc = 0.0
for fold in tqdm(range(k)):
    print(f"Fold {fold + 1} ensemble starting now.")
    train_dataset = dc.data.DiskDataset(directories[2 * fold])
    test_dataset = dc.data.DiskDataset(directories[2 * fold + 1])
    print("train_dataset: ", len(train_dataset))
    print("test_dataset: ", len(test_dataset))
    fold_result = benchmark_ensemble(fold=fold,train_dataset=train_dataset,test_dataset=test_dataset,n_models=n_models,nb_epoch=nb_epoch,mean_auc = 0.0)
    print(f"Fold {fold + 1} ensemble score: ", fold_result)
    folds_results.append(fold_result)

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
plt.savefig("./ensemble_cv_exp/AUC-ROC/ensemble_mean.png")

cv_mean_result = np.mean(folds_results)
cv_mean_micro_averages = np.mean(micro_averages, axis=0)
with open("./ensemble_cv_exp/final_score_mean.txt", 'w+') as f:
    f.write(f"folds_results = {folds_results}\n")
    f.write(f"cv_mean_result = {cv_mean_result}\n")
    f.write(f"folds_micro_averages = {micro_averages}\n")
    f.write(f"cv_mean_micro_averages = {cv_mean_micro_averages}\n")
