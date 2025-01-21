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
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, auc as sklearn_auc
import os
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
data_smell = dataset.select([0])

k = 5
splitter = IterativeStratifiedSplitter(order=2)
directories = [''] * 2 * k
for fold in range(k):
    directories[2 * fold] = f'./ensemble_cv_exp/fold_{fold + 1}/train_data'
    directories[2 * fold + 1] = f'./ensemble_cv_exp/fold_{fold + 1}/cv_data'
folds_list = splitter.k_fold_split(dataset=dataset, k=k, directories=directories)

train_dataset = dc.data.DiskDataset(directories[2 * 0])
test_dataset = dc.data.DiskDataset(directories[2 * 0 + 1])

train_ratios = get_class_imbalance_ratio(train_dataset)
assert len(train_ratios) == n_tasks

# learning_rate = 0.001
learning_rate = dc.models.optimizers.ExponentialDecay(initial_rate=0.001, decay_rate=0.5, decay_steps=32 * 20,
                                                      staircase=True)
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

model_lists = ['SCENE_MPNNPOMModel', 'SCHNET_MPNNPOMModel', 'NNCONV_MPNNPOMModel', 'PNAConv_MPNNPOMModel', 'R_GCN_MPNNPOMModel',
              'GMMCONV_MPNNPOMModel', 'GATEDGRAPHCONV_MPNNPOMModel', 'GATEDGCNCONV_MPNNPOMModel','EGNN_MPNNPOMModel', 'DGNCONV_MPNNPOMModel']

for model_number, model_name in enumerate(model_lists):
    if model_number == 0:
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
                                   device_name='cuda',
                                   feature_visualization=True)
    elif model_number == 1:
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
                                    device_name='cuda',
                                    feature_visualization=True)
    elif model_number == 2:
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
                                    device_name='cuda',
                                    feature_visualization=True)
    elif model_number == 3:
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
                                     device_name='cuda',
                                     feature_visualization=True)

    elif model_number == 4:
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
                                   device_name='cuda',
                                   feature_visualization=True)

    elif model_number == 5:
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
                                     device_name='cuda',
                                     feature_visualization=True)

    elif model_number == 6:
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
                                            device_name='cuda',
                                            feature_visualization=True)
    elif model_number == 7:
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
                                          device_name='cuda',
                                          feature_visualization=True)

    elif model_number == 8:
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
                                  device_name='cuda',
                                  feature_visualization=True)

    elif model_number == 9:
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
                                     device_name='cuda',
                                     feature_visualization=True)

    model.restore(f"./ensemble_cv_exp/single_model/{model_name}/model/fold_1/experiments_1/checkpoint1.pt")
    preds_test = model.predict(data_smell)
