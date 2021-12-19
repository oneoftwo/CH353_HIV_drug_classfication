import model as MODEL
import parse as PARSE
import train as TRAIN
import config as CONFIG

import pickle
import random
import torch
from torch.utils.data import DataLoader
import os


# log
print('testing_mode:', CONFIG.testing_mode)
print('oversampling rate:', CONFIG.oversampling_rate)
print('max_atom:', CONFIG.max_atom)
print()
print('hid_dim:', CONFIG.hid_dim)
print('n_layer:', CONFIG.n_layer)
print('use_dropout:', CONFIG.use_dropout)
print()
print('batch_size:', CONFIG.batch_size)
print('learning rate:', CONFIG.learning_rate)
print()

# make dir
save_dir = CONFIG.save_dir
if not save_dir[-1] == '/':
    save_dir = save_dir + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(save_dir + 'model/'):
    os.makedirs(save_dir + 'model/')

# from txt get smiles list
major_smiles_list = PARSE.get_smiles_list('./negative_provided.txt') # 22677
minor_smiles_list = PARSE.get_smiles_list('./positive_provided.txt') # 708
random.shuffle(major_smiles_list)
random.shuffle(minor_smiles_list)

if CONFIG.testing_mode:
    major_smiles_list = major_smiles_list[:700]
    minor_smiles_list = minor_smiles_list[:700]

l_major = len(major_smiles_list)
l_minor = len(minor_smiles_list)

# split and oversample
r = CONFIG.val_ratio

train_smiles_list = major_smiles_list[int(l_major * r):] + \
        minor_smiles_list[int(l_minor * r):] * CONFIG.oversampling_rate
train_label_list = [0 for _ in range(l_major - int(l_major * r))] + \
        [1 for _ in range(l_minor - int(l_minor * r))] * CONFIG.oversampling_rate

val_smiles_list = major_smiles_list[:int(l_major * r)] + \
        minor_smiles_list[:int(l_minor * r)]
val_label_list = [0 for _ in range(int(l_major * r))] + \
        [1 for _ in range(int(l_minor * r))]

# make dataset
max_atom = CONFIG.max_atom
props = CONFIG.additional_properties
node_feature = CONFIG.node_feature
train_dataset = PARSE.GCNDataset(train_smiles_list, train_label_list, \
        progress_bar=False, print_faults=False, max_atom=max_atom, \
        node_feature=node_feature, add_node_features=props, \
        remove_aromatic=False, self_loop=True)
val_dataset = PARSE.GCNDataset(val_smiles_list, val_label_list, \
        progress_bar=False, print_faults=False, max_atom=max_atom, \
        node_feature=node_feature, add_node_features=props, \
        remove_aromatic=False, self_loop=True)

# make loader
train_batch_size = CONFIG.batch_size
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, \
        shuffle=True, num_workers=4, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=train_batch_size, \
        shuffle=True, num_workers=4, drop_last=False)

# define model
model = MODEL.VanillaMPNN(train_dataset[0]['h'].size(1), \
        train_dataset[0]['e'].size(2), CONFIG.hid_dim, CONFIG.n_layer)

print(model)

# train model
lr=CONFIG.learning_rate
train_loss_history, train_acc_history = [], []
train_precision_history, train_auroc_history = [], []
val_loss_history, val_acc_history = [], []
val_precision_history, val_auroc_history = [], []
val_model_score_history = []

for epoch_idx in range(CONFIG.n_epoch):
    print('epoch: {}'.format(epoch_idx))

    # train
    lr = lr * 0.97
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, train_loss, train_acc = TRAIN.processor(model.cuda(), train_loader, optimizer=optimizer)
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    
    # validate
    model, val_loss, val_acc = TRAIN.processor(model.cuda(), val_loader, optimizer=None)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)
    
    save_dir = CONFIG.save_dir
    if not save_dir[-1] == '/':
        save_dir = save_dir + '/'
    pickle.dump(model, open(save_dir + 'model/model_{}.pkl'.format(epoch_idx), 'wb'))
    
    precision, auroc = TRAIN.get_evaluation_criteria(model, val_loader)
    val_precision_history.append(precision)
    val_auroc_history.append(auroc)

    model_score = 3 / ((1 / (val_acc + 1e-6)) + (1 / (5 * (precision + 1e-6))) + (1 / (3 * (auroc + 1e-6))))
    val_model_score_history.append(model_score)

    # print values
    print('precision: {:3.4f}'.format(precision))
    print('AUROC: {:3.4f}'.format(auroc))
    print('t loss: {:3.4f}, t acc: {:3.4f}'.format(train_loss, train_acc))
    print('v loss: {:3.4f}, v acc: {:3.4f}'.format(val_loss, val_acc))
    print('model score: {:3.4f}'.format(model_score))
    print()

    from matplotlib import pyplot as plt
    
    save_dir = CONFIG.save_dir
    if not save_dir[-1] == '/':
        save_dir = save_dir + '/' 

    plt.plot(train_loss_history)
    plt.plot(val_loss_history)
    plt.savefig(save_dir + 'plot_loss')
    plt.clf()

    plt.plot(train_acc_history)
    plt.plot(val_acc_history)
    plt.savefig(save_dir + 'plot_acc')
    plt.clf()
    
    plt.plot(val_precision_history)
    plt.savefig(save_dir + 'plot_precision')
    plt.clf()

    plt.plot(val_auroc_history)
    plt.savefig(save_dir + 'plot_auroc')
    plt.clf()

    plt.plot(val_model_score_history)
    plt.savefig(save_dir + 'plot_score')
    plt.clf()

    

