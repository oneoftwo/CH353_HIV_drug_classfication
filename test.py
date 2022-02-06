import sys
import pickle
from torch.utils.data import DataLoader
import _model as MODEL
import _train as TRAIN
import _parse as PARSE
import config as CONFIG


###############################################################################
# config ######################################################################

# pos and neg txt file to test
positive_dir = './data/positive_provided.txt'
negative_dir = './data/negative_provided.txt'

# model pkl file
# oversampling 1 (no oversampling)
# model_dir = './final_models/model_best_os1.pkl'
# oversampling 5
model_dir = './exp_result/model/model_200.pkl'
# oversampling 20
# model_dir = './final_models/model_best_os20.pkl'

###############################################################################

model = pickle.load(open(model_dir, 'rb'))
print(model)

minor_smiles = PARSE.get_smiles_list(positive_dir)
major_smiles = PARSE.get_smiles_list(negative_dir)

total_smiles = major_smiles + minor_smiles
total_label = [0 for _ in major_smiles] + [1 for _ in minor_smiles]

max_atom = CONFIG.max_atom
props = CONFIG.additional_properties
node_feature = CONFIG.node_feature
test_dataset = PARSE.GCNDataset(total_smiles, total_label, progress_bar=False, \
        print_faults=False, max_atom=max_atom, node_feature=node_feature, \
        add_node_features=props, remove_aromatic=False, self_loop=True)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, \
        num_workers=4, drop_last=False)

model, loss, acc = TRAIN.processor(model.cuda(), test_loader)
precision, auroc = TRAIN.get_evaluation_criteria(model, test_loader)


model_score = 3 / ((1 / (acc + 1e-6)) + (1 / (5 * (precision + 1e-6))) + \
        (1 / (3 * (auroc + 1e-6))))


print('precision: {:3.4f}'.format(precision))                               
print('AUROC: {:3.4f}'.format(auroc))                                       
print('loss: {:3.4f}, acc: {:3.4f}'.format(loss, acc))          
print('model score: {:3.4f}'.format(model_score))                           
print()


