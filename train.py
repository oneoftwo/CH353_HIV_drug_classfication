import model as MODEL
import parse as PARSE

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, roc_auc_score

# single epoch processor
def processor(model, dataloader, optimizer=None):
    n_data = len(dataloader.dataset)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0
    correct = 0
    if optimizer == None:
        model.eval()
    else:
        model.train()
    for batch in tqdm(dataloader):
        h, e, adj, target = batch['h'], batch['e'], batch['adj'], batch['target']
        N_atom = batch['N'].cuda()
        h, e, adj = h.cuda(), e.cuda(), adj.cuda()
        pred_target = model(h, e, adj, N_atom).cpu()
        batch_loss = loss_fn(pred_target, target)
        total_loss += batch_loss.cpu().item()
        pred_target_idx = pred_target.argmax(dim=1)
        correct += (pred_target_idx == target).sum().item()
        if not optimizer == None:
            batch_loss.backward()
            optimizer.step()
    return model, total_loss / n_data, correct / n_data

def get_evaluation_criteria(model, dataloader):
    model.eval()
    n_data = len(dataloader.dataset)
    pred_list, true_list, pred_prob_list = [], [], []
    for batch in tqdm(dataloader):
        h, e , adj, target = batch['h'], batch['e'], batch['adj'], batch['target']
        N_atom = batch['N'].cuda()
        h, e, adj = h.cuda(), e.cuda(), adj.cuda()
        pred_target = model(h, e, adj, N_atom).cpu()
        
        pred_target_idx = torch.argmax(pred_target, dim=1)
        pred_target_prob = F.softmax(pred_target, dim=1)[:,1]
        
        true_list = true_list + list(target.detach().numpy())
        pred_list = pred_list + list(pred_target_idx.detach().numpy())
        pred_prob_list = pred_prob_list + list(pred_target_prob.detach().numpy())
    # print(pred_prob_list[:10])
    # print(pred_list[:10])
    cm = confusion_matrix(y_pred=pred_list, y_true=true_list)
    # print(cm)
    TN, FP, FN, TP = cm.ravel()
    auroc = roc_auc_score(y_score=pred_prob_list, y_true=true_list)
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    # print(TN, FP, FN, TP)
    # print(auroc)
    # print(precision)
    return precision, auroc

if __name__ == '__main__':
    pass


