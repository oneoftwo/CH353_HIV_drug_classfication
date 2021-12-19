import torch
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit import RDLogger

import numpy as np


# disable rdkit wranings
RDLogger.DisableLog('rdApp.*')


class GCNDataset(Dataset):
    
    def __init__(self, smiles_list, label_list, progress_bar=False, \
            node_feature=['C', 'N', 'O', 'F'], \
            remove_aromatic=True, max_atom=None, self_loop=True, \
            print_faults=False, \
            add_node_features=['charge', 'valence', 'in_ring', \
            'is_aromatic', 'num_hydrogen']):
        
        self.smiles_list = smiles_list
        self.label_list = label_list
        
        self.max_atom = max_atom
        self.node_feature = node_feature
        self.remove_aromatic = remove_aromatic
        self.self_loop = self_loop
        self.add_node_features = add_node_features
        self.print_faults = print_faults

        self.parsed_smiles_list, self.parsed_label_list = [], []
        self.h_list, self.e_list, self.adj_list = [], [], []
        
        self._parse_smiles_list(progress_bar)
        
        self.property_list = []
        self.atom_n_list = []
        for smiles in self.parsed_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            prop = 0
            for atom in mol.GetAtoms():
                if atom.GetIsAromatic:
                    prop = 1
            self.property_list.append(prop)
            self.atom_n_list.append(mol.GetNumAtoms())
    def __len__(self):
        return len(self.parsed_smiles_list)
    
    def __getitem__(self, idx):
        sample = {}
        sample['h'], sample['e'], sample['adj'] = self.h_list[idx], \
                self.e_list[idx], self.adj_list[idx]
        sample['smiles'] = self.parsed_smiles_list[idx]
        sample['target'] = self.parsed_label_list[idx]
        sample['aromatic'] = self.property_list[idx]
        sample['N'] = self.atom_n_list[idx]
        return sample
    
    def _parse_smiles_list(self, progress_bar):
        max_atom = self.max_atom
        if progress_bar:
            from tqdm import tqdm
            for idx, smiles in enumerate(tqdm(self.smiles_list)):
                matrixs = parse_smiles_to_graph(smiles, self.node_feature, \
                        self.remove_aromatic, self.max_atom, self.self_loop, \
                        self.print_faults, self.add_node_features)
                if not matrixs == None:
                    h, e, adj = matrixs
                    self.h_list.append(h)
                    self.e_list.append(e)
                    self.adj_list.append(adj)
                    self.parsed_smiles_list.append(smiles)
                    self.parsed_label_list.append(self.label_list[idx])
        else:
            for idx, smiles in enumerate(self.smiles_list):
                matrixs = parse_smiles_to_graph(smiles, self.node_feature, \
                        self.remove_aromatic, self.max_atom, self.self_loop, \
                        self.print_faults, self.add_node_features)
                if not matrixs == None:
                    h, e, adj = matrixs
                    self.h_list.append(h)
                    self.e_list.append(e)
                    self.adj_list.append(adj)
                    self.parsed_smiles_list.append(smiles)
                    self.parsed_label_list.append(self.label_list[idx])
        


def one_of_k_encoding(x, allowable_set):
        
        '''
        Maps inputs not in the allowable set to the last element
        input:
            x (any type): xxx
            allowable_set (list): list of allowable type
        output: 
        '''
        
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))


def parse_smiles_to_graph(smiles, node_feature=['C', 'N', 'O', 'F'], \
        remove_aromatic=True, max_atom=None, self_loop=True, \
        print_faults=False, \
        add_node_features=['charge', 'valence', 'in_ring', \
        'is_aromatic', 'num_hydrogen']):
    
    '''
    function for parsing smiles to graph representation
    input:
        smiles (str): SMILES representation
        node_feature (list): node feature list
        edge_feature (list): edge feature list
        remove_aromatic (bool): if True, remove aromaticity
        max_atom (int, None): if None padding not applied
        self_loop (bool): if True, the diagonal element of adj matrix is 1
        print_faults (bool): if True, print faults
        add_node_features (list): lists of additional node features to add
    output:
        h (torch.Tensor): node feature tensor [N node_feature]
        e (torch.Tensor): edge feature tensor [N N edge_feature]
        adj (torch.Tensor): adjacency tensor [N N]
            or
        None: failed to parse smiles
    '''
    
    try:
        # get mol
        mol = Chem.MolFromSmiles(smiles)
        
        # initialize
        edge_feature = [1.0, 2.0, 3.0, 1.5]
        if remove_aromatic:
            Chem.Kekulize(mol, clearAromaticFlags=True)
            edge_feature = edge_feature[:3]
        if max_atom == None:
            max_atom = mol.GetNumAtoms()
        num_atom = mol.GetNumAtoms()
        
        # sizing
        node_len, edge_len = 0, 0
        if True:
            node_feature = node_feature + ['ELSE']
            edge_feature = edge_feature + ['ELSE']
        charge_c_to_i = [-3, -2, -1, 0, 1, 2, 3, 'ELSE']
        valence_c_to_i = [0, 1, 2, 3, 4, 5, 6, 'ELSE']
        num_hydrogen_c_to_i = [0, 1, 2, 3, 4, 'ELSE']
        node_len += len(node_feature)
        edge_len += len(edge_feature)
        if 'charge' in add_node_features:
            node_len += len(charge_c_to_i)
        if 'valence' in add_node_features:
            node_len += len(valence_c_to_i)
        if 'in_ring' in add_node_features:
            node_len += 1
        if 'is_aromatic' in add_node_features:
            node_len += 1
        if 'num_hydrogen' in add_node_features:
            node_len += len(num_hydrogen_c_to_i)

        h = torch.zeros(max_atom, node_len)
        e = torch.zeros(max_atom, max_atom, edge_len)
        adj = torch.zeros(max_atom, max_atom)
        
        # construct
        for idx1 in range(num_atom):
            
            # construct node feature
            atom = mol.GetAtomWithIdx(idx1)
            atom_feature = one_of_k_encoding(atom.GetSymbol(), node_feature)
            if 'charge' in add_node_features:
                atom_feature += one_of_k_encoding(atom.GetFormalCharge(), \
                        charge_c_to_i)
            if 'valence' in add_node_features:
                atom_feature += one_of_k_encoding(atom.GetExplicitValence(), \
                        valence_c_to_i)
            if 'in_ring' in add_node_features:
                atom_feature += [atom.IsInRing()]
            if 'is_aromatic' in add_node_features:
                atom_feature += [atom.GetIsAromatic()]
            if 'num_hydrogen' in add_node_features:
                atom_feature += one_of_k_encoding(atom.GetTotalNumHs(), \
                        num_hydrogen_c_to_i)
            atom_feature = torch.tensor(np.array(atom_feature)).float()
            h[idx1, :] = atom_feature
            
            for idx2 in range(num_atom):
                
                # construct edge and adj
                if idx1 == idx2:
                    if self_loop:
                        adj[idx1, idx2] = 1.0
                    continue
                bond = mol.GetBondBetweenAtoms(idx1, idx2)
                if not bond == None:
                    adj[idx1, idx2] = 1.0
                    bond_type = bond.GetBondTypeAsDouble()
                    edge_one_hot = one_of_k_encoding(bond_type, edge_feature)
                    edge_one_hot = torch.tensor(edge_one_hot).float()
                    e[idx1, idx2, :] = edge_one_hot
                    e[idx2, idx1, :] = edge_one_hot
        
        return h, e, adj
    
    except:
        if print_faults:
            print('parsing smiles fault: ', smiles)
        return None

def get_smiles_list(fn):
    f = open(fn, 'r')
    smiles_list = []
    for line in f.readlines():
        smiles_list.append(line.strip())
    return smiles_list







