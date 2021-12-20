# major = negative = 0, minor = positive = 1

save_dir = './exp_8_final/' # './exp_x'

# all fixed
# 2021.12.19 11:58

testing_mode = False

# parse
oversampling_rate = 20
val_ratio = 0.2
max_atom = 60

node_feature = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I'] # non-metal
# additional_properties = ['charge', 'valence', 'in_ring', 'is_aromatic', 'num_hydrogen']
additional_properties = ['charge', 'valence', 'num_hydrogen'] # fundametal props
# additional_properties = ['num_hydrogen']

# model
hid_dim = 128
n_layer = 4
# res_connection = False
use_dropout = True

# train
batch_size = 64
learning_rate = 1e-5
n_epoch = 400

