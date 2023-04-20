import random
import argparse
import networkx as nx
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT, GCN
import wandb

from losses import OrbitSortingCrossEntropyLoss, CrossEntropyLossWrapper
from models import RniGCN, UniqueIdGCN, UniqueIdDeepSetsGCN, OrbitIndivGCN, MaxOrbitGCN
from plotting import plot_labeled_graph
from testing import model_accuracy
from wl import check_orbits_against_wl, compute_wl_orbits
from datasets import nx_molecule_dataset, orbit_molecule_dataset, pyg_dataset_from_nx, nx_from_torch_dataset, \
    combined_bioisostere_dataset, molecule_dataset_orbit_count, alchemy_max_orbit_dataset, \
    pyg_max_orbit_dataset_from_nx, MaxOrbitGCNTransform

parser = argparse.ArgumentParser()

# logging options
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--use_wandb', type=int, default=0)

# model
parser.add_argument('--model', type=str, default='max_orbit_gcn',
                    choices=['gcn', 'gat', 'unique_id_gcn', 'rni_gcn', 'orbit_indiv_gcn', 'max_orbit_gcn'])
parser.add_argument('--gnn_layers', type=int, default=4)
parser.add_argument('--gnn_hidden_size', type=int, default=40)
parser.add_argument('--rni_channels', type=int, default=10)

# dataset
parser.add_argument('--train_on_entire_dataset', type=int, default=1)
# filter out non-equivariant examples from the bioisostere dataset
parser.add_argument('--bioisostere_only_equivariant', type=int, default=0)
parser.add_argument('--dataset', type=str, default='bioisostere',
                    choices=['bioisostere', 'mutag', 'alchemy', 'zinc'])
# use with alchemy to create a max_orbit dataset, 0 means don't use max_orbit
parser.add_argument('--max_orbit', type=int, default=2)

# training
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--changed_node_loss_weight', type=float, default=1)
parser.add_argument('--loss', type=str, default='cross_entropy',
                    choices=['cross_entropy', 'orbit_sorting_cross_entropy'])

# misc
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use_cpu', type=int, default=0)

args = parser.parse_args()

# init logging
if args.use_wandb:
    wandb.init(project="orbit-gnn")

# fix RNG
if args.seed == 0:  # sample seed at random
    args.seed = random.randint(1, 10000)
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cpu:
    device = 'cpu'

# G = nx.Graph()
#
# G.add_nodes_from([
#     (0, {'x': (1.0, 1.0), 'y': 1}),
#     (1, {'x': (2.0, 1.0), 'y': 1}),
# ])
#
# for i in range(2, 7):
#     G.add_node(i, **{'x': (1.0, 1.0), 'y': 1})
#
# G.add_edges_from([
#     (0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (4, 6)
# ])

dataset = None
if args.dataset == 'bioisostere':
    print('Loading bioisostere dataset')
    bioisostere_data_list_inputs = torch.load('custom-datasets/chembl_bioisostere_dataset_inputs.pt')
    bioisostere_data_list_targets = torch.load('custom-datasets/chembl_bioisostere_dataset_targets.pt')
    bioisostere_data_list_combined = combined_bioisostere_dataset(
        bioisostere_data_list_inputs, bioisostere_data_list_targets,
        only_equivariant=args.bioisostere_only_equivariant)
    torch.save(bioisostere_data_list_combined, 'custom-datasets/bioisostere_data_list_combined.pt')
    dataset = bioisostere_data_list_combined
elif args.dataset == 'mutag':
    mutag_nx, num_node_classes = nx_molecule_dataset('MUTAG')
    print('MUTAG orbit size counts:', molecule_dataset_orbit_count(mutag_nx))
    # random.shuffle(mutag_nx)  # shuffle dataset
    orbit_mutag_nx = orbit_molecule_dataset(mutag_nx, num_features=7)
    orbit_mutag_dataset = pyg_dataset_from_nx(orbit_mutag_nx)
    dataset = orbit_mutag_dataset
elif args.dataset == 'alchemy':
    alchemy_nx, num_node_classes = nx_molecule_dataset('alchemy_full')
    if args.max_orbit >= 2:
        orbit_alchemy_nx = alchemy_max_orbit_dataset(
            dataset=alchemy_nx,
            num_node_classes=num_node_classes,
            extended_dataset_size=1000,  # TODO: make arg
            max_orbit=args.max_orbit
        )
        orbit_alchemy_pyg = pyg_max_orbit_dataset_from_nx(orbit_alchemy_nx)
        dataset = orbit_alchemy_pyg
    else:
        raise Exception('Alchemy currently only supported with args.max_orbit >= 2')
elif args.dataset == 'zinc':
    zinc_nx, num_node_classes = nx_molecule_dataset('ZINC_full')
    print('zinc orbit size counts:', molecule_dataset_orbit_count(zinc_nx))
    raise Exception('Zinc currently not supported for training')
else:
    raise Exception('Dataset "', args.dataset, '" not recognized')

# set up loss
if args.loss == 'cross_entropy':
    criterion = CrossEntropyLossWrapper()
elif args.loss == 'orbit_sorting_cross_entropy':
    criterion = OrbitSortingCrossEntropyLoss()
else:
    raise Exception('Loss "', args.loss, '" not recognized')

# set number of input and output channels
in_channels = dataset[0].x.size()[1]
# cannot get out_channels in the same way as in_channels, since targets are not one-hot
out_channels = in_channels  # same number of classes by default
if args.dataset == 'bioisostere':
    # bioisostere dataset has an output class for 'no change'
    out_channels += 1

# add transformed targets to dataset if using max_orbit_gcn
if args.model == 'max_orbit_gcn':
    transform = MaxOrbitGCNTransform(args.max_orbit, out_channels)
    transform.transform_dataset(dataset)
    # max orbit transformation has an extra output class for 'no change from default'
    out_channels += 1

# set up train / test split on dataset
train_dataset = dataset[0:int(len(dataset) * 0.8)]
if args.train_on_entire_dataset:
    train_dataset = dataset
test_dataset = dataset[int(len(dataset) * 0.8):]

print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))

# set up model
if args.model == 'gat':
    model = GAT(
        in_channels=in_channels,
        hidden_channels=args.gnn_hidden_size,
        num_layers=args.gnn_layers,
        out_channels=out_channels,
    )
elif args.model == 'gcn':
    model = GCN(
        in_channels=in_channels,
        hidden_channels=args.gnn_hidden_size,
        num_layers=args.gnn_layers,
        out_channels=out_channels,
    )
elif args.model == 'unique_id_gcn':
    model = UniqueIdGCN(
        in_channels=in_channels,
        hidden_channels=args.gnn_hidden_size,
        num_layers=args.gnn_layers,
        out_channels=out_channels
    )
elif args.model == 'rni_gcn':
    model = RniGCN(
        in_channels=in_channels,
        hidden_channels=args.gnn_hidden_size,
        num_layers=args.gnn_layers,
        out_channels=out_channels,
        rni_channels=args.rni_channels,
    )
elif args.model == 'orbit_indiv_gcn':
    model = OrbitIndivGCN(
        in_channels=in_channels,
        hidden_channels=args.gnn_hidden_size,
        num_layers=args.gnn_layers,
        out_channels=out_channels,
    )
elif args.model == 'max_orbit_gcn':
    model = MaxOrbitGCN(
        in_channels=in_channels,
        hidden_channels=args.gnn_hidden_size,
        num_layers=args.gnn_layers,
        out_channels=out_channels,
        max_orbit=args.max_orbit,
    )
else:
    raise Exception('Model "', args.model, '" not recognized')
# model = RniGCN(num_node_features=7, num_classes=2, gcn_layers=4, noise_dims=3).to(device)
# model = UniqueIdDeepSetsGCN(num_node_features=7, num_classes=2, gcn_layers=4).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

# train
print('Training model')
model.train()
for epoch in range(args.n_epochs):
    model.training = True
    epoch_loss = 0
    for data in train_dataset:
        optimizer.zero_grad()
        data = data.to(device)  # TODO: optimize code for GPU

        out = model(data.x, data.edge_index, orbits=data.orbits)
        targets = data.transformed_y if args.model == 'max_orbit_gcn' else data.y
        loss = criterion(out, targets, data.non_equivariant_orbits)

        # # custom weighting of loss for nodes that change
        # changed_node_index = -1
        # for node, node_feature in enumerate(data.y):
        #     if node_feature[-1] != 1:  # final bit != 1 means node changed
        #         changed_node_index = node
        #         break
        # if changed_node_index != -1:
        #     extra_loss_fn = torch.nn.MSELoss()
        #     loss += extra_loss_fn(out[changed_node_index], data.y[changed_node_index]) * args.changed_node_loss_weight

        # out = model(data)  # [N, C], where N = nodes and C = target classes
        #
        # # data.y has size [N, P], where N is the number of nodes in the graph,
        # # and P is the number of permutations of acceptable answers
        # # For now, P is exactly the size of the orbit targeted for removal (i.e. permutations on [0, 0, ..., 0, 1])
        #
        # possible_targets = torch.swapaxes(data.y, 0, 1)  # [P, N]
        # possible_losses = torch.zeros(possible_targets.size()[0])  # [P]
        # for i, possible_target in enumerate(possible_targets):  # possible_target is [N]
        #     possible_losses[i] = criterion(out, possible_target)
        # loss = torch.min(possible_losses)

        loss.backward()
        optimizer.step()
        epoch_loss += loss

    if (epoch + 1) % args.log_interval == 0:
        # compute train accuracy
        model.training = False
        node_accuracy, orbit_accuracy, graph_accuracy = model_accuracy(train_dataset, model, device)
        print('Epoch:', epoch + 1, '| Eval on training dataset | Epoch loss:', epoch_loss.item(), '| Node accuracy:',
              node_accuracy, '| Orbit accuracy:', orbit_accuracy, '| Graph accuracy:', graph_accuracy)

        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'train_node_accuracy': node_accuracy,
                'train_orbit_accuracy': orbit_accuracy,
                'train_graph_accuracy': graph_accuracy,
            })

# TODO: test in a way that makes it not matter which node in the orbit gets the target
# (just compare the set intersections for each orbit)

# print('\n--- MUTAG orbits ---')
# check_orbits_against_wl(mutag_nx)

# print('\n--- ENZYMES orbits ---')
# check_orbits_against_wl(enzymes_nx, max_graph_size_to_check=66)  # can do 66 in <30 seconds

# print('\n--- PROTEINS orbits ---')
# check_orbits_against_wl(proteins_nx, max_graph_size_to_check=60)  # can do 60 in <10 seconds
