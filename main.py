import random
import argparse
import networkx as nx
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import wandb

from models import GCN, RniGCN, UniqueIdGCN, UniqueIdDeepSetsGCN
from plotting import plot_labeled_graph
from wl import check_orbits_against_wl, compute_wl_orbits
from datasets import nx_molecule_dataset, orbit_molecule_dataset, pyg_dataset_from_nx, nx_from_torch_dataset, \
    combined_bioisostere_dataset


parser = argparse.ArgumentParser()

# logging options
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--use_wandb', type=int, default=0)

# model params
parser.add_argument('--gcn_layers', type=int, default=4)
parser.add_argument('--gcn_hidden_size', type=int, default=64)

# training
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--n_epochs', type=int, default=2000)

# misc
parser.add_argument('--seed', type=int, default=0)

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

print('Loading bioisostere dataset')
bioisostere_data_list_inputs = torch.load('custom-datasets/chembl_bioisostere_dataset_inputs.pt')
bioisostere_data_list_targets = torch.load('custom-datasets/chembl_bioisostere_dataset_targets.pt')
bioisostere_data_list_combined = combined_bioisostere_dataset(
    bioisostere_data_list_inputs, bioisostere_data_list_targets)
torch.save(bioisostere_data_list_combined, 'custom-datasets/bioisostere_data_list_combined.pt')

mutag_nx = nx_molecule_dataset('MUTAG')
# random.shuffle(mutag_nx)  # shuffle dataset
enzymes_nx = nx_molecule_dataset('ENZYMES')
proteins_nx = nx_molecule_dataset('PROTEINS')

orbit_mutag_nx = orbit_molecule_dataset(mutag_nx, num_features=7)
orbit_mutag_dataset = pyg_dataset_from_nx(orbit_mutag_nx)
# pyg_graph = orbit_mutag_dataset[0]
# print('graph:\n', pyg_graph, '\n\n')
# print('x:\n', pyg_graph.x, '\n\n')
# print('y:\n', pyg_graph.y, '\n\n')
# print('edge index:\n', pyg_graph.edge_index, '\n\n')

# set up model
criterion = torch.nn.CrossEntropyLoss()
# train_dataset = orbit_mutag_dataset[0:int(len(orbit_mutag_dataset) * 0.8)]
# test_dataset = orbit_mutag_dataset[int(len(orbit_mutag_dataset) * 0.8):]
train_dataset = bioisostere_data_list_combined[0:int(len(bioisostere_data_list_combined) * 0.8)]
test_dataset = bioisostere_data_list_combined[int(len(bioisostere_data_list_combined) * 0.8):]
model = GCN(
    num_node_features=train_dataset[0].x.size()[1],
    num_classes=train_dataset[0].y.size()[1],
    gcn_layers=args.gcn_layers,
    hidden_size=args.gcn_hidden_size,
).to(device)
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
        data = data.to(device)

        out = model(data)
        loss = criterion(out, data.y)

        # custom weighting of loss for nodes that change
        changed_node_index = -1
        for node, node_feature in enumerate(data.y):
            if node_feature[-1] != 1:  # final bit != 1 means node changed
                changed_node_index = node
                break
        if changed_node_index != -1:
            extra_loss_fn = torch.nn.MSELoss()
            loss += extra_loss_fn(out[changed_node_index], data.y[changed_node_index]) * 10

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

        total_graph_accuracy = 0
        total_node_accuracy = 0

        # train accuracy
        model.training = False
        for data in train_dataset:
            data = data.to(device)
            out = model(data)
            # gets 0.9375 node accuracy if just returning input (out = data.x)
            predictions = torch.argmax(out, dim=1)  # no need to softmax, since it's monotonic
            ground_truth = torch.argmax(data.y, dim=1)
            node_accuracy = torch.sum(predictions == ground_truth) / predictions.size()[0]
            graph_accuracy = 0 if node_accuracy < 0.99 else 1
            total_node_accuracy += node_accuracy
            total_graph_accuracy += graph_accuracy

        # print(model(train_dataset[7]))
        print(epoch + 1, epoch_loss, total_node_accuracy / len(train_dataset), total_graph_accuracy / len(train_dataset))
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'node_accuracy': total_node_accuracy / len(train_dataset),
                'graph_accuracy': total_graph_accuracy / len(train_dataset)
            })

# TODO: test in a way that makes it not matter which node in the orbit gets the target

# print('\n--- MUTAG orbits ---')
# check_orbits_against_wl(mutag_nx)

# print('\n--- ENZYMES orbits ---')
# check_orbits_against_wl(enzymes_nx, max_graph_size_to_check=66)  # can do 66 in <30 seconds

# print('\n--- PROTEINS orbits ---')
# check_orbits_against_wl(proteins_nx, max_graph_size_to_check=60)  # can do 60 in <10 seconds
