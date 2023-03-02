import random
import networkx as nx
import torch
from torch_geometric.loader import DataLoader

from models import GCN, RniGCN, UniqueIdGCN, UniqueIdDeepSetsGCN
from plotting import plot_labeled_graph
from wl import check_orbits_against_wl, compute_wl_orbits
from datasets import nx_molecule_dataset, orbit_molecule_dataset, pyg_dataset_from_nx, nx_from_torch_dataset, \
    combined_bioisostere_dataset

log_interval = 1

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
train_dataset = orbit_mutag_dataset[0:int(len(orbit_mutag_dataset) * 0.8)]
test_dataset = orbit_mutag_dataset[int(len(orbit_mutag_dataset) * 0.8):]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCN(num_node_features=7, num_classes=2, gcn_layers=4).to(device)
# model = RniGCN(num_node_features=7, num_classes=2, gcn_layers=4, noise_dims=3).to(device)
model = UniqueIdDeepSetsGCN(num_node_features=7, num_classes=2, gcn_layers=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# train
print('Training model')
model.train()
for epoch in range(3000):
    epoch_loss = 0
    for data in train_dataset:
        optimizer.zero_grad()

        out = model(data)  # [N, C], where N = nodes and C = target classes

        # data.y has size [N, P], where N is the number of nodes in the graph,
        # and P is the number of permutations of acceptable answers
        # For now, P is exactly the size of the orbit targeted for removal (i.e. permutations on [0, 0, ..., 0, 1])

        possible_targets = torch.swapaxes(data.y, 0, 1)  # [P, N]
        possible_losses = torch.zeros(possible_targets.size()[0])  # [P]
        for i, possible_target in enumerate(possible_targets):  # possible_target is [N]
            possible_losses[i] = criterion(out, possible_target)
        loss = torch.min(possible_losses)

        loss.backward()
        optimizer.step()
        epoch_loss += loss

    if (epoch + 1) % log_interval == 0:
        print(epoch + 1, epoch_loss)


# TODO: test in a way that makes it not matter which node in the orbit gets the target

# print('\n--- MUTAG orbits ---')
# check_orbits_against_wl(mutag_nx)

# print('\n--- ENZYMES orbits ---')
# check_orbits_against_wl(enzymes_nx, max_graph_size_to_check=66)  # can do 66 in <30 seconds

# print('\n--- PROTEINS orbits ---')
# check_orbits_against_wl(proteins_nx, max_graph_size_to_check=60)  # can do 60 in <10 seconds
