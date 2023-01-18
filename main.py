import random
import networkx as nx
import torch
from torch_geometric.loader import DataLoader

from models import GCN, RniGCN
from wl import check_orbits_against_wl
from datasets import nx_molecule_dataset, orbit_molecule_dataset, pyg_dataset_from_nx

log_interval = 100

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
train_dataloader = DataLoader(orbit_mutag_dataset[0:int(len(orbit_mutag_dataset) * 0.8)], batch_size=10)
test_dataloader = DataLoader(orbit_mutag_dataset[int(len(orbit_mutag_dataset) * 0.8):], batch_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_node_features=7, num_classes=2, gcn_layers=4).to(device)
# model = RniGCN(num_node_features=7, num_classes=2, gcn_layers=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# train
model.train()
for epoch in range(3000):
    epoch_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss

    if (epoch + 1) % log_interval == 0:
        print(epoch, epoch_loss)


# TODO: test in a way that makes it not matter which node in the orbit gets the target

# print('\n--- MUTAG orbits ---')
# check_orbits_against_wl(mutag_nx)

# print('\n--- ENZYMES orbits ---')
# check_orbits_against_wl(enzymes_nx, max_graph_size_to_check=66)  # can do 66 in <30 seconds

# print('\n--- PROTEINS orbits ---')
# check_orbits_against_wl(proteins_nx, max_graph_size_to_check=60)  # can do 60 in <10 seconds
