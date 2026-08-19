import torch
pt_path = r'\\140.112.16.234\d\Allen\Allen_Workspace\physicsnemo\examples\structural_mechanics\deforming_plate\04_preprocessed_pt\sample_00000.pt'
s = torch.load(pt_path, weights_only=False)
g = s[0]['graph']
if hasattr(g, 'node_type'):
    print('node_type unique values:', g.node_type.unique())
    print('node_type all same?', (g.node_type == g.node_type[0]).all().item())
else:
    print('no node_type attribute -- defaults to all zeros')
print('graph.x shape:', g.x.shape)
print('one-hot cols unique rows:', g.x[:, :3].unique(dim=0))
