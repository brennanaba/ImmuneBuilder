import torch
from einops import rearrange
from ImmuneBuilder.rigids import Rigid, Rot, rigid_body_identity, vec_from_tensor, global_frames_from_bb_frame_and_torsion_angles, all_atoms_from_global_reference_frames

class InvariantPointAttention(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, heads=12, head_dim=16, n_query_points=4, n_value_points=8, **kwargs):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.n_query_points = n_query_points

        node_scalar_attention_inner_dim = heads * head_dim
        node_vector_attention_inner_dim = 3 * n_query_points * heads
        node_vector_attention_value_dim = 3 * n_value_points * heads
        after_final_cat_dim = heads * edge_dim + heads * head_dim + heads * n_value_points * 4

        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.)) - 1.)
        self.point_weight = torch.nn.Parameter(point_weight_init_value)

        self.to_scalar_qkv = torch.nn.Linear(node_dim, 3 * node_scalar_attention_inner_dim, bias=False)
        self.to_vector_qk = torch.nn.Linear(node_dim, 2 * node_vector_attention_inner_dim, bias=False)
        self.to_vector_v = torch.nn.Linear(node_dim, node_vector_attention_value_dim, bias=False)
        self.to_scalar_edge_attention_bias = torch.nn.Linear(edge_dim, heads, bias=False)
        self.final_linear = torch.nn.Linear(after_final_cat_dim, node_dim)

        with torch.no_grad():
            self.final_linear.weight.fill_(0.0)
            self.final_linear.bias.fill_(0.0)

    def forward(self, node_features, edge_features, rigid):
        # Classic attention on nodes
        scalar_qkv = self.to_scalar_qkv(node_features).chunk(3, dim=-1)
        scalar_q, scalar_k, scalar_v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=self.heads), scalar_qkv)
        node_scalar = torch.einsum('h i d, h j d -> h i j', scalar_q, scalar_k) * self.head_dim ** (-1 / 2)

        # Linear bias on edges
        edge_bias = rearrange(self.to_scalar_edge_attention_bias(edge_features), 'i j h -> h i j')

        # Reference frame attention
        wc = (2 / self.n_query_points) ** (1 / 2) / 6
        vector_qk = self.to_vector_qk(node_features).chunk(2, dim=-1)
        vector_q, vector_k = map(lambda x: vec_from_tensor(rearrange(x, 'n (h p d) -> h n p d', h=self.heads, d=3)),
                                 vector_qk)
        rigid_ = rigid.unsqueeze(0).unsqueeze(-1)  # add head and point dimension to rigids

        global_vector_k = rigid_ @ vector_k
        global_vector_q = rigid_ @ vector_q
        global_frame_distance = wc * global_vector_q.unsqueeze(-2).dist(global_vector_k.unsqueeze(-3)).sum(
            -1) * rearrange(self.point_weight, "h -> h () ()")

        # Combining attentions
        attention_matrix = (3 ** (-1 / 2) * (node_scalar + edge_bias - global_frame_distance)).softmax(-1)

        # Obtaining outputs
        edge_output = (rearrange(attention_matrix, 'h i j -> i h () j') * rearrange(edge_features,
                                                                                    'i j d -> i () d j')).sum(-1)
        scalar_node_output = torch.einsum('h i j, h j d -> i h d', attention_matrix, scalar_v)

        vector_v = vec_from_tensor(
            rearrange(self.to_vector_v(node_features), 'n (h p d) -> h n p d', h=self.heads, d=3))
        global_vector_v = rigid_ @ vector_v
        attended_global_vector_v = global_vector_v.map(
            lambda x: torch.einsum('h i j, h j p -> h i p', attention_matrix, x))
        vector_node_output = rigid_.inv() @ attended_global_vector_v
        vector_node_output = torch.stack(
            [vector_node_output.norm(), vector_node_output.x, vector_node_output.y, vector_node_output.z], dim=-1)

        # Concatenate along heads and points
        edge_output = rearrange(edge_output, 'n h d -> n (h d)')
        scalar_node_output = rearrange(scalar_node_output, 'n h d -> n (h d)')
        vector_node_output = rearrange(vector_node_output, 'h n p d -> n (h p d)')

        combined = torch.cat([edge_output, scalar_node_output, vector_node_output], dim=-1)

        return node_features + self.final_linear(combined)


class BackboneUpdate(torch.nn.Module):
    def __init__(self, node_dim):
        super().__init__()

        self.to_correction = torch.nn.Linear(node_dim, 6)

    def forward(self, node_features, update_mask=None):
        # Predict quaternions and translation vector
        rot, t = self.to_correction(node_features).chunk(2, dim=-1)

        # I may not want to update all residues
        if update_mask is not None:
            rot = update_mask[:, None] * rot
            t = update_mask[:, None] * t

        # Normalize quaternions
        norm = (1 + rot.pow(2).sum(-1, keepdim=True)).pow(1 / 2)
        b, c, d = (rot / norm).chunk(3, dim=-1)
        a = 1 / norm
        a, b, c, d = a.squeeze(-1), b.squeeze(-1), c.squeeze(-1), d.squeeze(-1)

        # Make rotation matrix from quaternions
        R = Rot(
            (a ** 2 + b ** 2 - c ** 2 - d ** 2), (2 * b * c - 2 * a * d), (2 * b * d + 2 * a * c),
            (2 * b * c + 2 * a * d), (a ** 2 - b ** 2 + c ** 2 - d ** 2), (2 * c * d - 2 * a * b),
            (2 * b * d - 2 * a * c), (2 * c * d + 2 * a * b), (a ** 2 - b ** 2 - c ** 2 + d ** 2)
        )

        return Rigid(vec_from_tensor(t), R)


class TorsionAngles(torch.nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.residual1 = torch.nn.Sequential(
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim)
        )

        self.residual2 = torch.nn.Sequential(
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim)
        )

        self.final_pred = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 10)
        )

        with torch.no_grad():
            self.residual1[-1].weight.fill_(0.0)
            self.residual2[-1].weight.fill_(0.0)
            self.residual1[-1].bias.fill_(0.0)
            self.residual2[-1].bias.fill_(0.0)

    def forward(self, node_features, s_i):
        full_feat = torch.cat([node_features, s_i], axis=-1)

        full_feat = full_feat + self.residual1(full_feat)
        full_feat = full_feat + self.residual2(full_feat)
        torsions = rearrange(self.final_pred(full_feat), "i (t d) -> i t d", d=2)
        norm = torch.norm(torsions, dim=-1, keepdim=True)

        return torsions / norm, norm


class StructureUpdate(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, dropout=0.0, **kwargs):
        super().__init__()
        self.IPA = InvariantPointAttention(node_dim, edge_dim, **kwargs)
        self.norm1 = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(node_dim)
        )
        self.norm2 = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(node_dim)
        )
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(node_dim, 2 * node_dim),  
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, node_dim)
        )

        self.torsion_angles = TorsionAngles(node_dim)
        self.backbone_update = BackboneUpdate(node_dim)

        with torch.no_grad():
            self.residual[-1].weight.fill_(0.0)
            self.residual[-1].bias.fill_(0.0)

    def forward(self, node_features, edge_features, rigid_pred, update_mask=None):
        s_i = self.IPA(node_features, edge_features, rigid_pred)
        s_i = self.norm1(s_i)
        s_i = s_i + self.residual(s_i)
        s_i = self.norm2(s_i)
        rigid_new = rigid_pred @ self.backbone_update(s_i, update_mask)

        return s_i, rigid_new


class StructureModule(torch.nn.Module):
    def __init__(self, node_dim=23, n_layers=8, rel_pos_dim=64, embed_dim=128, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.rel_pos_dim = rel_pos_dim
        self.node_embed = torch.nn.Linear(node_dim, embed_dim)
        self.edge_embed = torch.nn.Linear(2 * rel_pos_dim + 1, embed_dim - 1)

        self.layers = torch.nn.ModuleList(
            [StructureUpdate(node_dim=embed_dim,
                             edge_dim=embed_dim,
                             propagate_rotation_gradient=(i == n_layers - 1),
                             **kwargs)
             for i in range(n_layers)])

    def forward(self, node_features, sequence):
        rigid_in = rigid_body_identity(len(sequence)).to(node_features.device)
        relative_positions = (torch.arange(node_features.shape[-2])[None] -
                              torch.arange(node_features.shape[-2])[:, None])
        relative_positions = relative_positions.clamp(min=-self.rel_pos_dim, max=self.rel_pos_dim) + self.rel_pos_dim

        rel_pos_embeddings = torch.nn.functional.one_hot(relative_positions, num_classes=2 * self.rel_pos_dim + 1)
        rel_pos_embeddings = rel_pos_embeddings.to(dtype=node_features.dtype, device=node_features.device)
        rel_pos_embeddings = self.edge_embed(rel_pos_embeddings)

        new_node_features = self.node_embed(node_features)

        for layer in self.layers:
            edge_features = torch.cat(
                [rigid_in.origin.unsqueeze(-1).dist(rigid_in.origin).unsqueeze(-1), rel_pos_embeddings], dim=-1)
            new_node_features, rigid_in = layer(new_node_features, edge_features, rigid_in)

        torsions, _ = self.layers[-1].torsion_angles(self.node_embed(node_features), new_node_features)

        all_reference_frames = global_frames_from_bb_frame_and_torsion_angles(rigid_in, torsions, sequence)
        all_atoms = all_atoms_from_global_reference_frames(all_reference_frames, sequence)

        # Remove atoms of side chains with outrageous clashes
        ds = torch.linalg.norm(all_atoms[None,:,None] - all_atoms[:,None,:,None], axis = -1)
        ds[torch.isnan(ds) | (ds==0.0)] = 10
        min_ds = ds.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
        all_atoms[min_ds < 0.2, 5:, :] = float("Nan")

        return  all_atoms, new_node_features
