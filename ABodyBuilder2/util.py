import torch
from AbodyBuilder2.constants import rigid_group_atom_positions2, chi2_centers, chi3_centers, chi4_centers, rel_pos, \
    residue_atoms_mask
from AbodyBuilder2.rigids import Rigid, Rot, vec_from_tensor


def stack_rigids(rigids, **kwargs):
    # Probably best to avoid using very much
    stacked_origin = Vector(torch.stack([rig.origin.x for rig in rigids], **kwargs),
                            torch.stack([rig.origin.y for rig in rigids], **kwargs),
                            torch.stack([rig.origin.z for rig in rigids], **kwargs))
    stacked_rot = Rot(
        torch.stack([rig.rot.xx for rig in rigids], **kwargs), torch.stack([rig.rot.xy for rig in rigids], **kwargs),
        torch.stack([rig.rot.xz for rig in rigids], **kwargs),
        torch.stack([rig.rot.yx for rig in rigids], **kwargs), torch.stack([rig.rot.yy for rig in rigids], **kwargs),
        torch.stack([rig.rot.yz for rig in rigids], **kwargs),
        torch.stack([rig.rot.zx for rig in rigids], **kwargs), torch.stack([rig.rot.zy for rig in rigids], **kwargs),
        torch.stack([rig.rot.zz for rig in rigids], **kwargs),
    )
    return Rigid(stacked_origin, stacked_rot)


def rotate_x_axis_to_new_vector(new_vector):
    # Extract coordinates
    c, b, a = new_vector[..., 0], new_vector[..., 1], new_vector[..., 2]

    # Normalize
    n = (c ** 2 + a ** 2 + b ** 2 + 1e-16) ** (1 / 2)
    a, b, c = a / n, b / n, -c / n

    # Set new origin
    new_origin = vec_from_tensor(torch.zeros_like(new_vector))

    # Rotate x-axis to point old origin to new one
    k = (1 - c) / (a ** 2 + b ** 2 + 1e-8)
    new_rot = Rot(-c, b, -a, b, 1 - k * b ** 2, a * b * k, a, -a * b * k, k * a ** 2 - 1)

    return Rigid(new_origin, new_rot)


def global_frames_from_bb_frame_and_torsion_angles(bb_frame, torsion_angles, seq):
    dev = bb_frame.origin.x.device

    # We start with psi
    psi_local_frame_origin = torch.tensor([rel_pos[x][2][1] for x in seq]).to(dev).pow(2).sum(-1).pow(1 / 2)
    psi_local_frame = rigid_transformation_from_torsion_angles(torsion_angles[:, 0], psi_local_frame_origin)
    psi_global_frame = bb_frame @ psi_local_frame

    # Now all the chis
    chi1_local_frame_origin = torch.tensor([rel_pos[x][3][1] for x in seq]).to(dev)
    chi1_local_frame = rotate_x_axis_to_new_vector(chi1_local_frame_origin) @ rigid_transformation_from_torsion_angles(
        torsion_angles[:, 1], chi1_local_frame_origin.pow(2).sum(-1).pow(1 / 2))
    chi1_global_frame = bb_frame @ chi1_local_frame

    chi2_local_frame_origin = torch.tensor([rigid_group_atom_positions2[x][chi2_centers[x]][1] for x in seq]).to(dev)
    chi2_local_frame = rotate_x_axis_to_new_vector(chi2_local_frame_origin) @ rigid_transformation_from_torsion_angles(
        torsion_angles[:, 2], chi2_local_frame_origin.pow(2).sum(-1).pow(1 / 2))
    chi2_global_frame = chi1_global_frame @ chi2_local_frame

    chi3_local_frame_origin = torch.tensor([rigid_group_atom_positions2[x][chi3_centers[x]][1] for x in seq]).to(dev)
    chi3_local_frame = rotate_x_axis_to_new_vector(chi3_local_frame_origin) @ rigid_transformation_from_torsion_angles(
        torsion_angles[:, 3], chi3_local_frame_origin.pow(2).sum(-1).pow(1 / 2))
    chi3_global_frame = chi2_global_frame @ chi3_local_frame

    chi4_local_frame_origin = torch.tensor([rigid_group_atom_positions2[x][chi4_centers[x]][1] for x in seq]).to(dev)
    chi4_local_frame = rotate_x_axis_to_new_vector(chi4_local_frame_origin) @ rigid_transformation_from_torsion_angles(
        torsion_angles[:, 4], chi4_local_frame_origin.pow(2).sum(-1).pow(1 / 2))
    chi4_global_frame = chi3_global_frame @ chi4_local_frame

    return stack_rigids(
        [bb_frame, psi_global_frame, chi1_global_frame, chi2_global_frame, chi3_global_frame, chi4_global_frame],
        dim=-1)


def all_atoms_from_global_reference_frames(global_reference_frames, seq):
    dev = global_reference_frames.origin.x.device

    all_atoms = torch.zeros((len(seq), 14, 3)).to(dev)
    for atom_pos in range(14):
        relative_positions = [rel_pos[x][atom_pos][1] for x in seq]
        local_reference_frame = [max(rel_pos[x][atom_pos][0] - 2, 0) for x in seq]
        local_reference_frame_mask = torch.tensor([[y == x for y in range(6)] for x in local_reference_frame]).to(dev)
        global_atom_vector = global_reference_frames[local_reference_frame_mask] @ vec_from_tensor(
            torch.tensor(relative_positions).to(dev))
        all_atoms[:, atom_pos] = torch.stack([global_atom_vector.x, global_atom_vector.y, global_atom_vector.z], dim=-1)

    all_atom_mask = torch.tensor([residue_atoms_mask[x] for x in seq]).to(dev)
    all_atoms[~all_atom_mask] = float("Nan")
    return all_atoms
