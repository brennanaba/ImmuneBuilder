from ImmuneBuilder.constants import res_to_num, atom_types, residue_atoms, restype_1to3, restypes

import numpy as np
import torch
import requests
import os


def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb+') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return filename


def get_one_hot(targets, nb_classes=21):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def get_encoding(sequence_dict, chain_ids="HL"):
    
    encodings = []

    for j,chain in enumerate(chain_ids):
        seq = sequence_dict[chain]
        one_hot_amino = get_one_hot(np.array([res_to_num(x) for x in seq]))
        one_hot_region = get_one_hot(j * np.ones(len(seq), dtype=int), 2)
        encoding = np.concatenate([one_hot_amino, one_hot_region], axis=-1)
        encodings.append(encoding)

    return np.concatenate(encodings, axis = 0)


def find_alignment_transform(traces):
    centers = traces.mean(-2, keepdim=True)
    traces = traces - centers

    p1, p2 = traces[0], traces[1:]

    C = torch.einsum("i j k, j l -> i k l", p2, p1)
    V, _, W = torch.linalg.svd(C)
    U = torch.matmul(V, W)
    U = torch.matmul(torch.stack([torch.ones(len(p2), device=U.device),torch.ones(len(p2), device=U.device),torch.linalg.det(U)], dim=1)[:,:,None] * V,  W)

    return torch.cat([torch.eye(3, device=U.device)[None], U]), centers
    

def to_pdb(numbered_sequences, all_atoms, chain_ids = "HL"):
    atom_index = 0
    pdb_lines = []
    record_type = "ATOM"
    seq = numbered_sequences[chain_ids[0]] + numbered_sequences[chain_ids[1]]
    chain_index = [0]*len(numbered_sequences[chain_ids[0]]) + [1]*len(numbered_sequences[chain_ids[1]])
    chain_id = chain_ids[0]

    for i, amino in enumerate(seq):
        for atom in atom_types:
            if atom in residue_atoms[amino[1]]:
                j = residue_atoms[amino[1]].index(atom)
                pos = all_atoms[i, j]
                if pos.mean() != pos.mean():
                    continue
                name = f' {atom}'
                alt_loc = ''
                res_name_3 = restype_1to3[amino[1]]
                if chain_id != chain_ids[chain_index[i]]:
                    chain_id = chain_ids[chain_index[i]]
                occupancy = 1.00
                b_factor = 0.00
                element = atom[0]
                charge = ''
                # PDB is a columnar format, every space matters here!
                atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                             f'{res_name_3:>3} {chain_id:>1}'
                             f'{(amino[0][0]):>4}{amino[0][1]:>1}   '
                             f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                             f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                             f'{element:>2}{charge:>2}')
                pdb_lines.append(atom_line)
                atom_index += 1

    return "\n".join(pdb_lines)


def sequence_dict_from_fasta(fasta_file):
    out = {}

    with open(fasta_file) as file:
        txt = file.read().split()

    for i in range(len(txt)-1):
        if ">" in txt[i]:
            chain_id = txt[i].split(">")[1]
        else:
            continue

        if all(c in restypes for c in txt[i+1]):
            out[chain_id] = txt[i+1]
    
    return out


def add_errors_as_bfactors(filename, errors, header=[]):

    with open(filename) as file:
        txt = file.readlines()

    new_txt = header.copy()
    residue_index = -1
    position = "  "

    for line in txt:
        if line[:4] == "ATOM":
            current_res = line[22:27]
            if current_res != position:
                position = current_res
                residue_index += 1
            line = line.replace("  0.00  ",f"{errors[residue_index]:>6.2f}  ")
        elif "REMARK   1 CREATED WITH OPENMM" in line:
            line = line.replace(" 1 CREATED WITH OPENMM", "STRUCTURE REFINED USING OPENMM")
            line = line[:-1] + (81-len(line))*" " + "\n"
        new_txt.append(line)

    with open(filename, "w+") as file:
        file.writelines(new_txt)


def are_weights_ready(weights_path):
    if not os.path.exists(weights_path):
        return False
    with open(weights_path, "rb") as f:
        filestart = str(f.readline())
    return filestart != "b'EMPTY'"
