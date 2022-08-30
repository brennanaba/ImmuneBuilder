import numpy as np
from ImmuneBuilder.constants import res_to_num, atom_types, residue_atoms, restype_1to3
from ImmuneBuilder.rigids import Rigid

def get_one_hot(targets, nb_classes=21):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])

def get_encoding(sequence_dict):
    
    encodings = []

    for j,chain in enumerate("HL"):
        seq = sequence_dict[chain]
        one_hot_amino = get_one_hot(np.array([res_to_num(x) for x in seq]))
        one_hot_region = get_one_hot(j * np.ones(len(seq), dtype=int), 2)
        encoding = np.concatenate([one_hot_amino, one_hot_region], axis=-1)
        encodings.append(encoding)

    return np.concatenate(encodings, axis = 0)
    

def to_pdb(sequence_dict, all_atoms, chain_ids = "HL"):
    atom_index = 0
    pdb_lines = []
    record_type = "ATOM"
    seq = sequence_dict["H"] + sequence_dict["L"]
    chain_index = [0]*len(sequence_dict["H"]) + [1]*len(sequence_dict["L"])
    chain_start, chain_id = 0, chain_ids[0]

    for i, amino in enumerate(seq):
        for atom in atom_types:
            if atom in residue_atoms[amino]:
                j = residue_atoms[amino].index(atom)
                pos = all_atoms[i, j]
                if pos.mean() != pos.mean():
                    continue
                name = f' {atom}'
                alt_loc = ''
                res_name_3 = restype_1to3[amino]
                if chain_id != chain_ids[chain_index[i]]:
                    chain_start = i
                    chain_id = chain_ids[chain_index[i]]
                insertion_code = ''
                occupancy = 1.00
                b_factor = 0.00
                element = atom[0]
                charge = ''
                # PDB is a columnar format, every space matters here!
                atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                             f'{res_name_3:>3} {chain_id:>1}'
                             f'{(i + 1 - chain_start):>4}{insertion_code:>1}   '
                             f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                             f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                             f'{element:>2}{charge:>2}')
                pdb_lines.append(atom_line)
                atom_index += 1

    return "\n".join(pdb_lines)