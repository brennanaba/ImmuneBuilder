import torch
import numpy as np
import os
import argparse
from ImmuneBuilder.models import StructureModule
from ImmuneBuilder.util import get_encoding, to_pdb, find_alignment_transform, download_file, sequence_dict_from_fasta
from ImmuneBuilder.refine import refine

torch.set_default_tensor_type(torch.DoubleTensor)
embed_dim = {
    "antibody_model_1":128,
    "antibody_model_2":256,
    "antibody_model_3":256,
    "antibody_model_4":256
}

model_urls = {
    "antibody_model_2": "https://dl.dropbox.com/s/hez39qf0kyncscw/antibody_model_2?dl=1",
    "antibody_model_3": "https://dl.dropbox.com/s/tsk4zw5xsj0a7pk/antibody_model_3?dl=1",
    "antibody_model_4": "https://dl.dropbox.com/s/quww8407ae7f076/antibody_model_4?dl=1",
}

class Antibody:
    def __init__(self, sequence_dict, predictions):
        self.sequence_dict = sequence_dict
        self.atoms = [x[0] for x in predictions]
        self.encodings = [x[1] for x in predictions]

        with torch.no_grad():
            traces = torch.stack([x[:,0] for x in self.atoms])
            self.R,self.t = find_alignment_transform(traces)
            self.aligned_traces = (traces-self.t) @ self.R
            self.error_estimates = (self.aligned_traces - self.aligned_traces.mean(0)).square().sum(-1)
            self.ranking = [x.item() for x in self.error_estimates.mean(-1).argsort()]
        

    def save_single_unrefined(self, filename, index=0):
        atoms = (self.atoms[index] - self.t[index]) @ self.R[index]
        unrefined = to_pdb(self.sequence_dict, atoms)

        with open(filename, "w+") as file:
            file.write(unrefined)

    
    def save_single_refined(self, filename, index=0):
        try:
            name, filetype = filename.split(".")
        except ValueError:
            name = "prediction"
            filetype = "pdb"

        unrefined_filename = name + "_unrefined." + filetype
        self.save_single_unrefined(unrefined_filename, index=index)
        refine(unrefined_filename, filename)


    def save(self, dirname="ABodyBuilder2_output", filename="final_model.pdb"):
        os.makedirs(dirname, exist_ok = True)

        for i in range(len(self.atoms)):

            unrefined_filename = os.path.join(dirname,f"rank{self.ranking.index(i)}_unrefined.pdb")
            self.save_single_unrefined(unrefined_filename, index=i)

        np.save(os.path.join(dirname,"error_estimates"), self.error_estimates.mean(0).numpy())
        refine(os.path.join(dirname,"rank0_unrefined.pdb"), os.path.join(dirname, filename))


class ABodyBuilder2:
    def __init__(self, model_ids = [1,2,3,4]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        current_directory = os.path.dirname(os.path.realpath(__file__))

        self.models = {}
        for id in model_ids:
            model_file = f"antibody_model_{id}"
            model = StructureModule(rel_pos_dim=64, embed_dim=embed_dim[model_file]).to(self.device)
            weights_path = os.path.join(current_directory, "trained_model", model_file)

            try:
                if not os.path.exists(weights_path):
                    print(f"Downloading weights for {model_file}...")
                    download_file(model_urls[model_file], weights_path)

                model.load_state_dict(torch.load(weights_path, map_location=torch.device(self.device)))
            except Exception:
                print(f"{model_file} not downloaded or corrupted.")
                continue

            model.eval()

            self.models[model_file] = model


    def predict(self, sequence_dict):
        with torch.no_grad():
            encoding = torch.tensor(get_encoding(sequence_dict), device = self.device)
            full_seq = "".join(sequence_dict.values())
            outputs = []

            for model_file in self.models:
                pred = self.models[model_file](encoding, full_seq)
                outputs.append(pred)

        return Antibody(sequence_dict, outputs)


def command_line_interface():
    parser = argparse.ArgumentParser()

    parser.add_argument("-H", "--heavy_sequence", help="Heavy chain amino acid sequence", default=None)
    parser.add_argument("-L", "--light_sequence", help="Light chain amino acid sequence", default=None)
    parser.add_argument("-f", "--fasta_file", help="Fasta file containing a heavy amd light chain named H and L", default=None)

    parser.add_argument("-o", "--output", help="Path to where the output model should be saved. Defaults to the same "
                                            "directory as input file.", default=None)

    args = parser.parse_args()

    if (args.heavy_sequence is not None) and (args.light_sequence is not None):
        seqs = {"H":args.heavy_sequence, "L":args.light_sequence}
    elif args.fasta_file is not None:
        seqs = sequence_dict_from_fasta(args.fasta_file)
    else:
        raise ValueError("Missing input sequences")
    
    model = ABodyBuilder2()
    ab = model.predict(seqs)

    output = args.output if args.output is not None else "ABodyBuilder2_output"
    ab.save(output)