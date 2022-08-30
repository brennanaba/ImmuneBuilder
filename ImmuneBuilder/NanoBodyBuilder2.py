import torch
import os
from ImmuneBuilder.models import StructureModule
from ImmuneBuilder.util import get_encoding, to_pdb
from ImmuneBuilder.refine import refine

torch.set_default_tensor_type(torch.DoubleTensor)

class Nanobody:
    def __init__(self, sequence_dict, output):
        self.atoms, self.encodings = output
        self.sequence_dict = sequence_dict

    def save_unrefined(self, filename):
        unrefined = to_pdb(self.sequence_dict, self.atoms)

        with open(filename, "w+") as file:
            file.write(unrefined)


    def save(self, filename):
        try:
            name, filetype = filename.split(".")
        except ValueError:
            name = "unrefined"
            filetype = "pdb"

        unrefined_filename = name + "_unrefined." + filetype
        self.save_unrefined(unrefined_filename)
        refine(unrefined_filename, filename)


class NanoBodyBuilder2:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        current_directory = os.path.dirname(os.path.realpath(__file__))

        self.model = StructureModule(rel_pos_dim=64, embed_dim=128).to(self.device)
        self.model.eval()
        path = os.path.join(current_directory, "trained_model", "nano_model_1")
        self.model.load_state_dict(torch.load(path, map_location=torch.device(self.device)))


    def predict(self, sequence_dict):
        with torch.no_grad():
            sequence_dict["L"] = ""
            encoding = torch.tensor(get_encoding(sequence_dict, "H"), device = self.device)
            full_seq = "".join(sequence_dict.values())

            output = self.model(encoding, full_seq)

        return Nanobody(sequence_dict, output)
