import torch
import numpy as np
import os
import sys
import argparse
import json
from ImmuneBuilder.models import StructureModule
from ImmuneBuilder.util import (
    get_encoding,
    to_pdb,
    find_alignment_transform,
    download_file,
    sequence_dict_from_fasta,
    add_errors_as_bfactors,
    are_weights_ready,
)
from ImmuneBuilder.refine import refine
from ImmuneBuilder.sequence_checks import number_sequences

EMBED_DIM = {
    "tcr_model_1": 128,
    "tcr_model_2": 128,
    "tcr_model_3": 256,
    "tcr_model_4": 256,
}

model_urls = {
    "tcr_model_1": "https://zenodo.org/record/7258553/files/tcr_model_1?download=1",
    "tcr_model_2": "https://zenodo.org/record/7258553/files/tcr_model_2?download=1",
    "tcr_model_3": "https://zenodo.org/record/7258553/files/tcr_model_3?download=1",
    "tcr_model_4": "https://zenodo.org/record/7258553/files/tcr_model_4?download=1",
}

header = "REMARK  TCR STRUCTURE MODELLED USING TCRBUILDER2                                \n"


class TCR:
    def __init__(self, numbered_sequences, predictions):
        self.numbered_sequences = numbered_sequences
        self.atoms = [x[0] for x in predictions]
        self.encodings = [x[1] for x in predictions]

        with torch.no_grad():
            traces = torch.stack([x[:, 0] for x in self.atoms])
            self.R, self.t = find_alignment_transform(traces)
            self.aligned_traces = (traces - self.t) @ self.R
            self.error_estimates = (
                (self.aligned_traces - self.aligned_traces.mean(0))
                .square()
                .sum(-1)
            )
            self.ranking = [
                x.item() for x in self.error_estimates.mean(-1).argsort()
            ]

    def save_single_unrefined(self, filename, index=0):
        atoms = (self.atoms[index] - self.t[index]) @ self.R[index]
        unrefined = to_pdb(self.numbered_sequences, atoms, chain_ids="BA")

        with open(filename, "w+") as file:
            file.write(unrefined)


    def save_all(self, dirname=None, filename=None, check_for_strained_bonds=True, n_threads=-1):
        if dirname is None:
            dirname = "TCRBuilder2_output"
        if filename is None:
            filename = "final_model.pdb"
        os.makedirs(dirname, exist_ok=True)

        for i in range(len(self.atoms)):
            unrefined_filename = os.path.join(
                dirname,
                f"rank_{self.ranking.index(i)}_model_{i}_unrefined.pdb",
            )
            self.save_single_unrefined(unrefined_filename, index=i)

        np.save(os.path.join(dirname,"error_estimates"), self.error_estimates.mean(0).cpu().numpy())
        final_filename = os.path.join(dirname, filename)
        refine(os.path.join(dirname,"rank0_unrefined.pdb"), final_filename, check_for_strained_bonds=check_for_strained_bonds, n_threads=n_threads)
        add_errors_as_bfactors(final_filename, self.error_estimates.mean(0).sqrt().cpu().numpy(), header=[header])

        if not refine_all:
            final_filename = os.path.join(dirname, filename)
            refine(
                os.path.join(
                    dirname,
                    f"rank_0_model_{self.ranking.index(0)}_unrefined.pdb",
                ),
                final_filename,
                check_for_strained_bonds=check_for_strained_bonds,
            )

            add_errors_as_bfactors(
                final_filename,
                self.error_estimates.mean(0).sqrt().cpu().numpy(),
                header=[header],
            )

        else:
            for i in range(len(self.atoms)):
                unrefined_filename = os.path.join(
                    dirname,
                    f"rank_{self.ranking.index(i)}_model_{i}_unrefined.pdb",
                )
                refined_filename = os.path.join(
                    dirname,
                    f"rank_{self.ranking.index(i)}_model_{i}_refined.pdb",
                )
                refine(
                    unrefined_filename,
                    refined_filename,
                    check_for_strained_bonds=check_for_strained_bonds,
                )
                add_errors_as_bfactors(
                    refined_filename,
                    self.error_estimates.mean(0).sqrt().cpu().numpy(),
                    header=[header],
                )

    def save(self, filename=None, check_for_strained_bonds=True, n_threads=-1):
        if filename is None:
            filename = "TCRBuilder2_output.pdb"

        for i in range(len(self.atoms)):
            self.save_single_unrefined(filename, index=self.ranking.index(i))
            success = refine(filename, filename, check_for_strained_bonds=check_for_strained_bonds, n_threads=n_threads)
            if success:
                break
            else:
                self.save_single_unrefined(filename, index=self.ranking.index(i))
                success = refine(filename, filename, check_for_strained_bonds=check_for_strained_bonds, n_threads=n_threads)
                if success:
                    break

        if not success:
            print(f"FAILED TO REFINE {filename}.\nSaving anyways.", flush=True)
        add_errors_as_bfactors(
            filename,
            self.error_estimates.mean(0).sqrt().cpu().numpy(),
            header=[header],
        )


class TCRBuilder2:
    def __init__(
        self, model_ids=[1, 2, 3, 4], weights_dir=None, numbering_scheme="imgt"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scheme = numbering_scheme
        if weights_dir is None:
            weights_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "trained_model"
            )
        if os.path.exists(os.path.join(weights_dir, "embed_dims.json")):
            with open(os.path.join(weights_dir, "embed_dims.json"), "r") as f:
                embed_dim = json.load(f)
        else:
            embed_dim = EMBED_DIM

        self.models = {}
        for id in model_ids:
            model_file = f"tcr_model_{id}"
            model = StructureModule(
                rel_pos_dim=64, embed_dim=embed_dim[model_file]
            ).to(self.device)
            weights_path = os.path.join(weights_dir, model_file)

            try:
                if not are_weights_ready(weights_path):
                    print(
                        f"Downloading weights for {model_file}...", flush=True
                    )
                    download_file(model_urls[model_file], weights_path)

                model.load_state_dict(
                    torch.load(
                        weights_path, map_location=torch.device(self.device)
                    )
                )
            except Exception as e:
                print(
                    f"ERROR: {model_file} not downloaded or corrupted.",
                    flush=True,
                )
                raise e

            model.eval()
            model.to(torch.get_default_dtype())

            self.models[model_file] = model

    def predict(self, sequence_dict):
        numbered_sequences = number_sequences(
            sequence_dict, scheme=self.scheme
        )
        sequence_dict = {
            chain: "".join([x[1] for x in numbered_sequences[chain]])
            for chain in numbered_sequences
        }

        with torch.no_grad():
            sequence_dict = {"H": sequence_dict["B"], "L": sequence_dict["A"]}
            encoding = torch.tensor(
                get_encoding(sequence_dict),
                device=self.device,
                dtype=torch.get_default_dtype(),
            )
            full_seq = sequence_dict["H"] + sequence_dict["L"]
            outputs = []

            for model_file in self.models:
                pred = self.models[model_file](encoding, full_seq)
                outputs.append(pred)

        return TCR(numbered_sequences, outputs)


def command_line_interface():
    description = """
        TCRBuilder2                                                 || || 
        A Method for T-Cell Receptor Structure Prediction          |VB|VA| 
        Author: Brennan Abanades Kenyon                            |CB|CA| 
        Supervisor: Charlotte Deane                             -------------
    """

    parser = argparse.ArgumentParser(
        prog="TCRBuilder2",
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-B",
        "--beta_sequence",
        help="Beta chain amino acid sequence",
        default=None,
    )
    parser.add_argument(
        "-A",
        "--alpha_sequence",
        help="Alpha chain amino acid sequence",
        default=None,
    )
    parser.add_argument(
        "-f",
        "--fasta_file",
        help="Fasta file containing a beta and alpha chain named B and A",
        default=None,
    )

    parser.add_argument("-o", "--output", help="Path to where the output model should be saved. Defaults to the same directory as input file.", default=None)
    parser.add_argument("--to_directory", help="Save all unrefined models and the top ranked refined model to a directory. " 
    "If this flag is set the output argument will be assumed to be a directory", default=False, action="store_true")
    parser.add_argument("--n_threads", help="The number of CPU threads to be used. If this option is set, refinement will be performed on CPU instead of GPU. By default, all available cores will be used.", type=int, default=-1)
    parser.add_argument("-u", "--no_sidechain_bond_check", help="Don't check for strained bonds. This is a bit faster but will rarely generate unphysical side chains", default=False, action="store_true")
    parser.add_argument("-v", "--verbose", help="Verbose output", default=False, action="store_true")

    args = parser.parse_args()

    if (args.beta_sequence is not None) and (args.alpha_sequence is not None):
        seqs = {"B": args.beta_sequence, "A": args.alpha_sequence}
    elif args.fasta_file is not None:
        seqs = sequence_dict_from_fasta(args.fasta_file)
    else:
        raise ValueError("Missing input sequences")

    check_for_strained_bonds = not args.no_sidechain_bond_check
    
    if args.n_threads > 0:
        torch.set_num_threads(args.n_threads)    

    if args.verbose:
        print(description, flush=True)
        print(
            f"Sequences loaded succesfully.\nAlpha and Beta chains are:",
            flush=True,
        )
        [print(f"{chain}: {seqs[chain]}", flush=True) for chain in "AB"]
        print("Running sequences through deep learning model...", flush=True)

    try:
        tcr = TCRBuilder2().predict(seqs)
    except AssertionError as e:
        print(e, flush=True)
        sys.exit(1)

    if args.verbose:
        print("TCR modelled succesfully, starting refinement.", flush=True)

    if args.to_directory:
        tcr.save_all(args.output,check_for_strained_bonds=check_for_strained_bonds, n_threads=args.n_threads)
        if args.verbose:
            print(
                "Refinement finished. Saving all outputs to directory",
                flush=True,
            )
    else:
        tcr.save(args.output,check_for_strained_bonds=check_for_strained_bonds, n_threads=args.n_threads)
        if args.verbose:
            outfile = (
                "TCRBuilder2_output.pdb"
                if args.output is None
                else args.output
            )
            print(
                f"Refinement finished. Saving final structure to {outfile}",
                flush=True,
            )
