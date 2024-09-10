import click
import torch
import tqdm
from Bio import SeqIO

from ImmuneBuilder import NanoBodyBuilder2
from ImmuneBuilder.util import get_encoding
from ImmuneBuilder.sequence_checks import number_sequences


def _number_sequence(sequence_dict):
    return number_sequences(sequence_dict, allowed_species=None, scheme = 'imgt')

def _get_sequence_from_numbered(numbered_sequence_dict):
    return {chain: "".join([x[1] for x in numbered_sequence_dict[chain]]) for chain in numbered_sequence_dict}

def _encode(sequence_dict):
    return torch.tensor(get_encoding(sequence_dict, 'H'), dtype=torch.get_default_dtype())

@click.command()
@click.argument('fasta', type=str, required=True)
@click.option('-d', '--device', help="Device to run on", type=str, default="cpu")
@click.option('-o', '--output', help="Output pickle file", type=str, required=True)
def main(fasta, device, output):
    models = NanoBodyBuilder2().models.values()
    for model in models:
        model.to(device)
        model.eval()

    results = dict()
    for record in tqdm.tqdm(SeqIO.parse(fasta, "fasta")):
        try:
            sequence_dict_numbered = _number_sequence({'H': str(record.seq)})
            sequence_dict = _get_sequence_from_numbered(sequence_dict_numbered)

            node_features = _encode(sequence_dict)
            sequence = sequence_dict['H']

            # need to add dummy light chain key
            sequence_dict_numbered['L'] = []

            outputs = []
            with torch.no_grad():
                for model in models:
                    out = model(node_features.to(device), sequence)
                    out = (out[0].detach().cpu(), out[1].detach().cpu())
                    outputs.append(out)
            results[record.id] = {'predictions': outputs, 'sequence_numbered': sequence_dict_numbered}
        except Exception as e:
            print(f"Error processing {record.id}: {e}")
            continue
    
    torch.save(results, output)

if __name__ == '__main__':
    main()