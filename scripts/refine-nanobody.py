from pathlib import Path

import click
import tqdm
import torch

from ImmuneBuilder.NanoBodyBuilder2 import Nanobody

@click.command()
@click.argument('predictions_pt', type=str, required=True)
@click.option('-o', '--output-fmt', help="Output. Use {id} for record id.", type=str, default="{id}.pdb")
def main(predictions_pt, output_fmt):
    predictions = torch.load(predictions_pt)
    for record_id, result in tqdm.tqdm(predictions.items()):
        out_fpath = Path(output_fmt.format(id=record_id))
        out_fpath.parent.mkdir(parents=True, exist_ok=True)

        nanobody = Nanobody(result['sequence_numbered'], result['predictions'])
        nanobody.save(str(out_fpath))

if __name__ == '__main__':
    main()