from pathlib import Path

import click
import tqdm
import torch

from ImmuneBuilder.NanoBodyBuilder2 import Nanobody

@click.command()
@click.argument('predictions_pt', type=str, required=True)
@click.option('-t', '--threads', help="Number of threads to use.", type=int, default=1)
@click.option('-o', '--output-fmt', help="Output. Use {id} for record id.", type=str, default="{id}.pdb")
def main(predictions_pt, output_fmt, threads):
    predictions = torch.load(predictions_pt)
    for record_id, result in tqdm.tqdm(predictions.items()):
        out_fpath = Path(output_fmt.format(id=record_id))
        out_fpath.parent.mkdir(parents=True, exist_ok=True)

        try: 
            nanobody = Nanobody(result['sequence_numbered'], result['predictions'])
            nanobody.save(str(out_fpath), n_threads=threads)
        except Exception as e:
            print(f"Error processing {record_id}: {e}")
            continue

if __name__ == '__main__':
    main()