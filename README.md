# ImmuneBuilder
Predict the structure of proteins of the immune system using deep learning

## Install

To install download from github:

```bash
$ git clone https://github.com/brennanaba/ImmuneBuilder.git
$ pip install ImmuneBuilder/
```

This package requires PyTorch. If you do not already have PyTorch installed, you can do so following these <a href="https://pytorch.org/get-started/locally/">instructions</a>.

It also requires OpenMM and pdbfixer for the refinement step. 
OpenMM and pdbfixer can be installed via conda using:

```bash
$ conda install -c conda-forge openmm pdbfixer
```

## Usage

To predict an antibody structure you can do the following


```python
from ImmuneBuilder import ABodyBuilder2
predictor = ABodyBuilder2()

output_file = "my_antibody.pdb"
sequences = {
  'H': 'EVQLVESGGGVVQPGGSLRLSCAASGFTFNSYGMHWVRQAPGKGLEWVAFIRYDGGNKYYADSVKGRFTISRDNSKNTLYLQMKSLRAEDTAVYYCANLKDSRYSGSYYDYWGQGTLVTVS',
  'L': 'VIWMTQSPSSLSASVGDRVTITCQASQDIRFYLNWYQQKPGKAPKLLISDASNMETGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCQQYDNLPFTFGPGTKVDFK'}

antibody = predictor.predict(sequences)
antibody.save(output_file)
```

Happy antibodies!!
