---

<div align="center">    
 
# ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins 

</div>

---
    

## Abstract

Immune receptor proteins play a key role in the immune system and have shown great promise as biotherapeutics. The structure of these proteins is critical for understanding what antigen they bind. Here, we present ImmuneBuilder, a set of deep learning models trained to accurately predict the structure of antibodies (ABodyBuilder2), nanobodies (NanoBodyBuilder2) and T-Cell receptors (TCRBuilder2). We show that ImmuneBuilder generates structures with state of the art accuracy while being much faster than AlphaFold2. For example, on a benchmark of 34 recently solved antibodies, ABodyBuilder2 predicts CDR-H3 loops with an RMSD of 2.81Å, a 0.09Å improvement over AlphaFold-Multimer, while being over a hundred times faster. Similar results are also achieved for nanobodies (NanoBodyBuilder2 predicts CDR-H3 loops with an average RMSD of 2.89Å, a 0.55Å improvement over AlphaFold2) and TCRs. By predicting an ensemble of structures, ImmuneBuilder also gives an error estimate for every residue in its final prediction.


## Colab

To test the method out without installing it you can try this <a href="https://colab.research.google.com/github/brennanaba/ImmuneBuilder/blob/main/notebook/ImmuneBuilder.ipynb">Google Colab</a>

## Install

You can install ImmuneBuilder via PyPI by doing:

```bash
$ pip install ImmuneBuilder
```

This package requires PyTorch. If you do not already have PyTorch installed, you can do so following these <a href="https://pytorch.org/get-started/locally/">instructions</a>.

It also requires OpenMM and pdbfixer for the refinement step. 
OpenMM and pdbfixer can be installed via conda using:

```bash
$ conda install -c conda-forge openmm pdbfixer
```

It also uses anarci for trimming and numbering sequences. We recommend installing ANARCI from <a href="https://github.com/oxpig/ANARCI/tree/master">here</a>, but it can also be installed using (maintained by a third party):

```bash
$ conda install -c bioconda anarci
```

## Usage

### Antibody structure prediction

To predict an antibody structure using the python API you can do the following.

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

ABodyBuilder2 can also be used via de command line. To do this you can use:

```bash
ABodyBuilder2 --fasta_file my_antibody.fasta -v
```

You can get information about different options by using:

```bash
ABodyBuilder2 --help
```

I would recommend using the python API if you intend to predict many structures as you only have to load the models once.

Happy antibodies!!

### Nanobody structure prediction

The python API for nanobodies is quite similar than for antibodies.

```python
from ImmuneBuilder import NanoBodyBuilder2
predictor = NanoBodyBuilder2()

output_file = "my_nanobody.pdb"
sequence = {'H': 'QVQLVESGGGLVQPGESLRLSCAASGSIFGIYAVHWFRMAPGKEREFTAGFGSHGSTNYAASVKGRFTMSRDNAKNTTYLQMNSLKPADTAVYYCHALIKNELGFLDYWGPGTQVTVSS'}

nanobody = predictor.predict(sequence)
nanobody.save(output_file)
```

And it can also be used from the command line:

```bash
NanoBodyBuilder2 --fasta_file my_nanobody.fasta -v
```

### TCR structure prediction

It is all pretty much the same for TCRs

```python
from ImmuneBuilder import TCRBuilder2
predictor = TCRBuilder2()

output_file = "my_tcr.pdb"
sequences = {
"A": "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
"B": "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"}

tcr = predictor.predict(sequences)
tcr.save(output_file)
```

And it can also be used from the command line:

```bash
TCRBuilder2 --fasta_file my_tcr.fasta -v
```

### Fasta formatting

If you wish to run the model on a sequence from a fasta file it must be formatted as follows:

```
>H
YOURHEAVYCHAINSEQUENCE
>L
YOURLIGHCHAINSEQUENCE
```

If you are running it on TCRs the chain labels should be A for the alpha chain and B for the beta chain. On nanobodies the fasta file should only contain a heavy chain labelled H.

## Issues and Pull requests

Please submit issues and pull requests on this <a href="https://github.com/brennanaba/ImmuneBuilder">repo</a>.

### Known issues

- Installing OpenMM from conda will automatically download the latest version of cudatoolkit which may not be compatible with your device. For more information on this please checkout the following <a href="https://github.com/brennanaba/ImmuneBuilder/issues/13">issue</a>.


## Citing this work

The code and data in this package is based on the following paper <a href="https://www.biorxiv.org/content/10.1101/2022.11.04.514231v1">ImmuneBuilder</a>. If you use it, please cite:

```tex
@article {Abanades2022.11.04.514231,
	title = {ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins.},
	author = {Abanades, Brennan and Wong, Wing Ki and Boyles, Fergus and Georges, Guy and Bujotzek, Alexander and Deane, Charlotte Mary},
 journal = {bioRxiv},
	year = {2022},
	doi = {10.1101/2022.11.04.514231},
	publisher = {Cold Spring Harbor Laboratory}
}
```

