---

<div align="center">    
 
# ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins 

</div>

---
    

## Abstract

Immune receptor proteins play a key role in the immune system and have shown great promise as biotherapeutics. The structure of these proteins is critical for understanding what antigen they bind. Here, we present ImmuneBuilder, a set of deep learning models trained to accurately predict the structure of antibodies (ABodyBuilder2), nanobodies (NanoBodyBuilder2) and T-Cell receptors (TCRBuilder2). We show that ImmuneBuilder generates structures with state of the art accuracy while being much faster than AlphaFold2. For example, on a benchmark of 34 recently solved antibodies, ABodyBuilder2 predicts CDR-H3 loops with an RMSD of 2.81Å, a 0.09Å improvement over AlphaFold-Multimer, while being over a hundred times faster. Similar results are also achieved for nanobodies (NanoBodyBuilder2 predicts CDR-H3 loops with an average RMSD of 2.89Å, a 0.55Å improvement over AlphaFold2) and TCRs. By predicting an ensemble of structures, ImmuneBuilder also gives an error estimate for every residue in its final prediction.

## Install

To install, download from github:

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


