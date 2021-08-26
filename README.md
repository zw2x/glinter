## GLINTER: Graph Learning of INTER-protein contacts

## Installation
Require `python>=3.7`

Clone the repository and install it
```bash
git clone https://github.com/zw2x/glinter.git
cd glinter
pip install -e .
```

Manually install the following softwares and models
* [torch>=1.6](https://pytorch.org/)
* [torch_geometric](https://github.com/rusty1s/pytorch_geometric)
* [MSMS](http://mgltools.scripps.edu/packages/MSMS/)
* [reduce](http://kinemage.biochem.duke.edu/software/reduce.php)
* [hh-suite](https://github.com/soedinglab/hh-suite)
* [esm_msa1_t12_100M_UR50S](https://github.com/facebookresearch/esm)

The taxonomy database and model weights can be found at 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5172929.svg)](https://doi.org/10.5281/zenodo.5172929)


## Usage
The following commands are executed in the repository `glinter`.

Replace the environment variables in `set_env.sh` and `run.sh` in the `scripts` directory 
by your installation paths and then run
```bash
source scripts/set_env.sh
```

There are two example pdb files in the `examples` directory. 
To predict the inter-protein contacts between `1a59A.pdb` and `1a59B.pdb`, run
```bash
scripts/run.sh examples/1a59A.pdb examples/1a59B.pdb examples/
```

The output is `examples/1a59A:1a59B/1a59A:1a59B.out.pkl`

## Note
Please use the [uniclust database](http://wwwuser.gwdg.de/~compbiol/uniclust/2016_09/)
`A3M_SpecBloc` requires the header of each hit starts with `tr|` and contains `OS=$TAX`, `$TAX` is the taxonomy name.

