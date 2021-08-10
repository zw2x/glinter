## GLINTER: Graph Learning of INTER-protein contacts

## Installation
Require `python>=3.7`

Clone the repository and install it
```bash
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

The taxonomy database and model weights can be found at: http://

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
