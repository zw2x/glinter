"""Functions for building the input features for the AlphaFold model."""

import os
import time
import logging
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

from alphafold.data.tools import hhblits
from alphafold.common import residue_constants
from alphafold.data import parsers, pipeline, templates

import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def _make_empty_template_features(num_res):
    """Construct a default template with all zeros."""
    template_features = {
        'template_aatype': np.zeros(
            (1, num_res, len(residue_constants.restypes_with_x_and_gap)),
            dtype=np.float32
        ),                                                         
        'template_all_atom_masks': np.zeros(                                     
            (1, num_res, residue_constants.atom_type_num),
            dtype=np.float32
        ),
        'template_all_atom_positions': np.zeros(
            (1, num_res, residue_constants.atom_type_num, 3), 
            dtype=np.float32
        ),
        'template_domain_names': np.array([''.encode()], dtype=object),
        'template_sequence': np.array([''.encode()], dtype=object),
        'template_sum_probs': np.array([0], dtype=np.float32)
    }
    return template_features

class SimpleDataPipeline:
    """Runs the alignment tools and assembles the input features."""

    def __init__(
        self,
        hhblits_binary_path: str,
        uniclust30_database_path: Optional[str],
        use_precomputed_msas: bool = False
    ):
        """Initializes the data pipeline."""
        self.hhblits_uniclust_runner = hhblits.HHBlits(
            binary_path=hhblits_binary_path,
            databases=[uniclust30_database_path]
        )
        self.use_precomputed_msas = use_precomputed_msas

    def process(
        self, input_fasta_path: str, msa_output_dir: str
    ) -> pipeline.FeatureDict:
        """Runs alignment tools on the input sequence and creates features."""
        with open(input_fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f'More than one input sequence found in {input_fasta_path}.'
            )
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)
        logger.info('start searching uniclust30')
        t0 = time.time()
        uniclust_out_path = os.path.join(msa_output_dir, 'uniclust_hits.a3m')
        hhblits_uniclust_result = pipeline.run_msa_tool(
            msa_runner=self.hhblits_uniclust_runner,
            input_fasta_path=input_fasta_path,
            msa_out_path=uniclust_out_path,
            msa_format='a3m',
            use_precomputed_msas=self.use_precomputed_msas
        )
        logger.info(f'done searching uniclust30 {time.time()-t0:.4f}')
        uniclust_msa = parsers.parse_a3m(hhblits_uniclust_result['a3m'])

        sequence_features = pipeline.make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res
        )

        msa_features = pipeline.make_msa_features((uniclust_msa,))

        empty_templates_features = _make_empty_template_features(num_res)

        return {**sequence_features, **msa_features, **empty_templates_features}
