# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
import argparse
from typing import Dict, Union, Optional

from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline_multimer
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
import numpy as np

import monomer_pipeline

# Internal import (7716).


model_preset = 'multimer'
use_precomputed_msas = False


MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3

def main_argv():
    def split_by_comma(s):
        fasta_paths = s.split(',')
        return [ fasta_path.strip() for fasta_path in fasta_paths ]

    parser = argparse.ArgumentParser('run simple alphafold-multimer')
    parser.add_argument('--fasta-paths', type=split_by_comma)
    parser.add_argument(
        '--hhblits-binary-path', type=str, default=shutil.which('hhblits')
    )
    parser.add_argument(
        '--jackhmmer-binary-path', type=str, default=shutil.which('jackhmmer')
    )
    parser.add_argument('--uniclust30-database-path', type=str)
    parser.add_argument('--uniprot-database-path', type=str)
    parser.add_argument('--use-precomputed-msas', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--random-seed', type=int, default=None)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--data-dir', type=str)

    argv = parser.parse_args()

    return argv

def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: pipeline_multimer.DataPipeline,
    model_runners: Dict[str, model.RunModel],
    random_seed: int,
    benchmark: bool = False,
    amber_relaxer = None,
    is_prokaryote: Optional[bool] = None):
  """Predicts structure using AlphaFold for the given sequence."""
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features.
  t_0 = time.time()
  if is_prokaryote is None:
    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir)
  else:
    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir,
        is_prokaryote=is_prokaryote)
  timings['features'] = time.time() - t_0

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)

  unrelaxed_pdbs = {}
  relaxed_pdbs = {}
  ranking_confidences = {}

  # Run the models.
  num_models = len(model_runners)
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()):
    t_0 = time.time()
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed)
    timings[f'process_features_{model_name}'] = time.time() - t_0

    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature_dict,
                                             random_seed=model_random_seed)
    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff

    if benchmark:
      t_0 = time.time()
      model_runner.predict(processed_feature_dict,
                           random_seed=model_random_seed)
      t_diff = time.time() - t_0
      timings[f'predict_benchmark_{model_name}'] = t_diff

    plddt = prediction_result['plddt']
    ranking_confidences[model_name] = prediction_result['ranking_confidence']

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(prediction_result, f, protocol=4)

    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode)

    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(unrelaxed_pdbs[model_name])

    if amber_relaxer:
      # Relax the prediction.
      t_0 = time.time()
      relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
      timings[f'relax_{model_name}'] = time.time() - t_0

      relaxed_pdbs[model_name] = relaxed_pdb_str

      # Save the relaxed PDB.
      relaxed_output_path = os.path.join(
          output_dir, f'relaxed_{model_name}.pdb')
      with open(relaxed_output_path, 'w') as f:
        f.write(relaxed_pdb_str)

  # Rank by model confidence and write out relaxed PDBs in rank order.
  ranked_order = []
  for idx, (model_name, _) in enumerate(
      sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)):
    ranked_order.append(model_name)
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      if amber_relaxer:
        f.write(relaxed_pdbs[model_name])
      else:
        f.write(unrelaxed_pdbs[model_name])

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    f.write(json.dumps(
        {label: ranking_confidences, 'order': ranked_order}, indent=4))


  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))


def main(argv):
  model_preset = 'multimer'
  run_multimer_system = 'multimer' in model_preset
  num_ensemble = 1

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in argv.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  is_prokaryote_list = [False] * len(fasta_names)

  monomer_data_pipeline = monomer_pipeline.SimpleDataPipeline(
      hhblits_binary_path=argv.hhblits_binary_path,
      uniclust30_database_path=argv.uniclust30_database_path,
      use_precomputed_msas=argv.use_precomputed_msas)

  data_pipeline = pipeline_multimer.DataPipeline(
      monomer_data_pipeline=monomer_data_pipeline,
      jackhmmer_binary_path=argv.jackhmmer_binary_path,
      uniprot_database_path=argv.uniprot_database_path,
      use_precomputed_msas=argv.use_precomputed_msas)

  model_runners = {}
  model_names = config.MODEL_PRESETS[model_preset]
  for model_name in model_names:
    model_config = config.model_config(model_name)
    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=argv.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner


  amber_relaxer = None

  random_seed = argv.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_names))

  # Predict structure for each of the sequences.
  for i, fasta_path in enumerate(argv.fasta_paths):
    is_prokaryote = is_prokaryote_list[i] if run_multimer_system else None
    fasta_name = fasta_names[i]
    predict_structure(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=argv.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=argv.benchmark,
        random_seed=random_seed,
        is_prokaryote=is_prokaryote)


if __name__ == '__main__':
  argv = main_argv()  
  main(argv)
