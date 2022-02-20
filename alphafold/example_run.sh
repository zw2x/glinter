# run the example script under the current directory

data_dir=$1
esm_path=$2
output_dir=$3
fasta_path=$4
python run_alphafold.py --data-dir ${data_dir} --output-dir ${output_dir} \
    --fasta-paths ${fasta_path} \
    --uniprot-database-path ${data_dir}/uniprot/uniprot.fasta \
    --uniclust30-database-path ${data_dir}/uniclust/uniclust30 \
    --use-precomputed-msa

bash build_feature.sh examples_output/example1/msas/A examples_output/example1
bash build_feature.sh examples_output/example1/msas/B examples_output/example1

bash build_glinter_features.sh examples_output/example1

bash run_glinter.sh examples_output/example1/ranked_0 ${esm_path}
