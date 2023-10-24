#
# 1xV4-16, 1B
# TFLOP/s: 
#
export RUN_NAME=run14-1xv4-32
export DATASET_PATH=gs://jk-gke-aiml-repository/datasets
export BASE_OUTPUT_DIRECTORY=gs://jk-gke-aiml-repository/runs
#export ARGS="enable_checkpointing=false steps=10 dcn_data_parallelism=1 ici_fsdp_parallelism=8 per_device_batch_size=32 remat_policy=full base_emb_dim=2560 base_num_heads=8 base_mlp_dim=8192 head_dim=256 base_num_decoder_layers=32" 
export ARGS="enable_checkpointing=false steps=10 dcn_data_parallelism=2 ici_fsdp_parallelism=8 per_device_batch_size=16 remat_policy=full base_emb_dim=4096 base_num_heads=16 base_mlp_dim=16384 head_dim=256 base_num_decoder_layers=16" 
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"

python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME dataset_path=$DATASET_PATH base_output_directory=$BASE_OUTPUT_DIRECTORY $ARGS

#enable_checkpointing=false per_device_batch_size=4 steps=5 remat_policy=full enable_flash_attention=true  base_emb_dim=4096 base_num_heads=16 base_mlp_dim=16384 head_dim=128 enable_data_shuffling=false max_target_length=4096 dataset_type=synthetic enable_profiler=true

