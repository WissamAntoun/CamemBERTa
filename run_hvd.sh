
horovod -np 6 \
./bind.sh --cpu=node --ib=single --cluster='' -- \
python run_pretraining.py --config_file configs/local_6gpus_hvd.json"