TRAIN_EXAMPLES=/home/admin1/data/tfrecord/maps/maps_config2_train_spec.tfrecord
RUN_DIR=~/data/self/

nohup python ./onsets_frames_transcription_train.py \
	  --examples_path="${TRAIN_EXAMPLES}" \
	    --run_dir="${RUN_DIR}" \
	      --mode='train' > log.txt 2>&1 &

