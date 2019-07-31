TRAIN_EXAMPLES=/media/admin1/Windows/MAPS_TFRECORD/maps_config2_train.tfrecord
RUN_DIR=~/data/aan/

nohup python ./onsets_frames_transcription_train.py \
	  --examples_path="${TRAIN_EXAMPLES}" \
	    --run_dir="${RUN_DIR}" \
	      --mode='train' > log.txt 2>&1 &

