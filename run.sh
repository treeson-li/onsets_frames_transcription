TRAIN_EXAMPLES=/home/admin1/data/tfrecord/maps/maps_config2_train_spec.tfrecord
#/media/admin1/Windows/MAPS_TFRECORD/maps_config2_train.tfrecord
#/home/admin1/data/tfrecord/maps/maps_config2_train_spec.tfrecord
RUN_DIR=~/data/orig_new_preprocess/
#~/data/onsets_frames_orig/

nohup python ./onsets_frames_transcription_train.py \
	  --examples_path="${TRAIN_EXAMPLES}" \
	    --run_dir="${RUN_DIR}" \
	      --mode='train' > log.txt 2>&1 &

