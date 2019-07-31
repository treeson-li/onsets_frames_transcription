import tensorflow as tf
import data
import configs


audio_tfrecord = '/media/admin1/Windows/MAPS_TFRECORD/maps_config2_train.tfrecord'
spec_tfrecord = '/home/admin1/data/tfrecord/maps/maps_config2_train_spec.tfrecord'

config = configs.CONFIG_MAP['onsets_frames']
hparams = config.hparams
hparams.batch_size = 1


dataset = data.provide_batch(
      examples=audio_tfrecord,
      preprocess_examples=True,
      hparams=hparams,
      is_training=False)
print('\n dataset is:')
print(dataset)

iterator = dataset.make_initializable_iterator()
next_record = iterator.get_next()
print('\n itetator and next record is :')
print(iterator)
print(next_record)

writer = tf.python_io.TFRecordWriter(spec_tfrecord, options=None)
exit
with tf.io.TFRecordWriter(spec_tfrecord) as writer:
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        try:
            i = 0
            while True:
                print('i is: ', i, end='\r')
                i += 1
                result = sess.run(next_record)
                feature = result[0]
                label = result[1]

                spec = feature[0].flatten()
                spectrogram_hash = feature[3].flatten()
                labels = label[0].flatten()
                label_weights = label[1].flatten()
                length = feature[1].flatten()
                onsets = label[2].flatten()
                offsets = label[3].flatten()
                velocities = label[4].flatten()
                sequence_id = feature[2]
                note_sequence = label[5]

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'spec': tf.train.Feature(float_list = tf.train.FloatList(value=spec)),
                            'spectrogram_hash':tf.train.Feature(int64_list = tf.train.Int64List(value=spectrogram_hash)),
                            'labels':tf.train.Feature(float_list = tf.train.FloatList(value=labels)),
                            'label_weights':tf.train.Feature(float_list = tf.train.FloatList(value=label_weights)),
                            'length':tf.train.Feature(int64_list = tf.train.Int64List(value=length)),
                            'onsets':tf.train.Feature(float_list = tf.train.FloatList(value=onsets)),
                            'offsets':tf.train.Feature(float_list = tf.train.FloatList(value=offsets)),
                            'velocities':tf.train.Feature(float_list = tf.train.FloatList(value=velocities)),
                            'sequence_id':tf.train.Feature(bytes_list = tf.train.BytesList(value=sequence_id)),
                            'note_sequence':tf.train.Feature(bytes_list = tf.train.BytesList(value=note_sequence)),
                        }))
                writer.write(example.SerializeToString()) 
                # print(type(spec), spec)
                # print(type(length), length)
                # print(type(sequence_id), sequence_id)
                # print(type(spectrogram_hash), spectrogram_hash)
                # print(type(labels), labels)
                # print(type(note_sequence), note_sequence)
        except tf.errors.OutOfRangeError:
            print('in except, done')

