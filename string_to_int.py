import os
import multiprocessing as mp
import tensorflow as tf


def convert_single(i):
    input_path = f'/home/admin/workspace/shared/GLD_v2_tfrecord/old_tfrecord/tf_record.{i}'
    output_path = f'/home/admin/workspace/shared/GLD_v2_tfrecord/new_tfrecord/tf_record.{i}'

    old_features = {
        'label': tf.io.FixedLenFeature([], tf.string),
        'data': tf.io.FixedLenFeature([], tf.string)
    }

    writer = tf.io.TFRecordWriter(output_path)
    for example in tf.compat.v1.io.tf_record_iterator(input_path):
        features = tf.io.parse_single_example(example, features=old_features)
        label = tf.cast(tf.compat.v1.string_to_number(features['label']), tf.int32)
        image = features['data']

        label_feature = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(label)]))
        image_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image.numpy()]))
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'label': label_feature,
                'image': image_feature}))
        writer.write(example.SerializeToString())
    writer.close()
    if i % 100 == 0:
        print(f'{i} records done.')

def main():
    pool = mp.Pool(10)
    pool.map(convert_single, range(1000))
    pool.close()
    pool.join()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    main()
