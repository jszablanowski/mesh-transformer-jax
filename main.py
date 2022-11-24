import tensorflow as tf 
raw_dataset = tf.data.TFRecordDataset("E:/personal/fine-tune-bucket/minipilot-bucket/small_133947.tfrecords")

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

# for example in tf.python_io.tf_record_iterator("E:\personal\fine-tune-bucket\minipilot-bucket\small_133947.tfrecords"):
#     example = tf.train.Example.FromString(example)