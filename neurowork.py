import tensorflow as tf


def get_train_batches(image_paths, labels):
    # Convert to tensors
    print(image_paths)
    image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    print(image_paths)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.shuffle(image_paths.__sizeof__() * 10, reshuffle_each_iteration=True)
    #for element in dataset.as_numpy_iterator():
     #   print(element)
