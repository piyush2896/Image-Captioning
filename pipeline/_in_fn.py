import tensorflow as  tf
from functools import partial

def preprocess(filename, caption_seqs, img_size, is_rgb=True):
    bytes = tf.read_file(filename)
    img_decoded = tf.image.decode_jpeg(bytes, channels=3 if is_rgb else 1)
    img_resized = tf.image.resize_images(img_decoded, img_size)
    img_normalized = tf.divide(tf.cast(img_resized, 'float'), tf.constant(255.))
    img_flipped = tf.image.random_flip_left_right(img_normalized)

    caption_seqs = tf.random_shuffle(caption_seqs)

    return img_flipped, caption_seqs[0]

def train_input_fn(files,
                   caption_seqs,
                   img_size,
                   is_rgb=True,
                   buffer_size=2000,
                   batch_size=32,
                   repeat=None):
    files = tf.constant(files)
    caption_seqs = tf.constant(caption_seqs)
    dataset = tf.data.Dataset.from_tensor_slices((files, caption_seqs))
    process = partial(preprocess, img_size=img_size, is_rgb=is_rgb)
    dataset = dataset.map(process)

    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(2000)
    imgs, caption_seqs = dataset.make_one_shot_iterator().get_next()

    in_caption_seqs = caption_seqs[:, :-1]
    out_caption_seqs = caption_seqs[:, 1:]

    return {
        'imgs': imgs,
        'word_ids':in_caption_seqs}, out_caption_seqs

def predict_input_fn(files,
                     caption_seqs,
                     img_size,
                     is_rgb=True,
                     buffer_size=2000,
                     batch_size=32):
    files = tf.constant(files)
    caption_seqs = tf.constant(caption_seqs)
    dataset = tf.data.Dataset.from_tensor_slices((files, caption_seqs))

    def process(filename, caption_seqs):
        bytes = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(bytes, channels=3 if is_rgb else 1)
        img_resized = tf.image.resize_images(img_decoded, img_size)
        img_normalized = tf.divide(tf.cast(img_resized, 'float'), tf.constant(255.))
        return img_normalized, caption_seqs

    dataset = dataset.map(process)

    dataset = dataset.batch(batch_size).prefetch(2000)
    imgs, caption_seqs = dataset.make_one_shot_iterator().get_next()
    return {
        'imgs': imgs,
        'word_ids': caption_seqs
    }