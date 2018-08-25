import tensorflow as tf
from .utils import mobilenet_module

def decoder(embedded_word_ids, feature_maps, reuse=False):
    with tf.name_scope('decoder'):
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.GRUCell(512, reuse=tf.AUTO_REUSE) for i in range(3)
        ])
        outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                           initial_state=tuple(
                                           [feature_maps  for i in range(3)]),
                                           inputs=embedded_word_ids,
                                           dtype=tf.float32)
    return outputs, state

def caption_decoder(features, labels, mode, params):
    module, _, _, _ = mobilenet_module()
    feature_map = module(features['imgs'])

    word_embeddings = tf.get_variable("word_embeddings", 
        [params['vocab_size'], params['embedding_size']])
    embedded_word_ids = tf.nn.embedding_lookup(word_embeddings,
        features['word_ids'])
    feature_maps = tf.layers.dense(feature_map, 512)
    if mode == tf.estimator.ModeKeys.PREDICT:
        gru_out, states = decoder(embedded_word_ids,
                                  feature_maps)
    else:
        gru_out, states = decoder(embedded_word_ids,
                                  feature_maps)
    outputs = tf.layers.dense(gru_out, params['vocab_size'])


    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'preds': outputs
        }
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                              logits=outputs)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.RMSPropOptimizer(1e-3)
        train_op = optimizer.minimize(cost,
                                      global_step=tf.train.get_global_step())
        spec= tf.estimator.EstimatorSpec(mode=mode,
                                         loss=cost, train_op=train_op)

    return spec
