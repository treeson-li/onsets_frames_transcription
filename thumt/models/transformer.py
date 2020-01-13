# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.interface as interface
import thumt.layers as layers
import tensorflow.contrib.slim as slim

def get_weights(params):
    svocab = params.vocabulary["source"]
    tvocab = params.vocabulary["target"]
    src_vocab_size = len(svocab)
    tgt_vocab_size = len(tvocab)
    vocab_size = tgt_vocab_size

    hidden_size = params.hidden_size
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        if cmp(svocab, tvocab) != 0:
            raise ValueError("Source and target vocabularies are not the same")

        weights = tf.get_variable("weights", [src_vocab_size, hidden_size],
                                  initializer=initializer)
        semb, temb = weights, weights
    else:
        semb = tf.get_variable("source_embedding",
                               [src_vocab_size, hidden_size],
                               initializer=initializer)
        temb = tf.get_variable("target_embedding",
                               [tgt_vocab_size, hidden_size],
                               initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        softmax_weights = temb
    else:
        softmax_weights = tf.get_variable("softmax", [vocab_size, hidden_size],
                                          initializer=initializer)

    return semb, temb, softmax_weights


def layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def residual_fn(x, y, keep_prob=None, is_training=True):
    y = slim.dropout(y, keep_prob, is_training=is_training, scope='residual_fn')
    return x + y


def ffn_layer(inputs, hidden_size, output_size, keep_prob=None, is_training=True,
              dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        hidden = slim.dropout(hidden, keep_prob, is_training=is_training, scope='ffn_layer')

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output


def transformer_encoder(inputs, bias, params, pos=None, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

        outputs = layer_process(x, params.layer_preprocess)

        return outputs


def transformer_decoder(inputs, memory, bias, mem_bias, params, pos=None, dtype=None,
                        scope=None, given_inputs=None):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias] + pos):
        x = inputs
        decoding_outputs = []
        for layer in range(params.num_decoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                # The Average Attention Network
                with tf.variable_scope("position_forward"):
                    # Cumulative Summing
                    if given_inputs is not None:
                        x_fwd = (x + given_inputs[layer]) / pos[0]
                        decoding_outputs.append(tf.expand_dims(x + given_inputs[layer], axis=0))
                    else:
                        if not params.aan_mask:
                            x_fwd = tf.cumsum(x, axis=1) / pos[0]
                        else:
                            x_fwd = tf.matmul(pos[0], x)
                    # FFN activation
                    if params.use_ffn:
                        y = ffn_layer(
                            layer_process(x_fwd, params.layer_preprocess),
                            params.filter_size,
                            params.hidden_size,
                            1.0 - params.relu_dropout,
                        )
                    else:
                        y = x_fwd

                    # Gating layer
                    z = layers.nn.linear(tf.concat([x, y], axis=-1), 
                        params.hidden_size*2, True, True, scope="z_project")
                    i, f = tf.split(z, [params.hidden_size, params.hidden_size], axis=-1)
                    y = tf.sigmoid(i) * x + tf.sigmoid(f) * y
                    
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

        outputs = layer_process(x, params.layer_preprocess)

        if given_inputs is not None:
            decoding_outputs = tf.concat(decoding_outputs, axis=0)
        return outputs, decoding_outputs

def AvgAttentionNet(inputs, params, pos=None, dtype=None, scope=None, is_training=True):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, pos]):
        x = inputs
        normn = tf.cast(params.avg_len, tf.float32)
        normTensor = tf.fill(tf.shape(pos), normn)
        pos = tf.where(tf.less(pos, normn), pos, normTensor)
        for layer in range(params.num_decoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                # The Average Attention Network
                with tf.variable_scope("position_forward"):
                    # Cumulative Summing                    
                    x_cum = tf.cumsum(x, axis=1)
                    x_cum_shift = tf.pad(x_cum, [[0, 0], [normn, 0], [0, 0]])
                    x_cum_shift = tf.slice(x_cum_shift, [0, 0, 0], [-1, tf.shape(x)[1], -1])
                    x_fwd = tf.subtract(x_cum, x_cum_shift)
                    x_fwd = tf.divide(x_fwd, pos)
                    # FFN activation
                    if params.use_ffn:
                        y = ffn_layer(
                            layer_process(x_fwd, params.layer_preprocess),
                            params.ffn_filter_size,
                            params.aan_size,
                            1.0 - params.relu_dropout,
                            is_training=is_training
                        )
                    else:
                        y = x_fwd

                    # Gating layer
                    z = layers.nn.linear(tf.concat([x, y], axis=-1), 
                        params.aan_size*2, True, True, scope="z_project")
                    i, f = tf.split(z, [params.aan_size, params.aan_size], axis=-1)
                    y = tf.sigmoid(i) * x + tf.sigmoid(f) * y
                    
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)
    
    return  x

def AAN_decoder(spec, labels, params, is_training=True):

    hidden_size = params.aan_size
    
    # concat spectrum and labels(no longer yet) as decoder input
    dec_input = slim.fully_connected(spec, hidden_size, scope='spec_dec_input')
    dec_input = slim.dropout(dec_input, 1-params.residual_dropout, is_training=is_training, scope='dec_input_dropout')

    # Preparing decoder input
    dec_pos_bias_fwd = tf.ones_like(dec_input)
    dec_pos_bias_fwd = tf.cumsum(dec_pos_bias_fwd, axis=1)

    # This is a lazy implementation, to assign the correct position embedding, I simply copy the
    # current decoder input 'given_position' times, and assign the whole position embeddings,
    # Only the last decoding value has the meaningful position embeddings
    # TODO: With Average Attention Network, Decoder side position embedding may be unnecessary 
    decoder_input = layers.attention.add_timing_signal(dec_input)

    keep_prob = 1.0 - params.residual_dropout
    decoder_input = slim.dropout(decoder_input, keep_prob, is_training=is_training, scope='dec_input_dropout')

    decoder_outputs = AvgAttentionNet(decoder_input, params, pos=dec_pos_bias_fwd, is_training=is_training)

    return decoder_outputs

def sparse_gate_encoder(inputs, params, dtype=None, scope=None, is_training=True):
    with tf.variable_scope(scope, default_name="sparse_encoder", dtype=dtype,
                           values=[inputs]):
        x = inputs
        for layer in range(params.num_decoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    x_att = layers.attention.sparse_multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        None,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                    )
                    
                    # FFN activation
                    if params.use_ffn:
                        y = ffn_layer(
                            layer_process(x_att, params.layer_preprocess),
                            params.ffn_filter_size,
                            params.hidden_size,
                            1.0 - params.relu_dropout,
                            is_training=is_training
                        )
                    else:
                        y = x_att

                    # Gating layer
                    z = layers.nn.linear(tf.concat([x, y], axis=-1), 
                        params.hidden_size*2, True, True, scope="z_project")
                    i, f = tf.split(z, [params.hidden_size, params.hidden_size], axis=-1)
                    y = tf.sigmoid(i) * x + tf.sigmoid(f) * y
                    
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

        outputs = layer_process(x, params.layer_preprocess)

        return outputs

def sparse_encoder(inputs, params, dtype=None, scope=None, is_training=True):
    with tf.variable_scope(scope, default_name="sparse_encoder", dtype=dtype,
                           values=[inputs]):
        x = inputs
        for layer in range(params.num_decoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = layers.attention.sparse_multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        None,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                    )
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                        is_training=is_training
                    )
                    x = residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = layer_process(x, params.layer_postprocess)

        outputs = layer_process(x, params.layer_preprocess)

        return outputs

def sparse_self_attention(spec, labels, params, is_training=True):

    hidden_size = params.aan_size
    
    # concat spectrum and labels(no longer yet) as decoder input
    dec_input = slim.fully_connected(spec, hidden_size, scope='spec_dec_input')
    dec_input = slim.dropout(dec_input, 1-params.residual_dropout, is_training=is_training, scope='dec_input_dropout')

    # TODO: position embedding
    decoder_input = layers.attention.add_timing_signal(dec_input)

    keep_prob = 1.0 - params.residual_dropout
    decoder_input = slim.dropout(decoder_input, keep_prob, is_training=is_training, scope='dec_input_dropout')

    decoder_outputs = sparse_gate_encoder(decoder_input, params, is_training=is_training)
    #decoder_outputs = sparse_encoder(decoder_input, params, is_training=is_training)

    return decoder_outputs

def model_graph(features, labels, mode, params, given_memory=None, given_src_mask=None, given_decoder=None, given_position=None):
    hidden_size = params.hidden_size

    src_seq = features["source"]
    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    if given_src_mask is not None:
        src_mask = given_src_mask 
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=tf.float32)

    src_embedding, tgt_embedding, weights = get_weights(params)
    bias = tf.get_variable("bias", [hidden_size])

    # id => embedding
    # src_seq: [batch, max_src_length]
    # tgt_seq: [batch, max_tgt_length]
    inputs = tf.gather(src_embedding, src_seq) * (hidden_size ** 0.5)
    targets = tf.gather(tgt_embedding, tgt_seq) * (hidden_size ** 0.5)
    inputs = inputs * tf.expand_dims(src_mask, -1)
    targets = targets * tf.expand_dims(tgt_mask, -1)

    # Preparing encoder & decoder input
    encoder_input = tf.nn.bias_add(inputs, bias)
    encoder_input = layers.attention.add_timing_signal(encoder_input)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal")

    if given_decoder is not None:
        dec_pos_bias_fwd = tf.cumsum(tgt_mask, axis=1)
        dec_pos_bias_fwd = tf.where(tf.less_equal(dec_pos_bias_fwd, 0.), tf.ones_like(dec_pos_bias_fwd), dec_pos_bias_fwd)
        dec_pos_bias_fwd = tf.expand_dims(tf.cast(dec_pos_bias_fwd, tf.float32), 2)
    else:
        if params.aan_mask:
            dec_pos_bias_fwd = layers.attention.attention_bias(tgt_mask, "aan")
        else:
            dec_pos_bias_fwd = tf.cumsum(tgt_mask, axis=1)
            dec_pos_bias_fwd = tf.where(tf.less_equal(dec_pos_bias_fwd,0.), tf.ones_like(dec_pos_bias_fwd), dec_pos_bias_fwd)
            dec_pos_bias_fwd = tf.expand_dims(tf.cast(dec_pos_bias_fwd, tf.float32), 2)

    # Shift left
    # If given_decoder is not None, indicating a inference procedure,
    if given_decoder is not None:
        # given_position: starts from 1, a value greater than 1 means non-start position
        decoder_input = targets * tf.to_float(given_position > 1.)
        decoder_input = tf.tile(decoder_input, [1, tf.to_int32(given_position), 1])
    else:
        decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    # This is a lazy implementation, to assign the correct position embedding, I simply copy the
    # current decoder input 'given_position' times, and assign the whole position embeddings,
    # Only the last decoding value has the meaningful position embeddings
    # TODO: With Average Attention Network, Decoder side position embedding may be unnecessary 
    decoder_input = layers.attention.add_timing_signal(decoder_input)
    if given_decoder is not None:
        decoder_input = decoder_input[:, -1:, :]

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params, pos=[None])
    # Given memory indicates the source-side encoding output during inference
    if given_memory is not None:
        encoder_output = given_memory
    # During inference, the bias for decoder is exactly the decoding position 
    if given_position is not None:
        dec_pos_bias_fwd = given_position
    decoder_output, decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                         dec_attn_bias, enc_attn_bias, params, pos=[dec_pos_bias_fwd], given_inputs = given_decoder)

    # inference mode, take the last position
    if mode == "infer":
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)

        return logits
    # SEARCH
    elif mode == "encoder":
        return encoder_output, src_mask
    # Particularly for AAN decoding, we need the last decoding states for acceleration
    elif mode == "decoder":
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)

        return logits, decoder_outputs

    # [batch, length, channel] => [batch * length, vocab_size]
    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, weights, False, True)

    # label smoothing
    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(tgt_seq))
    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss


class Transformer(interface.NMTModel):
    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=tf.AUTO_REUSE):
                loss = model_graph(features, features["target"],
                                   "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.residual_dropout = 0.0
            params.attention_dropout = 0.0
            params.relu_dropout = 0.0
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                logits = model_graph(features, None, "infer", params)

            return logits

        return evaluation_fn

    def get_inference_func(self):
        def inference_encoder_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.residual_dropout = 0.0
            params.attention_dropout = 0.0
            params.relu_dropout = 0.0
            params.label_smoothing = 0.0

            # SEARCH
            with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
                encoder_output, src_mask = model_graph(features, None, "encoder", params)

            return encoder_output, src_mask

        def inference_decoder_fn(features, encoder_output, src_mask, decoder_output, position, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.residual_dropout = 0.0
            params.attention_dropout = 0.0
            params.relu_dropout = 0.0
            params.label_smoothing = 0.0

            # SEARCH
            with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
                logits, decoder_output = model_graph(features, None, "decoder", params, given_memory=encoder_output, given_src_mask=src_mask, given_decoder=decoder_output, given_position=position)

            return logits, decoder_output

        return inference_encoder_fn, inference_decoder_fn


    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate_decay="noam",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            aan_mask=True,
            use_ffn=False,
            avg_len=45
        )

        return params