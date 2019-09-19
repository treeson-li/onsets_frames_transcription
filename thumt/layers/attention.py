# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from thumt.layers.nn import linear


def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4, name=None):
    """
    This function adds a bunch of sinusoids of different frequencies to a
    Tensor. See paper: Attention is all you need

    :param x: A tensor with shape [batch, length, channels]
    :param min_timescale: A floating point number
    :param max_timescale: A floating point number
    :param name: An optional string

    :returns: a Tensor the same shape as x.
    """

    with tf.name_scope(name, default_name="add_timing_signal", values=[x]):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )

        scaled_time = (tf.expand_dims(position, 1) *
                       tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])

        return x + signal


def split_heads(inputs, num_heads, name=None, mode=None, cxt=None, params=None):
    """ Split heads
    :param inputs: A tensor with shape [batch, length, channels]
    :param num_heads: An integer
    :param name: An optional string
    :returns: A tensor with shape [batch, heads, length, channels / heads]
    """

    with tf.name_scope(name, default_name="split_heads", values=[inputs]):
        x = inputs
        n = num_heads
        old_shape = x.get_shape().dims

        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        ret = tf.transpose(ret, [0, 2, 1, 3])
            
        if mode=='local_q':
            ret = tf.expand_dims(ret, axis=3)
        
        if mode=='local_kv':
            ret = get_local_context(ret, cxt, params)

        return ret
        
def get_local_context(inputs, cxt, params, name=None):
    with tf.name_scope(name, default_name="get_local_context", values=[inputs, cxt]):
        s = tf.shape(inputs)
        zeros = tf.zeros([s[0], s[1], cxt-1, s[3]])
        x0 = tf.concat([zeros, inputs], axis=2)
        length = s[2]
        dim = int(params.hidden_size / params.num_heads)

        def cond(i, x):
            return tf.less(i, length)

        def body(i, x):
            sliced = tf.slice(x0, [0, 0, i, 0], [params.batch_size, params.num_heads, cxt, dim])
            sliced = tf.expand_dims(sliced, axis=2)
            x = tf.cond(tf.equal(i, 0),
                        lambda: sliced,
                        lambda: tf.concat([x, sliced], axis=2))
            i = tf.add(i, 1)
            return  i, x

        zero = lambda: tf.constant(0, dtype=tf.int32)
        x = tf.zeros([params.batch_size, params.num_heads, 1, cxt, dim], dtype=tf.float32, name='concat_local_values')
        i = tf.Variable(initial_value=zero, dtype=tf.int32)
        i, x = tf.while_loop(cond, body, loop_vars=[i, x],
            shape_invariants=[i.get_shape(), tf.TensorShape([params.batch_size, params.num_heads, None, params.local_cxt, dim])])

        return x
            
def combine_heads(inputs, name=None):
    """ Combine heads
    :param inputs: A tensor with shape [batch, heads, length, channels]
    :param name: An optional string
    :returns: A tensor with shape [batch, length, heads * channels]
    """

    with tf.name_scope(name, default_name="combine_heads", values=[inputs]):
        x = inputs
        x = tf.transpose(x, [0, 2, 1, 3])
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        x.set_shape(new_shape)

        return x


def attention_bias(inputs, mode, inf=-1e9, name=None):
    """ A bias tensor used in attention mechanism
    :param inputs:
    :param mode:
    :param inf:
    :param name:
    :returns:
    """

    with tf.name_scope(name, default_name="attention_bias", values=[inputs]):
        if mode == "causal":
            length = inputs
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            ret = inf * (1.0 - lower_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        elif mode == "masking":
            mask = inputs
            ret = (1.0 - mask) * inf
            return tf.expand_dims(tf.expand_dims(ret, 1), 1)
        elif mode == "aan":
            length = tf.shape(inputs)[1]
            diagonal = tf.eye(length)
            cum_factor = tf.expand_dims(tf.cumsum(diagonal, axis=0), 0)
            mask = tf.expand_dims(inputs, 1) * tf.expand_dims(inputs, 2)
            mask *= cum_factor
            weight = tf.nn.softmax(mask + (1.0 - mask) * inf)
            weight *= mask
            return weight
        elif mode == "proximal":
            length = inputs
            r = tf.to_float(tf.range(length))
            diff = tf.expand_dims(r, 0) - tf.expand_dims(r, 1)
            m = tf.expand_dims(tf.expand_dims(-tf.log(1 + tf.abs(diff)), 0), 0)
            return m
        elif mode == "distance":
            length, distance = inputs
            distance = tf.where(distance > length, 0, distance)
            distance = tf.cast(distance, tf.int64)
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            mask_triangle = 1.0 - tf.matrix_band_part(
                tf.ones([length, length]), distance - 1, 0
            )
            ret = inf * (1.0 - lower_triangle + mask_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        else:
            raise ValueError("Unknown mode %s" % mode)


def attention(query, memories, bias, hidden_size, cache=None, reuse=None,
              dtype=None, scope=None):
    """ Standard attention layer

    :param query: A tensor with shape [batch, key_size]
    :param memories: A tensor with shape [batch, memory_size, key_size]
    :param bias: A tensor with shape [batch, memory_size]
    :param hidden_size: An integer
    :param cache: A dictionary of precomputed value
    :param reuse: A boolean value, whether to reuse the scope
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string, the scope of this layer
    :return: A tensor with shape [batch, value_size] and
        a Tensor with shape [batch, memory_size]
    """

    with tf.variable_scope(scope or "attention", reuse=reuse,
                           values=[query, memories, bias], dtype=dtype):
        mem_shape = tf.shape(memories)
        key_size = memories.get_shape().as_list()[-1]

        if cache is None:
            k = tf.reshape(memories, [-1, key_size])
            k = linear(k, hidden_size, False, False, scope="k_transform")

            if query is None:
                return {"key": k}
        else:
            k = cache["key"]

        q = linear(query, hidden_size, False, False, scope="q_transform")
        k = tf.reshape(k, [mem_shape[0], mem_shape[1], hidden_size])

        hidden = tf.tanh(q[:, None, :] + k)
        hidden = tf.reshape(hidden, [-1, hidden_size])

        # Shape: [batch, mem_size, 1]
        logits = linear(hidden, 1, False, False, scope="logits")
        logits = tf.reshape(logits, [-1, mem_shape[1]])

        if bias is not None:
            logits = logits + bias

        alpha = tf.nn.softmax(logits)

        outputs = {
            "value": tf.reduce_sum(alpha[:, :, None] * memories, axis=1),
            "weight": alpha
        }

    return outputs


def additive_attention(queries, keys, values, bias, hidden_size, concat=False,
                       keep_prob=None, dtype=None, scope=None):
    """ Additive attention mechanism. This layer is implemented using a
        one layer feed forward neural network

    :param queries: A tensor with shape [batch, heads, length_q, depth_k]
    :param keys: A tensor with shape [batch, heads, length_kv, depth_k]
    :param values: A tensor with shape [batch, heads, length_kv, depth_v]
    :param bias: A tensor
    :param hidden_size: An integer
    :param concat: A boolean value. If ``concat'' is set to True, then
        the computation of attention mechanism is following $tanh(W[q, k])$.
        When ``concat'' is set to False, the computation is following
        $tanh(Wq + Vk)$
    :param keep_prob: a scalar in [0, 1]
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string, the scope of this layer

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, length_q]
        outputs: A tensor with shape [batch, length_q, depth_v]
    """

    with tf.variable_scope(scope, default_name="additive_attention",
                           values=[queries, keys, values, bias], dtype=dtype):
        length_q = tf.shape(queries)[2]
        length_kv = tf.shape(keys)[2]
        q = tf.tile(tf.expand_dims(queries, 3), [1, 1, 1, length_kv, 1])
        k = tf.tile(tf.expand_dims(keys, 2), [1, 1, length_q, 1, 1])

        if concat:
            combined = tf.tanh(linear(tf.concat([q, k], axis=-1), hidden_size,
                                      True, True, name="qk_transform"))
        else:
            q = linear(queries, hidden_size, True, True, name="q_transform")
            k = linear(keys, hidden_size, True, True, name="key_transform")
            combined = tf.tanh(q + k)

        # shape: [batch, heads, length_q, length_kv]
        logits = tf.squeeze(linear(combined, 1, True, True, name="logits"),
                            axis=-1)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")

        if keep_prob or keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)

        outputs = tf.matmul(weights, values)

        return {"weights": weights, "outputs": outputs}


def multiplicative_attention(queries, keys, values, bias, keep_prob=None,
                             name=None, is_training=True):
    """ Multiplicative attention mechanism. This layer is implemented using
        dot-product operation.

    :param queries: A tensor with shape [batch, heads, length_q, depth_k]
    :param keys: A tensor with shape [batch, heads, length_kv, depth_k]
    :param values: A tensor with shape [batch, heads, length_kv, depth_v]
    :param bias: A tensor
    :param keep_prob: a scalar in (0, 1]
    :param name: the name of this operation

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, length_q]
        outputs: A tensor with shape [batch, length_q, depth_v]
    """

    with tf.name_scope(name, default_name="multiplicative_attention",
                       values=[queries, keys, values, bias]):
        # shape: [batch, heads, length_q, length_kv]
        logits = tf.matmul(queries, keys, transpose_b=True)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")

        if is_training and keep_prob is not None and keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)

        outputs = tf.matmul(weights, values)

        return {"weights": weights, "outputs": outputs}


def multihead_attention(queries, memories, bias, num_heads, key_size,
                        value_size, output_size, keep_prob=None, output=True,
                        dtype=None, scope=None, is_training=True):
    """ Multi-head scaled-dot-product attention with input/output
        transformations.

    :param queries: A tensor with shape [batch, length_q, depth_q] if
    :param memories: A tensor with shape [batch, length_m, depth_m]
    :param bias: A tensor (see attention_bias)
    :param num_heads: An integer dividing key_size and value_size
    :param key_size: An integer
    :param value_size: An integer
    :param output_size: An integer
    :param keep_prob: A floating point number in (0, 1]
    :param output: Whether to use output transformation
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, length_q]
        outputs: A tensor with shape [batch, length_q, depth_v]
    """

    if key_size % num_heads != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_size, num_heads))

    if value_size % num_heads != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_size, num_heads))

    with tf.variable_scope(scope, default_name="multihead_attention",
                           values=[queries, memories], dtype=dtype):
        if memories is None:
            # self attention
            size = key_size * 2 + value_size
            combined = linear(queries, size, True, True, scope="qkv_transform")
            q, k, v = tf.split(combined, [key_size, key_size, value_size],
                               axis=-1)
        else:
            q = linear(queries, key_size, True, True, scope="q_transform")
            combined = linear(memories, key_size + value_size, True,
                              scope="kv_transform")
            k, v = tf.split(combined, [key_size, value_size], axis=-1)

        # split heads
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)

        # scale query
        key_depth_per_head = key_size // num_heads
        q *= key_depth_per_head ** -0.5

        # attention
        results = multiplicative_attention(q, k, v, bias, keep_prob, is_training=is_training)

        # combine heads 
        weights = results["weights"]
        x = combine_heads(results["outputs"])

        if output:
            outputs = linear(x, output_size, True, True,
                             scope="output_transform")
        else:
            outputs = x

        return {"weights": weights, "outputs": outputs}

def get_position_context(inputs, cxt, idx, len, params, mode, name=None):
    with tf.name_scope(name, default_name="get_position_context", values=[inputs, cxt]):

        if inputs == None:
            return None
        s = tf.shape(inputs)
#        print('-'*50)
#        print('inputs shape', s)
#        print('idx, len', idx, len)
        if mode == 'cxt_qb':
            x = tf.slice(inputs, [0, 0, idx, 0], [s[0], s[1], len, s[3]])
            x = tf.expand_dims(x, axis=3)
            return  x

        if mode is not 'cxt_kv':
            return None

        zeros = tf.zeros([s[0], s[1], cxt-1, s[3]], dtype=tf.float32)
        x0 = tf.concat([zeros, inputs], axis=2)
        dim = int(params.hidden_size / params.num_heads)

        def cond(i, x):
            return tf.less(i, len)

        def body(i, x):
            sliced = tf.slice(x0, [0, 0, idx+i, 0], [params.batch_size, params.num_heads, cxt, dim])
            sliced = tf.expand_dims(sliced, axis=2)
            x = tf.cond(tf.equal(i, 0),
                        lambda: sliced,
                        lambda: tf.concat([x, sliced], axis=2))
            i = tf.add(i, 1)
            return  i, x

        x = tf.zeros([params.batch_size, params.num_heads, 1, cxt, dim], dtype=tf.float32, name='concat_pos_context')
        i = tf.constant(0, dtype=tf.int32)
        i, x = tf.while_loop(cond, body, loop_vars=[i, x],
            shape_invariants=[i.get_shape(), tf.TensorShape([params.batch_size, params.num_heads, None, cxt, dim])])

        return x

def multihead_sparse_attention(queries, memories, bias, num_heads, key_size,
                        value_size, output_size, keep_prob=None, output=True,
                        dtype=None, scope=None, cxt=None, is_training=True, params=None):
    """ Multi-head scaled-dot-product attention with input/output
        transformations.

    :param queries: A tensor with shape [batch, length_q, depth_q] if
    :param memories: A tensor with shape [batch, length_m, depth_m]
    :param bias: A tensor (see attention_bias)
    :param num_heads: An integer dividing key_size and value_size
    :param key_size: An integer
    :param value_size: An integer
    :param output_size: An integer
    :param keep_prob: A floating point number in (0, 1]
    :param output: Whether to use output transformation
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, length_q]
        outputs: A tensor with shape [batch, length_q, depth_v]
    """

    if key_size % num_heads != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_size, num_heads))

    if value_size % num_heads != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_size, num_heads))

    with tf.variable_scope(scope, default_name="multihead_sparse_attention",
                           values=[queries, memories], dtype=dtype):
        if memories is None:
            # self attention
            size = key_size * 2 + value_size
            combined = linear(queries, size, True, True, scope="qkv_transform")
            q, k, v = tf.split(combined, [key_size, key_size, value_size],
                               axis=-1)
        else:
            q = linear(queries, key_size, True, True, scope="q_transform")
            combined = linear(memories, key_size + value_size, True,
                              scope="kv_transform")
            k, v = tf.split(combined, [key_size, value_size], axis=-1)

        # split heads
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)

        # scale query
        key_depth_per_head = key_size // num_heads
        q *= key_depth_per_head ** -0.5

        block_num = cxt

        def get_sliced_qkv(q, k, v, bias, idx, num):
            # q, k, v shape [batch, head, length, dim]
            shape = tf.shape(q)
            length = shape[2]
            width = tf.cast(tf.ceil(tf.divide(length, num)), tf.int32)
            start = width * idx
            width = tf.cond(tf.less(idx, num-1),
                            lambda: width,
                            lambda: length - width * (num-1))
            q1 = get_position_context(q, cxt, start, width, params, mode='cxt_qb')
            k1 = get_position_context(k, cxt, start, width, params, mode='cxt_kv')
            v1 = get_position_context(v, cxt, start, width, params, mode='cxt_kv')
            b1 = get_position_context(bias, cxt, start, width, params, mode='cxt_qb')

            return  q1, k1, v1, b1
            

        def cond(weights, outputs, i):
            return tf.less(i, block_num)

        def body(weights, outputs, i):
            q1, k1, v1, bias1 = get_sliced_qkv(q, k, v, bias, i, block_num)
            
            # attention
            results = multiplicative_attention(q1, k1, v1, bias1, keep_prob, is_training=is_training)

            wt = tf.squeeze(results["weights"], squeeze_dims=3)
            ot = tf.squeeze(results["outputs"], squeeze_dims=3)
            weights = tf.cond(tf.equal(i, 0),
                        lambda: wt,
                        lambda: tf.concat([weights, wt], axis=2))
            outputs = tf.cond(tf.equal(i, 0),
                        lambda: ot,
                        lambda: tf.concat([outputs, ot], axis=2))
            i = i + 1
            return weights, outputs, i

        weights = tf.zeros([params.batch_size, params.num_heads, 1, cxt], dtype=tf.float32)
        outputs = tf.zeros([params.batch_size, params.num_heads, 1, params.hidden_size//params.num_heads], dtype=tf.float32)
        i = tf.constant(0, dtype=tf.int32)
        weights, outputs, i = tf.while_loop(cond, body, loop_vars=[weights, outputs, i],
                            shape_invariants = [tf.TensorShape([params.batch_size, params.num_heads, None, cxt]), 
                                                tf.TensorShape([params.batch_size, params.num_heads, None, params.hidden_size//params.num_heads]), 
                                                i.get_shape()])
        # combine heads 
        x = combine_heads(outputs)

        if output:
            outputs = linear(x, output_size, True, True,
                             scope="output_transform")
        else:
            outputs = x

        return {"weights": weights, "outputs": outputs}
