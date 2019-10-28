# code from blocksparse

import numpy as np
import tensorflow as tf

def split_heads(x, n):
    return tf.transpose(split_states(x, n), [0, 2, 1, 3])

def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))

def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def get_attn_mask(n, attn_mode, local_attn_ctx=None):    
    if attn_mode == 'all':
        b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = tf.minimum(n - 1, bandwidth - 1)
        b = tf.matrix_band_part(tf.ones([n, n]), ctx, 0)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = tf.reshape(tf.range(n, dtype=tf.int32), [n, 1])
        y = tf.transpose(x)
        z = tf.zeros([n, n], dtype=tf.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = tf.equal(tf.floormod(q - k, stride), 0)
        c3 = tf.logical_and(c1, c2)
        b = tf.cast(c3, tf.float32)
    elif attn_mode == 'diag_band':
        bandwidth = local_attn_ctx
        ctx = tf.minimum(n - 1, bandwidth - 1)//2
        b = tf.matrix_band_part(tf.ones([n, n]), ctx, ctx)
    elif attn_mode == 'full':
        b = tf.ones([n, n])
    else:
        raise ValueError('Not yet implemented')
    b = tf.reshape(b, [1, 1, n, n])
    return b

def attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None, keep_prob=None):
    q = split_heads(q, heads)
    k = split_heads(k, heads)
    v = split_heads(v, heads)
    n_timesteps = shape_list(k)[2]
    mask = tf.to_float(get_attn_mask(n_timesteps, attn_mode, local_attn_ctx))
    w = tf.matmul(q, k, transpose_b=True)
    scale_amount = 1.0 / np.sqrt(shape_list(q)[-1])
    orig_dtype = q.dtype 
    if orig_dtype == tf.float16:
        w = tf.cast(w, tf.float32)
    w = w * scale_amount
    w = w * mask + -1e9 * (1 - mask)
    w = tf.nn.softmax(w)
    w = tf.cast(w, orig_dtype)
    if keep_prob is not None and keep_prob < 1.0:
        w = tf.nn.dropout(w, keep_prob)
    a = tf.matmul(w, v)
    a = merge_heads(a)
    return a