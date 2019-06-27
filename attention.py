#
#
#  use attention mechenism to do piano transcription
# 

"""attention part"""

import constants
import tensorflow as tf
import tensorflow.contrib.slim as slim

def gru(units):
    # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
    # the code automatically does that.
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                    return_sequences=True, 
                                    return_state=True, 
                                    recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units, 
                               return_sequences=True, 
                               return_state=True, 
                               recurrent_activation='sigmoid', 
                               recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.fc = tf.keras.layers.Dense(enc_units)
        self.gru = gru(self.enc_units)

    def call(self, x, hidden):
        x = self.fc(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
    
class BahdanauAttention(tf.keras.Model):
    def __init__(self, batch_sz, units, att_len):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.pos = tf.zeros([batch_sz], dtype=tf.int32, name='attention_pos')
        self.att_len = att_len
        self.batch_sz = batch_sz
        self.units = units

    def fetch_att_values(self, values, att_len, pos, batch_sz, units_num):
        def cond(i, att_value, xpos):
            return tf.less(i, batch_sz)

        def body(i, att_value, xpos):
            # take self.pos as the center, and att_len as the range 
            start = tf.subtract(pos[i], tf.div(self.att_len, 2))
            end = tf.add(pos[i], tf.div(self.att_len, 2))
            start = tf.cond(tf.greater(start, 0), 
                            lambda: start, 
                            lambda: 0)
            end = tf.cond(tf.less(end, tf.shape(values)[0]), 
                            lambda: end, 
                            lambda: tf.shape(values)[0])
            xlen = tf.subtract(end, start)

            # slice the enc_output around of self.pos
            # enc_output shape == (batch_size, max_length, hidden_size)
            values_slice = tf.slice(values, [i, start, 0], [1, xlen, units_num])
            padding = lambda: tf.concat([values_slice, tf.zeros([1, self.att_len-xlen, units_num]) ], axis=1)
            # padding zeros if the length less than att_len
            values_slice = tf.cond(tf.equal(xlen, self.att_len), 
                                    lambda: values_slice, 
                                    padding)
            # concat the attention values along batch dim
            att_value = tf.cond(tf.equal(i, 0), 
                                lambda: tf.identity(values_slice), 
                                lambda: tf.concat([att_value, values_slice], axis=0))
            
            # get x asix numbers from start to end
            xranges = tf.cast(tf.range(start, end), dtype=tf.float32)
            xranges = tf.cond(tf.equal(xlen, self.att_len), 
                                lambda: xranges,
                                lambda: tf.concat([xranges, tf.zeros([self.att_len-xlen], dtype=tf.float32)], axis=0))
            xranges = tf.expand_dims(xranges, 0) # add batch dim
            # concat along batch dim
            xpos = tf.cond(tf.equal(i, 0),
                            lambda: tf.identity(xranges),
                            lambda: tf.concat([xpos, xranges], axis=0))
            i = tf.add(i, 1)
            return  i, att_value, xpos

        zero = lambda: tf.constant(0, dtype=tf.int32)
        zeros= lambda: tf.zeros([self.att_len, 1, self.units], dtype=tf.float32)
        zeros2= lambda: tf.zeros([1, self.att_len], dtype=tf.float32)
        i = tf.Variable(initial_value=zero, dtype=tf.int32)
        att_value = tf.Variable(initial_value=zeros, dtype=tf.float32)
        xpos = tf.Variable(initial_value=zeros2, dtype=tf.float32)
        print("shape1= ", tf.TensorShape([None, units_num, units_num]))
        print("shape2= ", tf.TensorShape([None, att_len]))
        i, att_value, xpos = tf.while_loop(cond, body, loop_vars=[i, att_value, xpos], 
            shape_invariants=[i.get_shape(), tf.TensorShape([None, att_len, units_num]), tf.TensorShape([None, att_len])])

        return  att_value, xpos

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # slice attention values from enc_output(values), xpos is the time index
        values_att, xpos = self.fetch_att_values(values, self.att_len, self.pos, self.batch_sz, self.units)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values_att) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # enc_output shape == (batch_size, max_length, hidden_size)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = tf.multiply(attention_weights, values_att)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # update attention center position
        # xpos shape == (batch_size, max_length)
        self.pos = tf.cast(tf.reduce_sum(tf.multiply(attention_weights, xpos), axis=1), dtype=tf.int32)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, dec_units, batch_sz, att_len):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.fc = tf.keras.layers.Dense(dec_units)
        self.gru = gru(self.dec_units)

        # used for attention
        self.attention = BahdanauAttention(batch_sz=batch_sz, units=self.dec_units, att_len=att_len)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.fc(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        return output, state, attention_weights