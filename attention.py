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
        self.pos = tf.zeros([batch_sz, 1], name='attention_pos')
        self.att_len = int(att_len)
        self.batch_sz = batch_sz

    def call(self, query, values):

        def fetch_att_values(values, pos, batch_sz):
            for i in range(batch_sz):
                start = self.pos[i] - self.att_len/2
                end = self.pos[i] + self.att_len/2
                start = start if start > 0 else 0
                end = end if end < tf.shape(values)[0] else tf.shape(values)[0]
                values_slice = tf.slice(values, [start, i, 0], [end-start, 1, tf.shape(values)[2])
                



        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        #calc attention range
        start = self.pos - self.att_len/2
        end = self.pos + self.att_len/2
        start = start if start > 0 else 0
        end = end if end < tf.shape(values)[0] else tf.shape(values)[0]
        values_att = values[start:end]
        values_att = fetch_att_values(values, self.pos, self.batch_sz)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values_att) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values_att
        context_vector = tf.reduce_sum(context_vector, axis=1)

        #update attention position
        xpos = range(start, end)
        self.pos = int(tf.reduce_sum(tf.multiply(xpos, attention_weights)))

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, dec_units, batch_sz, att_len):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.fc = tf.keras.layers.Dense(dec_units)
        self.gru = gru(self.dec_units)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units, att_len)

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