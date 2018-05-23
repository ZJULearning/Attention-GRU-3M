"""
Time lstm
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import collections
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear
from tensorflow.python.ops.math_ops import sigmoid, tanh

class TLSTMCell(RNNCell):
    """Time LSTM based on LSTM
    """

    def __init__(self,
                 num_units,
                 use_peepholes=False, cell_clip=None,
                 initializer=None,
                 forget_bias=1.0,
                 activation=None, reuse=None):
        """Initialize the parameters for an TLSTM cell.
        """

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

        self._state_size = (LSTMStateTuple(num_units, num_units))
        self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        """Run one step of TLSTM.
        """
        sigmoid = math_ops.sigmoid
        tanh = math_ops.tanh

        (c_prev, m_prev) = state

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        batch_size, feature_size = inputs.get_shape().as_list()
        feature_size = feature_size - 1

        seq = tf.slice(inputs, begin=[0, 0], size=[batch_size, feature_size])
        delta_t = tf.slice(inputs, begin=[0, 48], size=[batch_size, 1])

        scope = scope or vs.get_variable_scope()
        with vs.variable_scope(scope, initializer=self._initializer) as unit_scope:

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate

            lstm_matrix = _linear([seq, m_prev], output_size=4 * self._num_units, bias=True)

            # Time gate
            with vs.variable_scope(unit_scope) as time_gate_scope:
                w_t1 = vs.get_variable(
                    "w_t1", shape=[1, self._num_units], dtype=dtype)
                bias_t1 = vs.get_variable(
                    "bias_t1", [self._num_units], dtype=dtype,
                    initializer=init_ops.constant_initializer(0.0, dtype=dtype))
                w_tx1 = vs.get_variable(
                    "w_tx1", shape=[feature_size, self._num_units], dtype=dtype)
                w_tx2 = vs.get_variable(
                    "w_tx2", shape=[feature_size, self._num_units], dtype=dtype)
                w_t2 = vs.get_variable(
                    "w_t2", shape=[1, self._num_units], dtype=dtype)
                bias_t2 = vs.get_variable(
                    "bias_t2", [self._num_units], dtype=dtype,
                    initializer=init_ops.constant_initializer(0.0, dtype=dtype))
                w_to = vs.get_variable(
                    "w_to", shape=[1, self._num_units], dtype=dtype)

            w_t1_with_constraint = tf.minimum(w_t1, 0)
            t1_act = (self._activation(math_ops.matmul(delta_t, w_t1_with_constraint)) +
                      math_ops.matmul(seq, w_tx1) + bias_t1)
            t2_act = (self._activation(math_ops.matmul(delta_t, w_t2)) +
                      math_ops.matmul(seq, w_tx2) + bias_t2)
            t1 = sigmoid(t1_act)
            t2 = sigmoid(t2_act)


            i, j, f, o = array_ops.split(
                value=lstm_matrix, num_or_size_splits=4, axis=1)

            # Diagonal connections
            if self._use_peepholes:
                with vs.variable_scope(unit_scope) as projection_scope:
                    w_f_diag = vs.get_variable(
                        "w_f_diag", shape=[self._num_units], dtype=dtype)
                    w_i_diag = vs.get_variable(
                        "w_i_diag", shape=[self._num_units], dtype=dtype)
                    w_o_diag = vs.get_variable(
                        "w_o_diag", shape=[self._num_units], dtype=dtype)

            if self._use_peepholes:
                c_hat = ((1 - sigmoid(i + w_i_diag * c_prev)*t1) * c_prev +
                         sigmoid(i + w_i_diag * c_prev)*t1 * self._activation(j))
                c = ((1 - sigmoid(i + w_i_diag * c_prev)) * c_prev +
                     sigmoid(i + w_i_diag * c_prev)*t2 * self._activation(j))
            else:
                c_hat = ((1 - sigmoid(i)) * c_prev +
                         sigmoid(i + w_i_diag * c_prev)*t1 * self._activation(j))
                c = ((1 - sigmoid(i)) * c_prev +
                     sigmoid(i + w_i_diag * c_prev)*t2 * self._activation(j))

            if self._cell_clip is not None:
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
            if self._use_peepholes:
                m = (sigmoid(o + math_ops.matmul(delta_t, w_to) + w_o_diag * c) *
                     self._activation(c_hat))
            else:
                m = sigmoid(o + math_ops.matmul(delta_t, w_to)) * self._activation(c_hat)

        new_state = (LSTMStateTuple(c, m))
        return m, new_state

_TGRUStateTuple = collections.namedtuple("TGRUStateTuple", ("c", "h"))
class TGRUStateTuple(_TGRUStateTuple):
    """Tuple used by TGRU Cells for `state_size`, `zero_state`, and output state.
       Stores two elements: `(c, h)`, in that order.
       Only used when `state_is_tuple=True` .
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if not c.dtype == h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" % (str(c.dtype), str(h.dtype)))
        return c.dtype

class TGRUCell(RNNCell):
    """Gated Recurrent Unit cell with time gate"""

    def __init__(self, num_units, use_tgate=False, input_size=None, activation=tanh, reuse=None):
        if input_size is not None:
            tf.logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._reuse = reuse
        self._use_tgate = use_tgate
        self._state_size = (TGRUStateTuple(num_units, num_units))


    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._num_units * 2

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        dtype = inputs.dtype
        batch_size, feature_size = inputs.get_shape().as_list()
        if self._use_tgate:
            # Time gate
            feature_size = feature_size - 1
            tvscope = vs.get_variable_scope()
            with vs.variable_scope(tvscope, initializer=None) as unit_scope:
                with vs.variable_scope(unit_scope) as time_gate_scope:
                    w_t1 = vs.get_variable(
                        "w_t1", shape=[1, self._num_units], dtype=dtype)
                    bias_t1 = vs.get_variable(
                        "bias_t1", [self._num_units], dtype=dtype,
                        initializer=init_ops.constant_initializer(0.0, dtype=dtype))
                    w_tx1 = vs.get_variable(
                        "w_tx1", shape=[feature_size, self._num_units], dtype=dtype)
                seq = tf.slice(inputs, begin=[0, 0], size=[batch_size, feature_size])
                delta_t = tf.slice(inputs, begin=[0, 56], size=[batch_size, 1])


                t1_act = (self._activation(math_ops.matmul(delta_t, w_t1)) +
                          math_ops.matmul(seq, w_tx1) + bias_t1)
                t1 = sigmoid(t1_act)
                inputs = seq
        # for initial state
        (state, state_decay) = state
        with vs.variable_scope("gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            value = sigmoid(_linear(
                [inputs, state], 2 * self._num_units, True, 1.0))
            r, u = array_ops.split(value=value,
                                   num_or_size_splits=2,
                                   axis=1)
        with vs.variable_scope("candidate"):
            c = self._activation(_linear([inputs, r * state],
                                         self._num_units, True))
        new_h = u * state + (1 - u) * c

        if self._use_tgate:
            new_h_decay = u * t1 * state_decay + (1 - u * t1) * c
            new_state = (new_h, new_h_decay)
            new_state = (TGRUStateTuple(new_h, new_h_decay))
            new_h = tf.concat([new_h, new_h_decay], axis=1)
        else:
            new_state = (new_h, new_h)
            new_state = (TGRUStateTuple(new_h, new_h))

        return new_h, new_state
