# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.ifelse
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros, sharedX, alloc_zeros_matrix
from ..layers.core import Layer
from six.moves import range

class SimpleRNN(Layer):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model, 
        included for demonstration purposes 
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, input_dim, output_dim, 
        init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
        truncate_gradient=-1, return_sequences=False):
        super(SimpleRNN,self).__init__()
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.input = T.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.U = self.init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W, self.U, self.b]

        if weights is not None:
            self.set_weights(weights)

    def _step(self, x_t, h_tm1, u):
        '''
            Variable names follow the conventions from: 
            http://deeplearning.net/software/theano/library/scan.html

        '''
        return self.activation(x_t + T.dot(h_tm1, u))

    def get_output(self, train):
        X = self.get_input(train) # shape: (nb_samples, time (padded with zeros at the end), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        X = X.dimshuffle((1,0,2)) 

        x = T.dot(X, self.W) + self.b
        
        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        outputs, updates = theano.scan(
            self._step, # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=x, # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=self.U, # static inputs to _step
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}


class SimpleDeepRNN(Layer):
    '''
        Fully connected RNN where the output of multiple timesteps 
        (up to "depth" steps in the past) is fed back to the input:

        output = activation( W.x_t + b + inner_activation(U_1.h_tm1) + inner_activation(U_2.h_tm2) + ... )

        This demonstrates how to build RNNs with arbitrary lookback. 
        Also (probably) not a super useful model.
    '''
    def __init__(self, input_dim, output_dim, depth=3,
        init='glorot_uniform', inner_init='orthogonal', 
        activation='sigmoid', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_sequences=False):
        super(SimpleDeepRNN,self).__init__()
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.depth = depth
        self.return_sequences = return_sequences
        self.input = T.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.Us = [self.inner_init((self.output_dim, self.output_dim)) for _ in range(self.depth)]
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W] + self.Us + [self.b]

        if weights is not None:
            self.set_weights(weights)

    def _step(self, *args):
        o = args[0]
        for i in range(1, self.depth+1):
            o += self.inner_activation(T.dot(args[i], args[i+self.depth]))
        return self.activation(o)

    def get_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2)) 

        x = T.dot(X, self.W) + self.b
        
        outputs, updates = theano.scan(
            self._step,
            sequences=x,
            outputs_info=[dict(
                initial=T.alloc(np.cast[theano.config.floatX](0.), self.depth, X.shape[1], self.output_dim),
                taps = [(-i-1) for i in range(self.depth)]
            )],
            non_sequences=self.Us,
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "depth":self.depth,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "inner_activation":self.inner_activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}



class GRU(Layer):
    '''
        Gated Recurrent Unit - Cho et al. 2014

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    '''
    def __init__(self, input_dim, output_dim=128, 
        init='glorot_uniform', inner_init='orthogonal',
        activation='sigmoid', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_sequences=False):

        super(GRU,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim)) 
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self, 
        xz_t, xr_t, xh_t, 
        h_tm1, 
        u_z, u_r, u_h):
        z = self.inner_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * hh_t
        return h_t

    def get_output(self, train):
        X = self.get_input(train) 
        X = X.dimshuffle((1,0,2)) 

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step, 
            sequences=[x_z, x_r, x_h], 
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "inner_activation":self.inner_activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}



class LSTM(Layer):
    '''
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, input_dim, output_dim=128, 
        init='glorot_uniform', inner_init='orthogonal', 
        activation='tanh', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_sequences=False):
    
        super(LSTM,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_i = self.init((self.input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((self.input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = shared_zeros((self.output_dim))

        self.W_c = self.init((self.input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((self.input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self, 
        xi_t, xf_t, xo_t, xc_t, 
        h_tm1, c_tm1, 
        u_i, u_f, u_o, u_c): 
        i_t = self.inner_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def get_output(self, train):
        X = self.get_input(train) 
        X = X.dimshuffle((1,0,2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o
        
        [outputs, memories], updates = theano.scan(
            self._step, 
            sequences=[xi, xf, xo, xc],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ], 
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c], 
            truncate_gradient=self.truncate_gradient 
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "inner_activation":self.inner_activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}
        
# added by zhaowuxia begins
# a layer of num_blocks single-cell LSTM blocks, [output of block[i], x] feeds block[i+1]
# each block has its own forget gate
class DEEPLSTM(Layer):
    '''
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if return_seq_num > 1:
            (nb_samples, return_seq_num, output_dim*num_blocks)
        else:
            (nb_samples, output_dim*num_blocks)
        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, input_dim, output_dim=128*1, 
        init='glorot_uniform', inner_init='orthogonal', 
        activation='tanh', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_seq_num=1, 
        num_blocks=1):
    
        super(DEEPLSTM,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_seq_num = return_seq_num
        self.num_blocks = num_blocks

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_i = self.init((self.num_blocks, self.input_dim, self.output_dim))
        self.U_i = self.inner_init((self.num_blocks, self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.num_blocks, self.output_dim))

        self.W_f = self.init((self.num_blocks, self.input_dim, self.output_dim))
        self.U_f = self.inner_init((self.num_blocks, self.output_dim, self.output_dim))
        # large initialization of forget gate is better
        #self.b_f = sharedX(np.ones((self.num_blocks, self.output_dim))*5)
        self.b_f = shared_zeros((self.num_blocks, self.output_dim))

        self.W_c = self.init((self.num_blocks, self.input_dim, self.output_dim))
        self.U_c = self.inner_init((self.num_blocks, self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.num_blocks, self.output_dim))

        self.W_o = self.init((self.num_blocks, self.input_dim, self.output_dim))
        self.U_o = self.inner_init((self.num_blocks, self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.num_blocks, self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step1(self, 
        xi_t, xf_t, xo_t, xc_t, 
        h_tm1, c_tm1, 
        u_i, u_f, u_o, u_c):
        
        i_t = self.inner_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        
        return h_t, c_t
    
    def _step2(self, 
        xi_t, xf_t, xo_t, xc_t, in_t,
        h_tm1, c_tm1, 
        u_i, u_f, u_o, u_c):
        
        i_t = self.inner_activation(xi_t + T.dot(in_t, u_i))
        f_t = self.inner_activation(xf_t + T.dot(in_t, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(in_t, u_c))
        o_t = self.inner_activation(xo_t + T.dot(in_t, u_o))
        h_t = o_t * self.activation(c_t)
        
        return h_t, c_t

    def get_output(self, train):
        X = self.get_input(train) 
        X = X.dimshuffle((1,0,2)) #[T, sz, input_dim]
        
        outputs = []

        for i in range(self.num_blocks):
            xi = T.dot(X, self.W_i[i]) + self.b_i[i]
            xf = T.dot(X, self.W_f[i]) + self.b_f[i]
            xc = T.dot(X, self.W_c[i]) + self.b_c[i]
            xo = T.dot(X, self.W_o[i]) + self.b_o[i]
       
            if i == 0:
                [output, memories], updates = theano.scan(
                    self._step1, 
                    sequences=[xi, xf, xo, xc],
                    outputs_info=[
                        T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                        T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
                    ], 
                    non_sequences=[self.U_i[i], self.U_f[i], self.U_o[i], self.U_c[i]], 
                    truncate_gradient=self.truncate_gradient 
                )
                outputs.append(output)
            else:
                [output, memories], updates = theano.scan(
                    self._step2, 
                    sequences=[xi, xf, xo, xc, outputs[i-1]],
                    outputs_info=[
                        T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                        T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
                    ], 
                    non_sequences=[self.U_i[i], self.U_f[i], self.U_o[i], self.U_c[i]], 
                    truncate_gradient=self.truncate_gradient 
                )
                outputs.append(output)
        
        outputs = T.concatenate(outputs, axis=-1) #[T, sz, output_dim * num_blocks]
        if self.return_seq_num <= 0:
            return outputs.dimshuffle((1,0,2)) #[sz, T, out * nb]
        elif self.return_seq_num > 1:
            return outputs[-self.return_seq_num:].dimshuffle((1,0,2)) #[sz, return_seq_num, out * nb]
        else:
            return outputs[-1] #[sz, output_dim * num_blocks]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "inner_activation":self.inner_activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_seq_num":self.return_seq_num,
            "num_blocks":self.num_blocks
            }

# added by zhaowuxia begins
# a layer of clockwork RNN layer
class CLOCKWORK(Layer):
    '''A Clockwork RNN layer updates "modules" of neurons at specific rates.
    In a vanilla :class:`RNN` layer, all neurons in the hidden pool are updated
    at every time step by mixing an affine transformation of the input with an
    affine transformation of the state of the hidden pool neurons at the
    previous time step:
    .. math::
       h_t = g(x_tW_{xh} + h_{t-1}W_{hh} + b_h)
    In a Clockwork RNN layer, neurons in the hidden pool are split into
    :math:`M` "modules" of equal size (:math:`h^i` for :math:`i = 1, \dots, M`),
    each of which has an associated clock period (a positive integer :math:`T_i`
    for :math:`i = 1, \dots, M`). The neurons in module :math:`i` are updated
    only when the time index :math:`t` of the input :math:`x_t` is an even
    multiple of :math:`T_i`. Thus some of modules (those with large :math:`T`)
    only respond to "slow" features in the input, and others (those with small
    :math:`T`) respond to "fast" features.
    Furthermore, "fast" modules with small periods receive inputs from "slow"
    modules with large periods, but not vice-versa: this allows the "slow"
    features to influence the "fast" features, but not the other way around.
    The state :math:`h_t^i` of module :math:`i` at time step :math:`t` is thus
    governed by the following mathematical relation:
    .. math::
       h_t^i = \left\{ \begin{align*}
          &g\left( x_tW_{xh}^i + b_h^i +
             \sum_{j=1}^i h_{t-1}^jW_{hh}^j\right)
             \mbox{ if } t \mod T_i = 0 \\
          &h_{t-1}^i \mbox{ otherwise.} \end{align*} \right.
    Here, the modules have been ordered such that :math:`T_j > T_i` for
    :math:`j < i`.
    In ``theanets``, this update relation is implemented using a nested loop.
    The outer loop calls Theano's ``scan()`` operator to iterate over the input
    data at each time step. The inner loop iterates over the modules, updating
    each module if the clock cycle is correct, and copying over the previous
    value of the module if not.
    Parameters
    ----------
    periods : sequence of int
        The periods for the modules in this clockwork layer. The number of
        values in this sequence specifies the number of modules in the layer.
        The layer size must be an integer multiple of the number of modules
        given in this sequence.
    References
    ----------
    .. [1] J. Koutn��k, K. Greff, F. Gomez, & J. Schmidhuber. (2014) "A Clockwork
           RNN." http://arxiv.org/abs/1402.3511
    '''
    def __init__(self, input_dim, output_dim=128, 
        init='glorot_uniform', inner_init='orthogonal', 
        activation='tanh', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_seq_num=1, periods = [1]):
    
        super(CLOCKWORK,self).__init__()
        assert(output_dim % len(periods) == 0)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_seq_num = return_seq_num
        self.periods = np.asarray(sorted(periods,reverse=True))

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))
        n = self.output_dim // len(self.periods)
        self.U = [ self.inner_init(((i+1)*n, n)) for i in range(len(self.periods))]
        self.params = [
            self.W, self.b
        ] + self.U

        if weights is not None:
            self.set_weights(weights)

    def _step(self, 
        t, x_t,
        p_tm1, h_tm1):
       
        n = self.output_dim // len(self.periods)
        p_t = T.concatenate([
            theano.ifelse.ifelse(
                T.eq(t%TT, 0),
                x_t[:, i*n:(i+1)*n] + T.dot(h_tm1[:,:(i+1)*n], self.U[i]),
                p_tm1[:, i*n:(i+1)*n])
            for i, TT in enumerate(self.periods)], axis=1)
        return p_t, self.activation(p_t)
    
    def get_output(self, train):
        X = self.get_input(train) 
        X = X.dimshuffle((1,0,2)) #[T, sz, input_dim]
        
        x = T.dot(X, self.W) + self.b
        (pre, outputs), updates = theano.scan(
            self._step, 
            sequences=[T.arange(X.shape[0]), x], 
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)],
            truncate_gradient=self.truncate_gradient
        )
        
        if self.return_seq_num <= 0:
            return outputs.dimshuffle((1,0,2)) #[sz, T, out * nb]
        elif self.return_seq_num > 1:
            return outputs[-self.return_seq_num:].dimshuffle((1,0,2)) #[sz, return_seq_num, out * nb]
        else:
            return outputs[-1] #[sz, output_dim * num_blocks]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "inner_activation":self.inner_activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_seq_num":self.return_seq_num,
            "periods":self.periods}
