from lasagne.layers import Layer
from lasagne.random import get_rng
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
from theano import printing as pt

class CutLayer(Layer):
    """CutLayer
    Parameters
    ----------
    incoming: a :class:`Layer` instance or tuple
        the layer feeding into this layer, or the expected input shape
    p: float
        the probability of disconnecting p % number of inputs
    shared: float
        the portion for neurons to be influenced by A
    """
    def __init__(self, incoming, shared=0.5, p=0.5, rescale=True, **kwargs):
        super(CutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self.shared = shared

    def get_output_for(self, input, **kwargs):
        """
        Parameters
        ----------
        input: tensor
            output form the previous layer
        """
        retain_prob = 1 - self.p
        # use nonsymbolic shape for dropout mask if possible
        input_shape = self.input_shape
        if any(s is None for s in input_shape):
            input_shape = input.shape
        # prev half: shared% from prev half input | back half (100-shared)% from back half output
        A_shape = (input_shape[0], input_shape[1]/2)
        B_shape = (input_shape[0],(input_shape[1]+1)/2)
        A_select = self._srng.binomial(A_shape, p=self.shared, dtype=input.dtype)
        B_select = self._srng.binomial(B_shape, p=1-self.shared, dtype=input.dtype)
        selected = T.concatenate([A_select, B_select], axis=1)
        return input * selected * self._srng.binomial(input_shape, p=retain_prob, dtype=input.dtype)
        
cut = CutLayer