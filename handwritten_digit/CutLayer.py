from lasagne.layers import Layer
from lasagne.random import get_rng
from theano.tensor.shared_randomstreams import RandomStreams

class CutLayer(Layer):
    """CutLayer
    Parameters
    ----------
    incoming: a :class:`Layer` instance or tuple
        the layer feeding into this layer, or the expected input shape
    p: float
        the probability of disconnecting p % number of inputs
    """
    def __init__(self, incoming, p=0.5, rescale=True, **kwargs):
        super(CutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.selected = None
        self.p = p

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
        if self.selected is None:
            self.selected = self._srng.binomial(input_shape, p=retain_prob, dtype=input.dtype)
        return input * self.selected

cut = CutLayer