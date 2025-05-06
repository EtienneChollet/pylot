from pylot.torch.torchlib import torch, nn

# Hookless implementation of weight norm

class WeightNorm(nn.Module):

    def __init__(self, module, weights=None):
        super(WeightNorm, self).__init__()
        assert not isinstance(module, WeightNorm), "Nested weight norms"
        self.module = module
        if weights is None:
            weights = list(module._parameters.keys())
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = nn.Parameter(g.data)
            v = nn.Parameter(v.data)

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(f"{name_w}_g", g)
            self.module.register_parameter(f"{name_w}_v", v)

    def _setweights(self):
        for name_w in self.weights:
            g = getattr(self.module, f"{name_w}_g")
            v = getattr(self.module, f"{name_w}_v")
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)
