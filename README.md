# Implementation of SimCLR in PyTorch

The whole code has been implemented in Jupyter Notebook.

## Using LARS

```params = []
param_names = []
for n,p in self.net.named_parameters():
    params.append(p)
    param_names.append(n)
parameters = [{'params':params,'param_names':param_names}]
optimizer = LARS(parameters,
                 lr = self.lr,
                 weight_decay = self.weight_decay,
                 exclude_from_weight_decay=["batch_normalization", "bias"])```
