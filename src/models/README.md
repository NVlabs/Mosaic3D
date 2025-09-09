# Model Directory Structure

## Networks

This contains the network modules that are used in the lightning module. The rest of the training, test logics such as loss, metrics, and visualiaion are handled in the lightning module.

A network should provide a `forward` method that takes in the input data dictionary and returns the output data.

## Lighning Modules

The lightning modules process training, validation, and prediction. A network is initialized in the lightning module and the loss, metrics, and visualization are handled here.

```
models
│
├── README.md
│
├── components
│   ├── __init__.py
│   ├── base.py
│   ├── text_model.py
│   ├── ...
│
├── losses
│   ├── __init__.py
│   ├── base.py
│   ├── ...
│
├── metrics
│   ├── __init__.py
│   ├── base.py
│   ├── ...
│
├── networks
│   ├── __init__.py
│   ├── network_base.py
│   ├── warp
│   │   ├── __init__.py
│   │   ├── pointconvunet.py
│   │
│   ├── regionplc
│   │   ├── __init__.py
│   │   ├── regionplc.py
|
├── lighning_modules
|   ├── module_base.py
|   ├── vanilla_module.py
|   ├── ...
```
