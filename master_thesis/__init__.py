from TFNetworkLayer import auto_register_layer_classes
from .bert import GatherPositionsLayer
auto_register_layer_classes(list(globals().values()))
