from .ccpd_data import CCPD
from .aolp import AOLP
from .clpd import CLPD

def get_dataset(config):

    if config.DATASET.DATASET == "CCPD":
        return CCPD
    elif config.DATASET.DATASET == "AOLP":
        return AOLP
    elif config.DATASET.DATASET == "CLPD":
        return CLPD
    else:
        raise NotImplemented()