"""
Collections of Lit Model Arguments
"""
from typing import Optional, Tuple

from dataclasses import dataclass
from idrecibrew2.mixin import MappingMixin  # type: ignore


@dataclass
class LitSeq2SeqTransformersArgs(MappingMixin):  # type: ignore
    """
    Arguments for LitSeq2SeqTransformer

    Parameters
    ----------
    vocab_size: int
        Vocab size
    model_type: str
        Backbone of the model. Possible choices are:
        `indobart-v2`
    optimizer_type: str
        What kind of optimizer you wanna use. Possible choices:
        `adam`
    learning_rate: float
        Learning rate of the model
    weight_decay: float
        Regularization
    """

    vocab_size: int
    model_type: str = "indobart-v2"
    optimizer_type: str = "adam"
    learning_rate: float = 1e-5
    warmup_strategy: Optional[str] = None
    weight_decay: float = 0.0001
    optimizer_beta: Tuple[float, float] = (0.9, 0.999)
    optimizer_epsilon: float = 1e-8
