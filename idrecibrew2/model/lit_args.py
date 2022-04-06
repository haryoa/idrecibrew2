"""
Collections of Lit Model Arguments
"""
from typing import Optional

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
    optimizer: str
        What kind of optimizer you wanna use. Possible choices:
        `adam`
    learning_rate: float
        Learning rate of the model
    """

    vocab_size: int
    model_type: str = "indobart-v2"
    optimizer: str = "adam"
    learning_rate: float = 1e-5
    warmup_strategy: Optional[str] = None
