"""
Training evaluation module
"""

from typing import List
from transformers import PreTrainedTokenizer
from torchmetrics import BLEUScore
import torch


class Seq2SeqTrainingEval:  # pylint: disable=too-few-public-methods
    """
    Seq2Seq Training Evaluation
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        self.eval_metric = BLEUScore()

    def compute_eval(self, preds: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute BLEU Score based on the prediction and target
        on evaluation

        Parameters
        ----------
        preds : torch.Tensor
            Prediction result from evaluation batches 
            in the training process. Shape should be (num_data, seqlen)
        target : torch.Tensor
            Target from evaluation batches in the training process

        Returns
        -------
        float
            Return the bleu score of the model
        """
        agg_preds, agg_tgts = [], []

        for pred, tgt in zip(preds, target):
            decoded_predicted: List[str] = list(map(self.tokenizer.decode, pred))
            target[target == -100] = self.tokenizer.pad_token_id  # type: ignore
            decoded_target: List[str] = list(map(self.tokenizer.decode, tgt))
            agg_preds.append(decoded_predicted)
            agg_tgts.append([decoded_target])
        bleu_score: float = self.eval_metric(agg_preds, agg_tgts)
        return bleu_score
