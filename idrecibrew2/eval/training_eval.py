"""
Training evaluation module
"""

from typing import Dict, List
from transformers import PreTrainedTokenizer
# from torchmetrics import BLEUScore
import torch
import sacrebleu


class Seq2SeqTrainingEval:  # pylint: disable=too-few-public-methods
    """
    Seq2Seq Training Evaluation
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
#         self.eval_metric = BLEUScore()
    
    def compute_eval(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        target_cols: str = "tgts",
        pred_cols: str = "preds",
        special_func = None
    ) -> float:
        """
        Compute BLEU Score based on the prediction and target
        on evaluation

        Parameters
        ----------
        outputs : List[Dict[str, torch.Tensor]]
            Output from different batches in a list.
        target_cols : str
            target key in outputs
        pred_cols : str
            prediction key in ouptuts
        special_func
            Nonee

        Returns
        -------
        float
            Return the bleu score of the model
        """
        agg_preds, agg_tgts = [], []

        for output in outputs:
            pred = output[pred_cols]
            target = output[target_cols]
            if special_func is None:
                func_map = (lambda x: self.tokenizer.decode(x, skip_special_tokens=True))
            else:
                func_map = special_func
            decoded_predicted: List[str] = list(map(func_map, pred))
            target[target == -100] = self.tokenizer.pad_token_id  # type: ignore
            decoded_target: List[str] = list(map(func_map, target))
            agg_preds.extend(decoded_predicted)
            agg_tgts.extend(decoded_target)
        bleu_score: float = sacrebleu.corpus_bleu(agg_preds, [agg_tgts]).score
        return bleu_score
