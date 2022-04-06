"""
Test for modeling
"""

import torch
from pytorch_lightning import Trainer
from idrecibrew2.model import LitSeq2SeqTransformers, LitSeq2SeqTransformersArgs
from idrecibrew2.eval import Seq2SeqTrainingEval
from idrecibrew2.data.indonlg_tokenizer.tokenizer import IndoNLGTokenizer


def test_forward_model():
    input_ids = torch.LongTensor([[1, 2, 3, 2, 1], [2, 1, 3, 0, 0]])
    attention_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    labels = torch.LongTensor([[1, 3, 2], [4, 2, 1]])
    tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2")
    eval_obj = Seq2SeqTrainingEval(tokenizer=tokenizer)
    lit_model_args = LitSeq2SeqTransformersArgs(
        model_type="indobart-v2",
        vocab_size=tokenizer.vocab_size,
        optimizer_type="adam",
        learning_rate=1e-5,
    )
    lit_model = LitSeq2SeqTransformers(config=lit_model_args)
    lit_model.set_eval_object(eval_obj)
    result = lit_model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )
    assert result.logits.shape == (2, 3, tokenizer.vocab_size)
