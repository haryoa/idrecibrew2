import torch
from idrecibrew2.eval.training_eval import Seq2SeqTrainingEval
from idrecibrew2.data.indonlg_tokenizer.tokenizer import IndoNLGTokenizer
from numpy.testing import assert_almost_equal


def test_eval_training():
    output_dict_tensor = [
        {
            "val_loss": 0.22,
            "preds": torch.Tensor([[0, 555, 666, 777], [6666, 222, 444, 555]]),
            "tgts": torch.Tensor([[0, 14141, 5222, 2], [0, 3333, 7777, 2]]),
        },
        {
            "val_loss": 0.11,
            "preds": torch.Tensor([[0, 444, 662, 1553, 2444, 7778], [0, 5214, 2234, 1234, 555, 666]]),
            "tgts": torch.Tensor([[0, 444, 662, 1553, 2444, 7778], [0, 5214, 2234, 1234, 555, 666]]),
        },
    ]

    tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2")
    eval_tok = Seq2SeqTrainingEval(tokenizer)
    result_bleu = eval_tok.compute_eval(output_dict_tensor, target_cols="tgts", pred_cols="preds")
    assert_almost_equal(result_bleu, 0.84089, 5)
