"""
Module for model training
"""

from typing import Any, Dict, List, Optional, Union

from pytorch_lightning import LightningModule
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoModelForSeq2SeqLM,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
)

from .lit_args import LitSeq2SeqTransformersArgs
from idrecibrew2.eval.training_eval import Seq2SeqTrainingEval


class LitSeq2SeqTransformers(LightningModule):
    """
    Lightning Module for Seq2Sweq
    """

    def __init__(self, config: LitSeq2SeqTransformersArgs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.eval_obj: Optional[Seq2SeqTrainingEval] = None
        self._init_model()

    def set_eval_object(self, eval_obj: Seq2SeqTrainingEval) -> None:
        """
        Set evaluation object to evaluate

        Parameters
        ----------
        eval_obj : Seq2SeqTrainingEval
            Object for training evaluation
        """
        self.eval_obj = eval_obj

    def forward(self, **input_to_hf):  # type: ignore  # pylint: disable=all
        return self.model(**input_to_hf)

    def training_step(self, batch, batch_idx):  # type: ignore  # pylint disable=all
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore  # pylint disable=all

        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        # Greedy
        argmax = None
        
        if self.eval_obj is not None:
            argmax = self.model.generate(
                input_ids=batch["input_ids"].to(self.device),
                max_length=150,
                num_return_sequences=1,
                num_beams=1,
                num_beam_groups=1,
            )

        self.log("val_loss", val_loss)
        return {"val_loss": val_loss, "preds": argmax, "tgts": batch.labels}

    def validation_epoch_end(self, outputs):  # type: ignore  # pylint disable=all
        if self.eval_obj is not None:
            bleu_score = self.eval_obj.compute_eval(
                outputs, target_cols="tgts", pred_cols="preds"
            )
            self.log("val_bleu", bleu_score, prog_bar=True)

    def test_step(self, batch, batch_idx):  # type: ignore  # pylint disable=all

        outputs = self(**batch)
        val_loss, logits = outputs[:2]

       # Greedy
        argmax = None
        
        if self.eval_obj is not None:
            argmax = self.model.generate(
                input_ids=batch["input_ids"].to(self.device),
                max_length=150,
                num_return_sequences=1,
                num_beams=1,
                num_beam_groups=1,
            )

        self.log("test_loss", val_loss)
        return {"test_loss": val_loss, "preds": argmax, "tgts": batch.labels}

    def test_epoch_end(self, outputs):  # type: ignore  # pylint disable=all
        if self.eval_obj is not None:
            bleu_score = self.eval_obj.compute_eval(outputs["preds"], outputs["tgts"])
            self.log("test_bleu", bleu_score, prog_bar=True)

    def _init_model(self) -> None:
        if self.config.model_type == "indobart-v2":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(  # type: ignore
                "indobenchmark/indobart-v2"
            )
        elif self.config.model_type == "indogpt":
            self.model = GPT2LMHeadModel.from_pretrained("indobenchmark/indogpt")
        else:  # others are seq2seqLM like bart by inputting the model_type
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_type)

    def configure_optimizers(self) -> Any:
        """
        Prepare optimizer and schedule (linear warmup and decay)
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = self._get_optimizer(optimizer_grouped_parameters)

        return self._get_out_scheduler(optimizer)

    def _get_out_scheduler(self, optimizer: Union[Adam, AdamW]) -> Any:

        if self.config.warmup_strategy == "linear":

            linear_scheduler = get_linear_schedule_with_warmup(  # type: ignore
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.total_steps,
            )
            returned = [optimizer], [linear_scheduler]
        elif self.config.warmup_strategy == "step_lr":
            step_lr = StepLR(
                optimizer,
                step_size=self.config.warmup_steps,
                gamma=self.config.warmup_gamma,
            )
            returned = [optimizer], [step_lr]
        else:
            returned = optimizer  # type: ignore
        return returned

    def _get_optimizer(
        self, optimizer_grouped_parameters: List[Dict[str, Any]]
    ) -> Union[Adam, AdamW]:
        if self.config.optimizer_type == "adamw":
            opt: Union[AdamW, Adam] = AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=self.config.optimizer_beta,
                eps=self.config.optimizer_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "adam":
            opt = Adam(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=self.config.optimizer_beta,
                eps=self.config.optimizer_epsilon,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError("optimizer not implemented")
        return opt
