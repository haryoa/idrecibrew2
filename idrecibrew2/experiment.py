"""
Experiment related module
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
import rich
from rich.console import Console

from idrecibrew2.data import Seq2SeqDataFactory, Seq2SeqDataFactoryArgs
from idrecibrew2.data.indonlg_tokenizer.tokenizer import IndoNLGTokenizer
from idrecibrew2.eval import Seq2SeqTrainingEval
from idrecibrew2.model import LitSeq2SeqTransformers, LitSeq2SeqTransformersArgs


console = Console()


@dataclass
class ScenarioConfig:
    """
    Scneario Config as the input of Experiment
    """

    output_dir: str
    train_csv_path: str
    dev_csv_path: str
    data_factory_args: Dict[str, Any]
    lit_trainer_args: Dict[str, Any]
    model_args: Dict[str, Any]
    model_ckpt_args: Optional[Dict[str, Any]] = None
    early_stopping_args: Optional[Dict[str, Any]] = None
    wandb_loggers_args: Optional[Dict[str, Any]] = None
    batch_size: int = 3  # shared between training and validation
    data_n_workers: int = 1
    tokenizer: str = "indobenchmark/indobart-v2"
    skip_eval_bleu: bool = False  # skip bleu computation to make it fast


scenarios = {
    "indobert-v2": {
        "output_dir": "outputs/indobert-v2/",
        "train_csv_path": "data/processed/train.csv",
        "dev_csv_path": "data/processed/dev.csv",
        "data_factory_args": {"source_column": "src", "label_column": "tgt"},
        "lit_trainer_args": {
            "precision": 16,
            "max_epochs": 100,
        },
        "model_args": {
            "model_type": "indobart-v2",
            "optimizer_type": "adam",
            "learning_rate": 1e-5,
        },
        "model_ckpt_args": {
            "monitor": "val_loss",
            "filename": "model-{epoch:02d}-{val_loss:.3f}-{val_bleu:.3f}",
            "mode": "min",
            "save_last": True,
        },
        "early_stopping_args": {"monitor": "val_loss", "patience": 5, "mode": "min"},
        "wandb_loggers_args": {"project": "indorecibrew2", "name": "indobert-v2"},
        "batch_size": 32,
        "data_n_workers": 4,
    },
    "indogpt-2": {
        "output_dir": "outputs/indogpt-2/",
        "train_csv_path": "data/processed/train.csv",
        "dev_csv_path": "data/processed/dev.csv",
        "data_factory_args": {
            "source_column": "src",
            "label_column": "tgt",
            "training_type": "lm",
        },
        "lit_trainer_args": {
            "precision": 16,
            "max_epochs": 99,
        },
        "model_args": {
            "model_type": "indogpt",
            "optimizer_type": "adamw",
            "learning_rate": 1e-4,
        },
        "model_ckpt_args": {
            "monitor": "val_loss",
            "filename": "model-{epoch:02d}-{val_loss:.3f}-{val_bleu:.3f}",
            "mode": "min",
            "save_last": True,
        },
        "early_stopping_args": {"monitor": "val_loss", "patience": 5, "mode": "min"},
        "wandb_loggers_args": {"project": "indorecibrew2", "name": "indogpt-2"},
        "batch_size": 32,
        "data_n_workers": 4,
        "tokenizer": "indobenchmark/indogpt",
        "skip_eval_bleu": True,  # we check the BLEU in the testing and evaluation
    },
    "indogpt": {
        "output_dir": "outputs/indogpt/",
        "train_csv_path": "data/processed/train.csv",
        "dev_csv_path": "data/processed/dev.csv",
        "data_factory_args": {
            "source_column": "src",
            "label_column": "tgt",
            "training_type": "lm",
        },
        "lit_trainer_args": {
            "precision": 16,
            "max_epochs": 100,
        },
        "model_args": {
            "model_type": "indogpt",
            "optimizer_type": "adam",
            "learning_rate": 5e-5,
        },
        "model_ckpt_args": {
            "monitor": "val_loss",
            "filename": "model-{epoch:02d}-{val_loss:.3f}-{val_bleu:.3f}",
            "mode": "min",
            "save_last": True,
        },
        "early_stopping_args": {"monitor": "val_loss", "patience": 5, "mode": "min"},
        "wandb_loggers_args": {"project": "indorecibrew2", "name": "indogpt"},
        "batch_size": 32,
        "data_n_workers": 4,
        "tokenizer": "indobenchmark/indogpt",
        "skip_eval_bleu": True,  # we check the BLEU in the testing and evaluation
    },
    "indo-t5": {
        "output_dir": "outputs/indo-t5/",
        "train_csv_path": "data/processed/train_t5.csv",
        "dev_csv_path": "data/processed/dev_t5.csv",
        "data_factory_args": {
            "source_column": "src",
            "label_column": "tgt",
            "training_type": "seq2seq",
        },
        "lit_trainer_args": {
#             "precision": 16,  the pretrained-model cannot do fp16 precision..
            "max_epochs": 100,
        },
        "model_args": {
            "model_type": "Wikidepia/IndoT5-base",
            "optimizer_type": "adamw",
            "learning_rate": 1e-4,  # following : transformers_summarization_wandb.ipynb in transformers
        },
        "model_ckpt_args": {
            "monitor": "val_loss",
            "filename": "model-{epoch:02d}-{val_loss:.3f}",
            "mode": "min",
            "save_last": True,
        },
        "early_stopping_args": {"monitor": "val_loss", "patience": 5, "mode": "min"},
        "wandb_loggers_args": {"project": "indorecibrew2", "name": "indo-t5"},
        "batch_size": 32,
        "data_n_workers": 4,
        "tokenizer": "Wikidepia/IndoT5-base",
        "skip_eval_bleu": True,  # Skip for faster training (trf decoder is kinda slow)
    },
     "indo-t5-2": {
        "output_dir": "outputs/indo-t5-2/",
        "train_csv_path": "data/processed/train_t5.csv",
        "dev_csv_path": "data/processed/dev_t5.csv",
        "data_factory_args": {
            "source_column": "src",
            "label_column": "tgt",
            "training_type": "seq2seq",
        },
        "lit_trainer_args": {
#             "precision": 16,  the pretrained-model cannot do fp16 precision..
            "max_epochs": 99,
        },
        "model_args": {
            "model_type": "Wikidepia/IndoT5-base",
            "optimizer_type": "adamw",
            "learning_rate": 1e-4,  # following : transformers_summarization_wandb.ipynb in transformers
        },
        "model_ckpt_args": {
            "monitor": "val_loss",
            "filename": "model-{epoch:02d}-{val_loss:.3f}",
            "mode": "min",
            "save_last": True,
        },
        "early_stopping_args": {"monitor": "val_loss", "patience": 5, "mode": "min"},
        "wandb_loggers_args": {"project": "indorecibrew2", "name": "indo-t5-2"},
        "batch_size": 32,
        "data_n_workers": 4,
        "tokenizer": "Wikidepia/IndoT5-base",
        "skip_eval_bleu": True,  # Skip for faster training (trf decoder is kinda slow)
    },
}


class Experiment:
    """
    Experiment class
    """

    def __init__(
        self,
        scenario: Optional[str] = None,
        gpus: Optional[List[int]] = None,
        is_test: bool = False,
    ) -> None:
        """
        Doing experiment using this class.

        Parameters
        ----------
        scenario : Optional[str], optional
            If None, it won't run!, by default None
        gpus : Optional[List[int]], optional
            GPUS used, if None, use CPU instead, by default None
        is_test : bool, optional
            Run with a testing purpose, by default False
        """
        if scenario is None:
            raise ValueError("scenario must not be None")
        self.scenario_dict = scenarios.get(scenario)
        self.scenario_config = ScenarioConfig(**self.scenario_dict)  # type: ignore
        os.makedirs(self.scenario_config.output_dir, exist_ok=True)
        self.dataloaders: Dict[str, DataLoader] = {}  # type: ignore
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.gpus = gpus
        self.is_test = is_test

    def pipe_data(self) -> None:
        """
        Data processing pipeline stuffs goes here
        """
        if self.scenario_config.tokenizer in (
            "indobenchmark/indobart-v2",
            "indobenchmark/indogpt",
        ):
            self.tokenizer = IndoNLGTokenizer.from_pretrained(
                self.scenario_config.tokenizer
            )
        else:
            print("Auto tokenizer!")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.scenario_config.tokenizer
            )
        data_args = Seq2SeqDataFactoryArgs(
            tokenizer=self.tokenizer, **self.scenario_config.data_factory_args
        )
        data_factory = Seq2SeqDataFactory(data_args=data_args)

        for key, csv_file in [
            ("train_dataloaders", self.scenario_config.train_csv_path),
            ("val_dataloaders", self.scenario_config.dev_csv_path),
        ]:
            self.dataloaders[key] = data_factory.produce_dataloader_from_csv(
                csv_file=csv_file,
                batch_size=self.scenario_config.batch_size,
                n_workers=self.scenario_config.data_n_workers,
                shuffle=key == "train_dataloaders",
            )

    def pipe_train(self) -> None:
        """
        Training process goes here!
        """
        if self.tokenizer is None:
            raise ValueError("you must assign self.tokenizer by invoking `pipe_data()`")
        eval_obj = Seq2SeqTrainingEval(tokenizer=self.tokenizer)
        lit_model_args = LitSeq2SeqTransformersArgs(
            vocab_size=self.tokenizer.vocab_size, **self.scenario_config.model_args
        )
        lit_model = LitSeq2SeqTransformers(config=lit_model_args)
        if not self.scenario_config.skip_eval_bleu:
            lit_model.set_eval_object(eval_obj)
        callbacks: List[Any] = []
        logger = None
        if not self.is_test:
            if self.scenario_config.model_ckpt_args is not None:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=self.scenario_config.output_dir,
                        **self.scenario_config.model_ckpt_args,
                    )
                )
            if self.scenario_config.early_stopping_args is not None:
                callbacks.append(
                    EarlyStopping(**self.scenario_config.early_stopping_args)
                )
            if self.scenario_config.wandb_loggers_args is not None:
                logger = WandbLogger(**self.scenario_config.wandb_loggers_args)
                logger.experiment.config.update(self.scenario_dict)

        trainer = Trainer(
            callbacks=callbacks,
            logger=logger,  # type: ignore
            gpus=self.gpus,
            fast_dev_run=self.is_test,
            **self.scenario_config.lit_trainer_args,
        )
        trainer.fit(lit_model, **self.dataloaders)

    def run(self):
        self.pipe_data()
        self.pipe_train()
