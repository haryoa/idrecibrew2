"""
Experiment related module
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from idrecibrew2.data import Seq2SeqDataFactory, Seq2SeqDataFactoryArgs
from idrecibrew2.data.indonlg_tokenizer.tokenizer import IndoNLGTokenizer
from idrecibrew2.eval import Seq2SeqTrainingEval
from idrecibrew2.model import (LitSeq2SeqTransformers,
                               LitSeq2SeqTransformersArgs)


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


scenarios = {
    "indobert-v2": {
        "output_dir": "outputs/indobert-v2/",
        "train_csv_path": "data/processed/train.csv",
        "dev_csv_path": "data/processed/dev.csv",
        "data_factory_args": {
            "source_column": "src",
            "label_column": "tgt"
        },
        "lit_trainer_args": {
            "precision": 16,
            "max_epochs": 100,
        },
        "model_args": {
            "model_type": "indobart-v2",
            "optimizer_type": "adam",
            "learning_rate": 1e-5
        },
        "model_ckpt_args": {
            "monitor": "val_loss",
            "filename": "model-{epoch:02d}-{val_loss:.3f}-{val_bleu:.3f}",
            "mode": "min",
            "save_last": True,
        },
        "early_stopping_args": {
            "monitor": "val_loss",
            "patience": 5,
            "mode": "min"
        },
        "wandb_loggers_args": {
            "project": "indorecibrew2",
            "name": "indobert-v2"
        },
        "batch_size": 32,
        "data_n_workers": 4
    }
}


class Experiment:
    """
    Experiment class
    """
    def __init__(
        self, scenario: Optional[str] = None, gpus: Optional[List[int]] = None, is_test: bool = False
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
        self.scenario_dict = scenarios.get(scenario)
        self.scenario_config = ScenarioConfig(**self.scenario_dict)
        os.makedirs(self.scenario_config.output_dir, exist_ok=True)
        self.dataloaders = {}
        self.tokenizer = None
        self.gpus = gpus
        self.is_test = is_test

    def pipe_data(self):
        """
        Data processing pipeline stuffs goes here
        """
        self.tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2")
        data_args = Seq2SeqDataFactoryArgs(
            tokenizer=self.tokenizer,
            **self.scenario_config.data_factory_args
        )
        data_factory = Seq2SeqDataFactory(data_args=data_args)
        
        for key, csv_file in [("train_dataloaders", self.scenario_config.train_csv_path), ("val_dataloaders", self.scenario_config.dev_csv_path)]:
            self.dataloaders[key] = data_factory.produce_dataloader_from_csv(
                csv_file=csv_file, batch_size=self.scenario_config.batch_size, n_workers=self.scenario_config.data_n_workers, shuffle=key=='train_dataloaders'
            )
    
    def pipe_train(self):
        """
        Training process goes here!
        """
        eval_obj = Seq2SeqTrainingEval(tokenizer=self.tokenizer)
        lit_model_args = LitSeq2SeqTransformersArgs(
            vocab_size=self.tokenizer.vocab_size,
            **self.scenario_config.model_args
        )
        lit_model = LitSeq2SeqTransformers(config=lit_model_args)
        lit_model.set_eval_object(eval_obj)
        callbacks = []
        logger = None
        if not self.is_test:
            if self.scenario_config.model_ckpt_args is not None:
                callbacks.append(ModelCheckpoint(dirpath=self.scenario_config.output_dir, **self.scenario_config.model_ckpt_args))
            if self.scenario_config.early_stopping_args is not None:
                callbacks.append(EarlyStopping(**self.scenario_config.early_stopping_args))
            if self.scenario_config.wandb_loggers_args is not None:
                logger = WandbLogger(**self.scenario_config.wandb_loggers_args)
                logger.experiment.config.update(self.scenario_dict)
        
        trainer = Trainer(callbacks=callbacks, logger=logger, gpus=self.gpus, fast_dev_run=self.is_test, **self.scenario_config.lit_trainer_args)
        trainer.fit(lit_model, **self.dataloaders)
    
    def run(self):
        self.pipe_data()
        self.pipe_train()
