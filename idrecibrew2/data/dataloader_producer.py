"""
Seq2Seq datafactory place
"""
from dataclasses import dataclass
from typing import Any, Dict

from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    DataCollatorForSeq2Seq,
)
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader


@dataclass
class Seq2SeqDataFactoryArgs:
    """
    Arguments for Seq2SeqDataFactory

    Parameters
    ----------
    tokenizer: PreTrainedTokenizer
        Tokenizer that will be used for tokenizing the text
    source_column: str
        Source or input column of the data
    label_column: str
        Target column of the data
    training_type: str
        Architecture of the model. Choices:
        `seq2seq` or `lm`
    """

    tokenizer: PreTrainedTokenizer
    source_column: str = "src"
    label_column: str = "tgt"
    max_input_length: int = 128
    max_target_length: int = 256
    training_type: str = "seq2seq"


class Seq2SeqDataFactory:
    """
    Data producer for Seq2Seq
    """

    def __init__(self, data_args: Seq2SeqDataFactoryArgs) -> None:
        """
        initialization Seq2Seq

        Parameters
        ----------
        data_args : Seq2SeqDataFactoryArgs
            ARguments for seq2seq
        """
        self.data_args = data_args

    def produce_dataloader_from_csv(
        self,
        csv_file: str,
        batch_size: int = 2,
        n_workers: int = 1,
        shuffle: bool = True,
    ) -> DataLoader:  # type: ignore
        """
        Produce Dataloader from CSV file

        Parameters
        ----------
        csv_file : str
            a csv filepath
        batch_size : int, optional
            Batchsize to be produced, by default 2
        n_workers : int, optional
            how many workers to train, by default 1
        shuffle : bool, optional
            Whether to shuffle or not, by default True

        Returns
        -------
        DataLoader
            Return the desired Dataloader
        """
        # IDK why but Dataset.from_csv is kinda slow
        df_csv = pd.read_csv(csv_file)
        dataset = Dataset.from_pandas(df_csv)
        old_column = dataset.column_names
        preprocess_func = self._preprocess_function_seq2seq if self.data_args.training_type == "seq2seq" else self._preprocess_function_lm
        dataset = dataset.map(preprocess_func, batched=True)
        dataset = dataset.remove_columns(old_column)
        collator = (
            DataCollatorForSeq2Seq(self.data_args.tokenizer, padding=True)
            if self.data_args.training_type == "seq2seq"
            else DataCollatorForLanguageModeling(
                tokenizer=self.data_args.tokenizer, mlm=False
            )
        )
        dl_ready: DataLoader = DataLoader(  # type: ignore
            dataset,  # type: ignore
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=shuffle,
            collate_fn=collator,
        )
        return dl_ready

    def get_vocab_size(self) -> int:
        """
        Get vocab size of the data

        Returns
        -------
        int
            size
        """
        return self.data_args.tokenizer.vocab_size

    def _preprocess_function_lm(self, examples: Any) -> Dict[str, Any]:
        """
        preprocess function for language modeling (gpt)
        model input become [inputs]>>[target]
        """
        inputs = examples[self.data_args.source_column]
        labels = examples[self.data_args.label_column]
        concatenated_inp = [inp + " >>> " + lab for inp, lab in list(zip(inputs, labels))]
        model_inputs: Dict[str, Any] = self.data_args.tokenizer(
            concatenated_inp,
            max_length=self.data_args.max_input_length,
            truncation=True,
        )
        return model_inputs

    def _preprocess_function_seq2seq(self, examples: Any) -> Dict[str, Any]:
        """
        Preprocess function for seq2seq problem! (bart and t5)
        """
        inputs = examples[self.data_args.source_column]
        model_inputs: Dict[str, Any] = self.data_args.tokenizer(
            inputs, max_length=self.data_args.max_input_length, truncation=True
        )

        # Setup the tokenizer for targets
        with self.data_args.tokenizer.as_target_tokenizer():
            labels = self.data_args.tokenizer(
                examples[self.data_args.label_column],
                max_length=self.data_args.max_target_length,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
