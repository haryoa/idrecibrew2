from idrecibrew2.data import Seq2SeqDataFactory, Seq2SeqDataFactoryArgs
from idrecibrew2.data.indonlg_tokenizer.tokenizer import IndoNLGTokenizer


def test_load_dataloader():
    tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2")
    data_args = Seq2SeqDataFactoryArgs(
        source_column="src",
        label_column="tgt",
        tokenizer=tokenizer,
    )
    data_factory = Seq2SeqDataFactory(data_args=data_args)
    dtl = data_factory.produce_dataloader_from_csv(
        csv_file="tests/data_dummy/test.csv", batch_size=3, n_workers=1, shuffle=True
    )
    asserted = next(iter(dtl))

    assert (
        "input_ids" in asserted
        and "attention_mask" in asserted
        and "labels" in asserted
    )
