# ID Recibrew2

Training code for Indonesian Recipe Generator! 

For more information, please visit my [blog](https://haryoa.github.io/posts/id-recipe-generator)

## Requirements

You need to have Python version 3.8

1. First, install Pytorch from the [Pytorch site](https://pytorch.org/).
2. Then, install `pandas` (either using conda or pip)
3. do `pip install -r requirements.txt`

## Training

Run this command

```
python -m idrecibrew2.cli run-sc --scenario <SCENARIO> --gpus <GPUS>
```

- `<SCENARIO>`: experiment scenario that you want to run.
- `<GPUS>`: Gpus used, It must contain a list argument.

There are multiple scenarios that you can use to train the model:

1. `indobert-v2`: Training by fine-tuning a indobart-v2 model
2. `indogpt-2` : Training by fine-tuning a indogpt model
3. `indo-t5-2` : Training by fine-tuning a t5 model

