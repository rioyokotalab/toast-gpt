
# nanoGPT with Second Order Optimization

This is a repository to optimize nanoGPT using second order optimization.

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start

You can prepare the openwebtext with the following command.

```
$ python data/openwebtext/prepare.py
```

Here are the commands to train a 10M size model with Adam.

```
$ python train.py --batch_size=12 --block_size=1024 --dataset=openwebtext --eval_interval=20 --eval_iters=20 --gradient_accumulation_steps=32 --learning_rate=0.003 --log_interval=1 --lr_decay_iters=10000 --max_iters=10000 --n_embd=512 --n_head=4 --n_layer=4 --optim=AdamW --warmup_iters=2000
```