"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

import argparse
import torch

from optimizer.kfac.preconditioner import KFACPreconditioner
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Initialize argument parser
parser = argparse.ArgumentParser(description='Training configuration for GPT-2 on OpenWebText.')

# I/O
parser.add_argument('--out_dir', default='out', type=str)
parser.add_argument('--eval_interval', default=50, type=int)
parser.add_argument('--log_interval', default=1, type=int)
parser.add_argument('--eval_iters', default=200, type=int)
parser.add_argument('--eval_only', action='store_true', default=False)
parser.add_argument('--always_save_checkpoint', action='store_true', default=True)
parser.add_argument('--init_from', default='scratch', choices=['scratch', 'resume', 'gpt2*'], type=str)

# wandb logging
parser.add_argument('--wandb_log', action='store_false', default=True)
parser.add_argument('--wandb_entity', default='project-toast', type=str)
parser.add_argument('--wandb_project', default='owt', type=str)
parser.add_argument('--wandb_run_name', default='gpt2', type=str)

# data
parser.add_argument('--dataset', default='openwebtext', type=str)
parser.add_argument('--gradient_accumulation_steps', default=5*8, type=int)
parser.add_argument('--batch_size', default=12, type=int)
parser.add_argument('--block_size', default=1024, type=int)

# model
parser.add_argument('--n_layer', default=4, type=int)
parser.add_argument('--n_head', default=4, type=int)
parser.add_argument('--n_embd', default=512, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--bias', action='store_true', default=False)
parser.add_argument('--optim', default='AdamW', type=str)

# optimizer config
parser.add_argument('--learning_rate', default=3e-3, type=float)
parser.add_argument('--max_iters', default=600000, type=int)
parser.add_argument('--weight_decay', default=1e-1, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.95, type=float)
parser.add_argument('--grad_clip', default=1.0, type=float)
# shampoo
parser.add_argument('--matrix_eps', default=1.0e-6, type=float)
parser.add_argument('--start_preconditioning_step', default=25, type=int)
parser.add_argument('--preconditioning_compute_steps', default=10, type=int)
parser.add_argument('--statistics_compute_steps', default=100, type=int)
parser.add_argument('--shampoo_block_size', default=128, type=int)
parser.add_argument('--gradient_value_clip', default=-1, type=float)

parser.add_argument('--kl_clip', default=1e-3, type=float)

# learning rate decay settings
parser.add_argument('--decay_lr', action='store_true', default=True)
parser.add_argument('--warmup_iters', default=2000, type=int)
parser.add_argument('--lr_decay_iters', default=600000, type=int)
parser.add_argument('--lr_ratio', default=1e-2, type=float)

parser.add_argument('--base_batch_size_token', default=491520, type=int)
parser.add_argument('--lr_batch_exp', default=1, type=float)

# DDP settings
parser.add_argument('--backend', default='nccl', type=str, choices=['nccl', 'gloo'])
parser.add_argument('--grafting', default='AdaGrad', type=str, choices=['None', 'SGD', 'AdaGrad'])
# system
parser.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda', 'mps'])
parser.add_argument('--dtype', default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16', type=str, choices=['float32', 'bfloat16', 'float16'])
parser.add_argument('--compile', action='store_true', default=False)

# Parse the arguments
args = parser.parse_args()
args.n_gpus = torch.cuda.device_count()
args.batch_size_token = args.batch_size * args.block_size * args.gradient_accumulation_steps * args.n_gpus

args.lr = args.learning_rate * (args.batch_size_token / args.base_batch_size_token)**args.lr_batch_exp
args.min_lr = args.lr * args.lr_ratio
# Construct the config dictionary
config = vars(args)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=args.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    args.device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(args.device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert args.gradient_accumulation_steps % ddp_world_size == 0
    args.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = args.gradient_accumulation_steps * ddp_world_size * args.batch_size * args.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(args.out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in args.device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', args.dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(args.device, non_blocking=True), y.pin_memory().to(args.device, non_blocking=True)
    else:
        x, y = x.to(args.device), y.to(args.device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                  bias=args.bias, vocab_size=None, dropout=args.dropout) # start with model_args from command line
if args.init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif args.init_from == 'resume':
    print(f"Resuming training from {args.out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif args.init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {args.init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=args.dropout)
    model = GPT.from_pretrained(args.init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if args.block_size < model.config.block_size:
    model.crop_block_size(args.block_size)
    model_args['block_size'] = args.block_size # so that the checkpoint will have the right value
model.to(args.device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(args.optim, args.weight_decay, args.lr, (args.beta1, args.beta2), args, device_type)
if 'K-FAC' in args.optim:
    preconditioner = KFACPreconditioner(model,
                                        factor_update_steps=args.preconditioning_compute_steps,
                                        inv_update_steps=args.statistics_compute_steps,
                                        damping=args.matrix_eps,
                                        factor_decay=args.beta2,
                                        kl_clip=args.kl_clip,
                                        accumulation_steps=args.gradient_accumulation_steps)
if args.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)

# logging
if args.wandb_log and master_process:
    import wandb
    wandb.init(config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if args.decay_lr else args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % args.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if args.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or args.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {args.out_dir}")
                torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))
        if math.isnan(losses['train']):
            break
    if iter_num == 0 and args.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(args.gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / args.gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        if 'K-FAC' in args.optim:
            preconditioner.step()
    # clip the gradient
    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % args.log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * args.gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(args.batch_size * args.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > args.max_iters:
        break

if ddp:
    destroy_process_group()
