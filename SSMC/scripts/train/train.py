from pathlib import Path
from models.mlpc import MLPC
from models.frozen_mistral import FrozenMistral
from transformers import AutoTokenizer
from utils.load_data import load_data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
import argparse
import os  
import torch
import torch.distributed as dist  
import torch.nn as nn
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

'''
This was very sucky to debug
'''

# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
# sets up ddp based on the pytorch spec to handle training across multiple gpus 
# this greatly improved my training time by allowing me to train one batch per GPU (4 batches of training data), 
# which allowed me to use more mlps and train for more epochs 
def setup_ddp():
    '''
    sets up ddp based on pytorch spec to handle distributed training across 4 gpus
    trained one section per GPU
    Allowed me to use more mlps and train more more epochs (originally was maxing out at 12 MLPS for ~4500 epochs)
    '''
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"GPU {rank} of {world_size}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=120))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    return rank, local_rank, world_size, device

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def load_config():
    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_model(mlpc, encoder, section=None, rank=0):
    '''
    Saves the model for a specific section
    '''
    model_dir = Path(__file__).resolve().parent.parent.parent / "saved_models" / "a6000"
    model_dir.mkdir(exist_ok=True, parents=True)
    
    if section:
        model_path = model_dir / f"mlpc_{encoder}_{section}.pt"
    else:
        model_path = model_dir / f"mlpc_{encoder}_rank{rank}.pt"
    
    if rank == 0 or not dist.is_initialized():
        if isinstance(mlpc, nn.parallel.DistributedDataParallel):
            state_dict = mlpc.module.state_dict()
        else:
            state_dict = mlpc.state_dict()
            
        torch.save(state_dict, model_path)
        print(f"Model saved to {model_path}")

def evaluate_section(config, encoder, target_section, checkpoint_path=None):
    """
    Train a single section
    """
    rank, local_rank, world_size, device = setup_ddp()
    print(f"Loading {target_section} data...")
    training = load_data(config, encoder, split="train", target_section=target_section)
    validation = load_data(config, encoder, split="validation", target_section=target_section)
    
    if target_section not in training:
        cleanup()
        return
    
    print(f"Training {target_section} with {world_size} GPUs")
    
    # Get embeddings for a section
    section_embeddings, section_texts = training[target_section] 
    validation_embeddings, validation_texts = validation[target_section]
    # Split training embeddiings across gpus evenly (pretty agnostic to the number have used with 2,3, and 4)
    train_memos = len(section_embeddings)
    train_per_gpu = train_memos // world_size
    train_start = rank * train_per_gpu
    train_end = train_start + train_per_gpu if rank < world_size - 1 else train_memos
    train_embeddings = section_embeddings[train_start:train_end].to(device)
    train_texts = section_texts[train_start:train_end]
    # Same thing for val
    val_memos = len(validation_embeddings)
    val_per_gpu = val_memos // world_size
    val_start = rank * val_per_gpu
    val_end = val_start + val_per_gpu if rank < world_size - 1 else val_memos
    val_embeddings = validation_embeddings[val_start:val_end].to(device)
    val_texts = validation_texts[val_start:val_end]
    train_section = {target_section: (train_embeddings, train_texts)}
    val_section = {target_section: (val_embeddings, val_texts)}
    
    mlpc = MLPC().to(device)
    
    if world_size > 1:
        mlpc = nn.parallel.DistributedDataParallel(
            mlpc, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=False
        )
    
    frozen_mistral = FrozenMistral(device=device)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # I think this is necessary for ddp but honestly the setup is so confusing
    if isinstance(mlpc, nn.parallel.DistributedDataParallel):
        optimizer = torch.optim.AdamW(mlpc.module.parameters(), lr=float(config["softsrv"]["mlpc"]["learning_rate"]))
    else:
        optimizer = torch.optim.AdamW(mlpc.parameters(), lr=float(config["softsrv"]["mlpc"]["learning_rate"]))
    
    start_epoch = 0
    best_vloss = float("inf")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    epochs = config["softsrv"]["mlpc"]["training_steps"]
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(f"{runs_dir}/{time_stamp}_rank{rank}")
    print(f"Training {target_section} with {encoder} embeddings")
    final_checkpoint_dir = Path(__file__).resolve().parent.parent / "final_checkpoints" / "a6000"
    final_checkpoint_dir.mkdir(exist_ok=True, parents=True)
    for i in range(start_epoch, epochs):
        if world_size > 1:
            dist.barrier()
        # train and validate every epoch, trying to get convergence in 10k steps
        train_loss = train(mlpc, train_section, frozen_mistral, optimizer, loss_fn, i, epochs, device, tokenizer, rank)
        val_loss = validate(mlpc, val_section, frozen_mistral, loss_fn, device, tokenizer, rank)
        writer.add_scalars("loss", {"train": train_loss, "validation": val_loss}, i+1)
        print(f"Epoch {i+1}/{epochs} | Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        if val_loss < best_vloss:
            best_vloss = val_loss
            print(f"New best val_loss: {val_loss:.4f} ")    
    writer.close()
    
    # apparently you can only save one rank but since the gpus have to sync after each iter, the weights are synced anyway so it doesnt really matter which 
    # gpu you save (there will always be a rank 0 fwiw so that's what i went with)
    if rank == 0:
        save_model(mlpc, encoder, section=target_section, rank=rank)
    cleanup()

def train(mlpc, training, frozen_mistral, optimizer, loss_fn, epoch, epochs, device, tokenizer, rank=0):
    '''
    Single epoch of training
    '''
    mlpc.train()
    total_loss = 0
    memo_cnt = 0

    for section, (embeddings, texts) in training.items():
        print(f"Training {section} - Epoch {epoch}/{epochs}")
        
        # Use tqdm for progress bar but without excessive prints
        for i in tqdm(range(0, len(embeddings), 1), desc=f"Epoch {epoch}/{epochs}"):
            batch_embd = embeddings[i:i+1].to(device)
            batch_text = texts[i:i+1]
            tokens = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
            
            optimizer.zero_grad() # not updating weights
            soft_prompts = mlpc(batch_embd) # get soft prompts
            outputs = frozen_mistral(soft_prompts) # generate text from soft prompts
            # had a shape mismatch error 
            logits = outputs.logits[:, :-1, :]  
            labels = tokens.input_ids[:, 1:]   
            # shape checking and reshaping
            if logits.shape[1] != labels.shape[1]:
                min_len = min(logits.shape[1], labels.shape[1])
                logits = logits[:, :min_len, :]
                labels = labels[:, :min_len]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()
            
            if isinstance(mlpc, nn.parallel.DistributedDataParallel):
                nn.utils.clip_grad_norm_(mlpc.module.parameters(), 1.0)
            else:
                nn.utils.clip_grad_norm_(mlpc.parameters(), 1.0)
                
            optimizer.step()
            total_loss += loss.item() * len(batch_text)
            memo_cnt += len(batch_text)
    
    if dist.is_initialized():
        torch.cuda.synchronize(device) # synch step, very important
        dist.barrier()
        loss_tensor = torch.tensor([total_loss, memo_cnt], dtype=torch.float32, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, memo_cnt = loss_tensor.tolist()

    return total_loss / memo_cnt if memo_cnt > 0 else 0

def validate(mlpc, validation, frozen_mistral, loss_fn, device, tokenizer, rank=0):
    '''
    Single epoch of validation
    same as train except with val data...
    validated every epoch, i don't know if this is excessive but i read that it's the best practice
    '''
    mlpc.eval()
    total_loss = 0
    memo_cnt = 0
    with torch.no_grad():
        for section, (embeddings, texts) in validation.items():
            print(f"Validating {section}...")
            for i in tqdm(range(0, len(embeddings), 1), desc="Validation"):
                batch_embd = embeddings[i:i+1].to(device)
                batch_text = texts[i:i+1]
                tokens = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
                soft_prompts = mlpc(batch_embd)
                outputs = frozen_mistral(soft_prompts)
                labels = tokens.input_ids[:, 1:] # original tokens (what model should predict)
                logits = outputs.logits[:, :-1, :] # mistral predictions
                # same shape mismatch stuff since output sequence is longer than input because soft prompts adds more tokens
                if logits.shape[1] != labels.shape[1]:
                    min_len = min(logits.shape[1], labels.shape[1])
                    logits = logits[:, :min_len, :]
                    labels = labels[:, :min_len]
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                total_loss += loss.item() * len(batch_text)
                memo_cnt += len(batch_text)
    if dist.is_initialized():
        torch.cuda.synchronize(device)
        dist.barrier()
        loss_tensor = torch.tensor([total_loss, memo_cnt], dtype=torch.float32, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, memo_cnt = loss_tensor.tolist()

    return total_loss / memo_cnt if memo_cnt > 0 else 0

'''
arg parser:
wrote a slurm script for each section/encoder combo (8 total) 
that can be easily run with these args
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SoftSRV model")
    parser.add_argument("--section", type=str, default="issue_statements", 
                        help="Section to train on")
    parser.add_argument("--encoder", type=str, default="bert", 
                        help="Encoder (Bert or Legal-Bert)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint")
    args = parser.parse_args()

    main_config = load_config()
    
    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
    
    evaluate_section(main_config, args.encoder, args.section, args.checkpoint)
