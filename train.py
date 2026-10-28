"""
Training script for nanoGPT Transformer Model.
Supports training from scratch and resuming from checkpoint.pt with Cosine LR scheduling.
"""

import os
import argparse
import torch
from model import BigramLanguageModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train nanoGPT Transformer")
    parser.add_argument('--data_path', type=str, required=True, help='Path to text training dataset')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint.pt')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pt', help='Path to save/load checkpoint')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=250)
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--n_emb', type=int, default=384)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=1337)
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load dataset
    with open(args.data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Dataset length: {len(text):,} characters | Unique characters (vocab size): {vocab_size}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
        x = torch.stack([d[i:i + args.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_emb=args.n_emb,
        block_size=args.block_size,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iters, eta_min=args.min_lr)

    start_iter = 0
    best_val_loss = float('inf')

    if args.resume and os.path.exists(args.checkpoint_path):
        print(f"Resuming from checkpoint '{args.checkpoint_path}'...")
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_iter = ckpt.get('iter', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed at iteration {start_iter - 1} with best val loss: {best_val_loss:.4f}")

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    print(f"Starting training from step {start_iter} to {start_iter + args.max_iters}...")
    for iter_num in range(start_iter, start_iter + args.max_iters):
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss()
            print(f"Step {iter_num:6d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                print(f"  --> Saving new best checkpoint to '{args.checkpoint_path}' (val_loss: {best_val_loss:.4f})")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter': iter_num,
                    'best_val_loss': best_val_loss,
                }, args.checkpoint_path)

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()

