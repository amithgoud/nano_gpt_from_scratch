"""
Sample / Inference Script
Generates text autoregressively using a trained checkpoint (checkpoint.pt).
"""

import os
import sys
import json
import argparse
import torch
from model import BigramLanguageModel

def build_or_load_vocab(dataset_path=None, vocab_path='vocab.json'):
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            chars = json.load(f)
        print(f"Loaded {len(chars)} vocabulary characters from '{vocab_path}'.")
    elif dataset_path and os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        print(f"Extracted {len(chars)} vocabulary characters from '{dataset_path}'.")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(chars, f, ensure_ascii=False)
        print(f"Saved vocabulary to '{vocab_path}'.")
    else:
        print("Note: No dataset file or vocab.json provided. Using ASCII printable default character mapping.")
        chars = [chr(i) for i in range(116)]
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos

def main():
    parser = argparse.ArgumentParser(description="Generate text using nanoGPT checkpoint")
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pt', help='Path to checkpoint file')
    parser.add_argument('--dataset', type=str, default=None, help='Path to original text corpus for vocab mapping')
    parser.add_argument('--vocab', type=str, default='vocab.json', help='Path to vocab.json')
    parser.add_argument('--prompt', type=str, default='', help='Starting text prompt')
    parser.add_argument('--max_tokens', type=int, default=500, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling threshold')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Build or load vocab
    chars, stoi, itos = build_or_load_vocab(args.dataset, args.vocab)
    vocab_size = len(chars)

    # Initialize model
    model = BigramLanguageModel(
        vocab_size=vocab_size if vocab_size >= 116 else 116,
        n_emb=384,
        block_size=256,
        n_head=6,
        n_layer=6,
        dropout=0.2
    )

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from '{args.checkpoint}'...")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        if 'iter' in ckpt:
            val_loss = ckpt.get('best_val_loss', 'N/A')
            print(f"Loaded checkpoint at step {ckpt['iter']} (best val loss: {val_loss})")
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found! Generating with uninitialized weights.")

    model.to(args.device)
    model.eval()

    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '?') for i in l])

    if args.prompt:
        context_tokens = encode(args.prompt)
        context = torch.tensor([context_tokens], dtype=torch.long, device=args.device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=args.device)

    print("\n--- Generating Text ---")
    out_tokens = model.generate(context, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)
    generated_text = decode(out_tokens[0].tolist())
    print(generated_text)
    print("\n-----------------------")

if __name__ == '__main__':
    main()

