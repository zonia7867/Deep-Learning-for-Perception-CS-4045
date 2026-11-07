"""Question 3 starter - recurrent networks: vanilla RNN vs LSTM (and GRU).

Dataset : IMDB Large Movie Review Dataset, 25k train / 25k test, binary sentiment.
Models  : torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU behind one wrapper. The recurrence
          itself is PyTorch's - you are not implementing gates by hand. What you own is
          the sequence handling, the experiments and the explanation.

    python starter/q3_rnn_lstm.py --cell rnn
    python starter/q3_rnn_lstm.py --cell lstm
    python starter/q3_rnn_lstm.py --cell gru
    python starter/q3_rnn_lstm.py --cell lstm --max-len 500        # length study (3.3)
    python starter/q3_rnn_lstm.py --cell lstm --bidirectional      # optional variant
    python starter/q3_rnn_lstm.py --cell rnn --grad-flow           # figure for 3.4
    python starter/q3_rnn_lstm.py --cell lstm --no-clip            # exploding gradients

Outputs in runs/q3/<tag>/: results.json, curves.png, grad_norms.json, best.pt
This question is comfortable on CPU: roughly 3-6 minutes per epoch, under 2 on a T4.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.common import count_parameters, get_device, load_config, save_results, set_seed
from utils.metrics import per_timestep_grad_norms
from utils.text import ImdbDataset, build_vocab, collate_pad, read_imdb_split
from utils.viz import plot_curves, plot_grad_norms

ROOT = Path(__file__).resolve().parents[1]
CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


# -------------------------------------------------------------------------- model
class SequenceClassifier(nn.Module):
    """Embedding -> recurrent layer -> linear readout.

    The recurrent layer is PyTorch's. Two things about it are yours to get right, and
    both are marked: packing the padded batch, and choosing the readout.
    """

    def __init__(self, cell: str, vocab_size: int, embed_dim: int = 100,
                 hidden_size: int = 128, num_layers: int = 1, bidirectional: bool = False,
                 dropout: float = 0.3, num_classes: int = 2, readout: str = "last") -> None:
        super().__init__()
        if cell not in CELLS:
            raise ValueError("cell must be one of {}".format(sorted(CELLS)))
        self.cell_type = cell
        self.readout = readout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_kwargs = dict(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if cell == "rnn":
            # tanh is the classic vanilla recurrence; 'relu' is also accepted here and is
            # worth a sentence in the report if you try it.
            rnn_kwargs["nonlinearity"] = "tanh"
        self.rnn = CELLS[cell](**rnn_kwargs)
        self.dropout = nn.Dropout(dropout)
        directions = 2 if bidirectional else 1
        self.head = nn.Linear(hidden_size * directions, num_classes)

    def forward(self, tokens, lengths):
        """tokens: (batch, time) padded with 0. lengths: (batch,) true lengths, on CPU."""
        embedded = self.embedding(tokens)             # (batch, time, embed)

        # ------------------------------------------------------------------ TODO 1
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # --------------------------------------------------------------- end TODO 1

        # LSTM returns (h_n, c_n); RNN and GRU return h_n only.
        h_n = hidden[0] if isinstance(hidden, tuple) else hidden

        # ------------------------------------------------------------------ TODO 2
        if self.readout == "last":
            if self.bidirectional:
                features = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                features = h_n[-1]
        elif self.readout == "mean":
            device = output.device
            mask = (torch.arange(output.size(1), device=device)[None, :] < lengths.to(device)[:, None]).unsqueeze(-1).float()
            features = (output * mask).sum(dim=1) / lengths.to(device).unsqueeze(-1).float()
        else:
            raise ValueError("readout must be 'last' or 'mean'")
        # --------------------------------------------------------------- end TODO 2

        return self.head(self.dropout(features))

# --------------------------------------------------------------------------- data
def get_loaders(cfg, max_len: int, batch_size: int):
    data_root = ROOT / "data"
    train_rows = read_imdb_split(data_root, "train")
    test_rows = read_imdb_split(data_root, "test")

    vocab = build_vocab([t for t, _ in train_rows],
                        max_size=cfg["q3_sequence"]["vocab_size"], min_freq=2)
    train_set = ImdbDataset(train_rows, vocab, max_len=max_len)
    test_set = ImdbDataset(test_rows, vocab, max_len=max_len)

    print("[data] vocab {} | train {} | test {}".format(
        len(vocab), len(train_set), len(test_set)))
    print("[data] train length stats:", train_set.length_stats())
    print("[data] OOV rate on train sample:", train_set.oov_rate(train_rows))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_pad, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_pad, num_workers=0)
    return train_loader, test_loader, vocab, train_set


# -------------------------------------------------------------------------- loops
def run_epoch(model, loader, criterion, device, optimiser=None, clip: float = 5.0):
    training = optimiser is not None
    model.train() if training else model.eval()
    loss_sum, correct, seen = 0.0, 0, 0
    grad_norm_sum, steps = 0.0, 0

    for tokens, lengths, labels in tqdm(loader, leave=False,
                                        desc="train" if training else "eval"):
        tokens, labels = tokens.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(tokens, lengths)
            loss = criterion(logits, labels)
        if training:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            # Total gradient norm BEFORE clipping - this is the number that tells you
            # whether the gradients are exploding (watch it for the vanilla RNN).
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip if clip > 0 else float("inf"))
            grad_norm_sum += float(total_norm)
            steps += 1
            optimiser.step()

        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)

    out = {"loss": loss_sum / seen, "acc": correct / seen}
    if training:
        out["mean_grad_norm"] = round(grad_norm_sum / max(steps, 1), 3)
    return out


def main() -> None:
    cfg = load_config()
    q3 = cfg["q3_sequence"]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default=q3["cell"], choices=sorted(CELLS))
    ap.add_argument("--max-len", type=int, default=q3["max_len"])
    ap.add_argument("--hidden-size", type=int, default=q3["hidden_size"])
    ap.add_argument("--num-layers", type=int, default=q3["num_layers"])
    ap.add_argument("--epochs", type=int, default=q3["epochs"])
    ap.add_argument("--batch-size", type=int, default=q3["batch_size"])
    ap.add_argument("--lr", type=float, default=q3["lr"])
    ap.add_argument("--readout", default="last", choices=["last", "mean"])
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--no-clip", action="store_true",
                    help="disable gradient clipping, for the exploding-gradient discussion")
    ap.add_argument("--grad-flow", action="store_true",
                    help="also measure the per-timestep gradient norms for task 3.4")
    args = ap.parse_args()

    set_seed(cfg.get("seed", 42))
    device = get_device()
    clip = 0.0 if args.no_clip else q3["clip_grad"]
    print("[device] {} | cell {} | max_len {} | clip {}".format(
        device, args.cell, args.max_len, clip))

    train_loader, test_loader, vocab, train_set = get_loaders(
        cfg, max_len=args.max_len, batch_size=args.batch_size)

    model = SequenceClassifier(
        cell=args.cell,
        vocab_size=len(vocab),
        embed_dim=q3["embed_dim"],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=q3["dropout"],
        readout=args.readout,
    ).to(device)
    total, trainable = count_parameters(model)
    recurrent_params = sum(p.numel() for p in model.rnn.parameters())
    print("[params] total {:,} | recurrent layer {:,}".format(total, recurrent_params))

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    tag = "{}_len{}_h{}{}{}{}".format(
        args.cell, args.max_len, args.hidden_size,
        "_bi" if args.bidirectional else "",
        "_mean" if args.readout == "mean" else "",
        "_noclip" if args.no_clip else "")
    out_dir = ROOT / "runs" / "q3" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, grad_norms_seen = 0.0, []
    wall_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, optimiser, clip=clip)
        ev = run_epoch(model, test_loader, criterion, device)
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(ev["loss"])
        history["val_acc"].append(ev["acc"])
        grad_norms_seen.append(tr["mean_grad_norm"])
        print("epoch {:02d} | train loss {:.4f} acc {:.4f} | test loss {:.4f} acc {:.4f} "
              "| mean grad norm {:.2f}".format(
                  epoch, tr["loss"], tr["acc"], ev["loss"], ev["acc"], tr["mean_grad_norm"]))
        if ev["acc"] > best_acc:
            best_acc = ev["acc"]
            torch.save(model.state_dict(), out_dir / "best.pt")
    train_minutes = (time.perf_counter() - wall_start) / 60.0

    plot_curves(history, out_dir / "curves.png",
                title="Q3 {} (max_len {})".format(args.cell, args.max_len))

    grad_profile = None
    if args.grad_flow:
        # One batch, one backward pass, the gradient norm arriving at every timestep.
        batch = next(iter(train_loader))
        grad_profile = per_timestep_grad_norms(model, batch, criterion, device)
        (out_dir / "grad_norms.json").write_text(
            json.dumps({args.cell: grad_profile}, indent=2), encoding="utf-8")
        plot_grad_norms({args.cell: grad_profile}, out_dir / "grad_norms.png",
                        title="gradient reaching each timestep ({})".format(args.cell))
        first, last = grad_profile[0], grad_profile[-1]
        print("\ngradient norm at t=0 {:.3e} | at t=T {:.3e} | ratio {:.2e}".format(
            first, last, first / max(last, 1e-12)))
        print("Run this for BOTH rnn and lstm, then plot the two profiles on one log axis.")

    save_results({
        "question": "Q3",
        "tag": tag,
        "cell": args.cell,
        "max_len": args.max_len,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "bidirectional": args.bidirectional,
        "readout": args.readout,
        "clip_grad": clip,
        "epochs": args.epochs,
        "vocab_size": len(vocab),
        "total_params": total,
        "recurrent_params": recurrent_params,
        "best_test_acc": round(best_acc, 4),
        "final_train_acc": round(history["train_acc"][-1], 4),
        "mean_grad_norm_per_epoch": grad_norms_seen,
        "grad_norm_first_vs_last": (
            None if grad_profile is None
            else {"t0": grad_profile[0], "tT": grad_profile[-1],
                  "ratio": round(grad_profile[0] / max(grad_profile[-1], 1e-12), 6)}),
        "train_minutes": round(train_minutes, 1),
        "device": str(device),
    }, ROOT / "runs" / "q3")

    print("\nDone. Add this row to your Q3 table, then run the next cell type or length.")


if __name__ == "__main__":
    main()
