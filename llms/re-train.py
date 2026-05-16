import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import random_split

from model import TransformerModel
from save_config import save_training_config
from train import (
    EPOCHS,
    LEARNING_RATE,
    MAX_LR,
    TARGET_GPU_MEMORY_FRACTION,
    WEIGHT_DECAY,
    TensorizedTextDataset,
    autotune_micro_batch_size,
    choose_accumulation_steps,
    get_device,
    make_loader,
    move_batch,
    optimizer_steps_per_epoch,
    should_use_amp,
)


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "dados.txt"
RETRAIN_DIR = BASE_DIR / "re-train"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model_weights.pth"


class SavedBPETokenizer:
    """Tokenizer BPE reconstruido a partir do config.json salvo."""

    def __init__(self, vocab_mappings):
        self.word_to_idx = vocab_mappings["word_to_idx"]
        self.idx_to_word = {
            int(idx): token
            for idx, token in vocab_mappings["idx_to_word"].items()
        }
        self.merges = [tuple(pair) for pair in vocab_mappings.get("merges", [])]
        self.SPECIAL_TOKENS = vocab_mappings.get(
            "special_tokens",
            ["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        )
        self.special_tokens = set(self.SPECIAL_TOKENS)
        self.unk_id = self.word_to_idx.get("<UNK>", 1)
        self.vocab_size = len(self.word_to_idx)

    def encode(self, text):
        ids = []
        for word in text.lower().split():
            for token in self._tokenize_word(word):
                ids.append(self.word_to_idx.get(token, self.unk_id))
        return ids

    def decode(self, ids):
        tokens = [
            self.idx_to_word.get(int(token_id), "<UNK>")
            for token_id in ids
        ]
        text = " ".join(
            token
            for token in tokens
            if token not in self.special_tokens
        )
        return text.replace("Ä ", " ").replace("</w>", "").strip()

    @staticmethod
    def _word_to_chars(word):
        if not word:
            return []

        chars = list(word)
        chars[-1] = chars[-1] + "</w>"
        return chars

    def _tokenize_word(self, word):
        chars = self._word_to_chars(word)

        for left, right in self.merges:
            merged = []
            idx = 0

            while idx < len(chars):
                if idx < len(chars) - 1 and chars[idx] == left and chars[idx + 1] == right:
                    merged.append(left + right)
                    idx += 2
                else:
                    merged.append(chars[idx])
                    idx += 1

            chars = merged

        return chars


class SavedTokenizerTextDataset(torch.utils.data.Dataset):
    """Dataset que usa o tokenizador salvo, sem treinar um vocabulario novo."""

    def __init__(self, text, tokenizer, seq_len):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.tokens = tokenizer.encode(text)
        print(f"Total de tokens no texto usando tokenizer salvo: {len(self.tokens)}")

    def __len__(self):
        return max(1, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        end_idx = min(idx + self.seq_len, len(self.tokens) - 1)
        start_idx = end_idx - self.seq_len

        x = torch.tensor(self.tokens[start_idx:end_idx], dtype=torch.long)
        y = torch.tensor(self.tokens[start_idx + 1:end_idx + 1], dtype=torch.long)

        if len(x) < self.seq_len:
            pad = self.seq_len - len(x)
            x = torch.cat([x, torch.zeros(pad, dtype=torch.long)])
            y = torch.cat([y, torch.zeros(pad, dtype=torch.long)])

        return x, y

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def itos(self, idx):
        return self.decode(self.tokens[idx:idx + self.seq_len])

    def stoi(self, text):
        return self.encode(text)

    def itos_vocab(self, idx):
        return self.tokenizer.idx_to_word.get(idx, "<UNK>")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_checkpoint_paths():
    """Usa ./re-train se existir; na primeira vez, usa os arquivos atuais."""
    RETRAIN_DIR.mkdir(exist_ok=True)

    retrain_config = RETRAIN_DIR / CONFIG_NAME
    retrain_weights = RETRAIN_DIR / WEIGHTS_NAME

    if retrain_config.exists() and retrain_weights.exists():
        print("Carregando checkpoint de continuidade em: re-train/")
        return retrain_config, retrain_weights

    base_config = BASE_DIR / CONFIG_NAME
    base_weights = BASE_DIR / WEIGHTS_NAME

    if not base_config.exists() or not base_weights.exists():
        raise FileNotFoundError(
            "Nao encontrei config.json/model_weights.pth nem em re-train/ nem no diretorio atual."
        )

    print("Primeiro re-treino: usando config.json e model_weights.pth do diretorio atual.")
    return base_config, base_weights


def build_model_from_config(config, weights_path, device):
    model_config = config["model_config"]
    model = TransformerModel(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        num_layers=model_config["num_layers"],
        max_seq_len=model_config["max_seq_len"],
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Pesos carregados de: {weights_path}")
    return model


def get_training_value(config, key, default):
    return config.get("training_config", {}).get(key, default)


if __name__ == "__main__":
    config_path, weights_path = resolve_checkpoint_paths()
    config = load_json(config_path)

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    device = get_device()
    use_amp = should_use_amp(device)

    seq_len = config["model_config"].get(
        "max_seq_len",
        config.get("dataset_info", {}).get("seq_len", 64),
    )
    tokenizer = SavedBPETokenizer(config["vocab_mappings"])
    dataset = TensorizedTextDataset(SavedTokenizerTextDataset(text, tokenizer, seq_len))
    n_samples = len(dataset)
    vocab_size = dataset.vocab_size

    if vocab_size != config["model_config"]["vocab_size"]:
        raise ValueError(
            f"Vocabulario do config ({config['model_config']['vocab_size']}) "
            f"nao bate com o tokenizer salvo ({vocab_size})."
        )

    train_size = max(1, int(0.8 * n_samples))
    val_size = max(0, n_samples - train_size)
    if val_size == 0 and n_samples > 1:
        train_size = n_samples - 1
        val_size = 1

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    model = build_model_from_config(config, weights_path, device)

    micro_batch_size = autotune_micro_batch_size(
        model=model,
        seq_len=seq_len,
        vocab_size=vocab_size,
        device=device,
        use_amp=use_amp,
        max_train_size=train_size,
    )
    accumulation_steps = choose_accumulation_steps(n_samples, micro_batch_size)
    effective_batch_size = micro_batch_size * accumulation_steps

    train_loader = make_loader(train_dataset, micro_batch_size, True, device)
    val_loader = make_loader(val_dataset, micro_batch_size, False, device)
    optimizer_updates_per_epoch = optimizer_steps_per_epoch(len(train_loader), accumulation_steps)

    print(
        f"Re-treino: {n_samples} amostras | treino={train_size} | val={val_size} | "
        f"micro_batch={micro_batch_size} | accumulation={accumulation_steps} | "
        f"batch efetivo={effective_batch_size}"
    )
    print(f"Steps de otimizacao por epoca: {optimizer_updates_per_epoch}")

    learning_rate = get_training_value(config, "learning_rate", LEARNING_RATE)
    weight_decay = get_training_value(config, "weight_decay", WEIGHT_DECAY)
    epochs = EPOCHS

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=optimizer_updates_per_epoch,
        epochs=epochs,
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    last_train_loss = float("nan")
    causal_mask = model.generate_causal_mask(seq_len).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_total = 0.0
        optimizer_updates = 0

        for step, batch in enumerate(train_loader):
            x, y = move_batch(batch, device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x, causal_mask)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                scaled_loss = loss / accumulation_steps

            scaler.scale(scaled_loss).backward()
            train_loss_total += loss.item()
            last_train_loss = loss.item()

            is_accumulation_boundary = (step + 1) % accumulation_steps == 0
            is_last_batch = (step + 1) == len(train_loader)
            if is_accumulation_boundary or is_last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1

        model.eval()
        val_loss_total = 0.0

        if len(val_loader) > 0:
            with torch.no_grad():
                for batch in val_loader:
                    x, y = move_batch(batch, device)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(x, causal_mask)
                        val_loss_total += criterion(logits.reshape(-1, vocab_size), y.reshape(-1)).item()

            val_loss = val_loss_total / len(val_loader)
        else:
            val_loss = last_train_loss

        train_loss = train_loss_total / max(1, len(train_loader))

        if epoch % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | lr={lr:.2e} | updates={optimizer_updates}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping na epoca {epoch + 1}: val_loss nao melhorou por {patience} epocas")
                break

    training_args = {
        "seq_len": seq_len,
        "micro_batch_size": micro_batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "max_lr": MAX_LR,
        "weight_decay": weight_decay,
        "epochs": epoch + 1,
        "device": device,
        "use_amp": use_amp,
        "target_gpu_memory_fraction": TARGET_GPU_MEMORY_FRACTION,
        "train_size": train_size,
        "val_size": val_size,
        "best_val_loss": best_val_loss,
        "timestamp": datetime.now().isoformat(),
        "source_checkpoint": str(weights_path),
    }

    output_config = RETRAIN_DIR / CONFIG_NAME
    output_weights = RETRAIN_DIR / WEIGHTS_NAME

    save_training_config(
        num_heads=model.num_heads,
        device=device,
        model=model,
        dataset=dataset,
        args=training_args,
        output_path=output_config,
    )
    torch.save(model.state_dict(), output_weights)
    print(f"Pesos do re-treino salvos em: {output_weights}")
