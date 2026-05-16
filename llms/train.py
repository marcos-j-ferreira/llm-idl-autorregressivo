import math
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import TransformerModel
from save_config import save_training_config
from tokenizer import TextDataset

# ============================================================================
# CONFIGURACOES GERAIS DO TREINO
# ----------------------------------------------------------------------------
# Estes valores controlam o comprimento das sequencias, uso maximo de memoria da
# GPU, quantidade maxima de epocas e hiperparametros basicos do otimizador.
# ============================================================================
SEQ_LEN = 64
TARGET_GPU_MEMORY_FRACTION = 0.85
EPOCHS = 1000
LEARNING_RATE = 1e-3
MAX_LR = 5e-4
WEIGHT_DECAY = 1e-5

# ============================================================================
# DATASET TENSORIZADO
# ----------------------------------------------------------------------------
# O TextDataset original cria tensores no __getitem__. Esta classe transforma as
# janelas X/y uma unica vez em tensores de CPU, reduzindo trabalho repetido da
# CPU durante o treino e ajudando a manter a GPU mais alimentada.
# ============================================================================
class TensorizedTextDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.seq_len = base_dataset.seq_len
        self.vocab_size = base_dataset.vocab_size
        self.tokenizer = getattr(base_dataset, "tokenizer", None)

        tokens = torch.tensor(base_dataset.tokens, dtype=torch.long)
        if tokens.numel() <= self.seq_len:
            pad = self.seq_len + 1 - tokens.numel()
            tokens = torch.cat([tokens, torch.zeros(pad, dtype=torch.long)])

        windows = tokens.unfold(0, self.seq_len + 1, 1)
        self.x = windows[:, :-1].contiguous()
        self.y = windows[:, 1:].contiguous()
        print(f"Dataset tensorizado em memoria: {len(self.x)} janelas de {self.seq_len} tokens")

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def encode(self, text):
        return self.base_dataset.encode(text)

    def decode(self, ids):
        return self.base_dataset.decode(ids)

    def itos(self, idx):
        return self.base_dataset.itos(idx)

    def stoi(self, text):
        return self.base_dataset.stoi(text)

    def itos_vocab(self, idx):
        return self.base_dataset.itos_vocab(idx)


# ============================================================================
# DETECCAO DE DISPOSITIVO E LIMITACAO DA GPU
# ----------------------------------------------------------------------------
# Seleciona CUDA quando disponivel, mostra informacoes da placa e limita o
# processo a aproximadamente 85% da VRAM para evitar ocupar a placa inteira.
# ============================================================================
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type != "cuda":
        print("AVISO: CUDA nao esta disponivel neste Python. O treino vai rodar na CPU.")
        print("Dica: rode com o Python do venv CUDA, por exemplo:")
        print(r"  C:\Users\BAHREN\Desktop\home\llms-gpu\venv_ml\Scripts\python.exe train.py")
        return device

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024**3

    print(f"GPU: {props.name} | VRAM total: {total_gb:.2f} GB | livre agora: {free_gb:.2f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")

    try:
        torch.cuda.set_per_process_memory_fraction(TARGET_GPU_MEMORY_FRACTION, 0)
        print(f"Limite de memoria do processo CUDA: {TARGET_GPU_MEMORY_FRACTION:.0%}")
    except RuntimeError as exc:
        print(f"AVISO: nao consegui limitar a memoria CUDA automaticamente: {exc}")

    torch.backends.cudnn.benchmark = True
    return device



# ============================================================================
# ESCOLHA INTELIGENTE DO TAMANHO DO MODELO
# ----------------------------------------------------------------------------
# Ajusta d_model, numero de cabecas, feed-forward e camadas usando o tamanho do
# dataset e a memoria da GPU. Em placas pequenas, limita o modelo para reduzir
# risco de erro de memoria.
# ============================================================================
def choose_model_config(n_samples, vocab_size, device):
    if n_samples < 500:
        config = {"d_model": 64, "num_heads": 2, "d_ff": 256, "num_layers": 2}
    elif n_samples < 5_000:
        config = {"d_model": 128, "num_heads": 4, "d_ff": 512, "num_layers": 4}
    else:
        config = {"d_model": 192, "num_heads": 6, "d_ff": 768, "num_layers": 6}

    if device.type == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        if total_gb < 5:
            config["d_model"] = min(config["d_model"], 128)
            config["num_heads"] = min(config["num_heads"], 4)
            config["d_ff"] = min(config["d_ff"], 512)
            config["num_layers"] = min(config["num_layers"], 4)

        if n_samples > 20_000 and total_gb >= 6:
            config = {"d_model": 256, "num_heads": 8, "d_ff": 1024, "num_layers": 6}

    print(
        "Modelo escolhido: "
        f"vocab={vocab_size}, d_model={config['d_model']}, heads={config['num_heads']}, "
        f"d_ff={config['d_ff']}, layers={config['num_layers']}"
    )
    return config



# ============================================================================
# MIXED PRECISION
# ----------------------------------------------------------------------------
# AMP pode acelerar GPUs modernas, mas a GTX 750 e outras placas antigas tendem
# a nao ganhar com isso. Por isso o script liga AMP apenas quando a GPU suporta
# bem esse tipo de treino.
# ============================================================================
def should_use_amp(device):
    if device.type != "cuda":
        return False

    major, _ = torch.cuda.get_device_capability(0)
    if major < 7:
        print("AMP desligado: GPUs antigas como a GTX 750 costumam treinar melhor em float32.")
        return False

    print("AMP ligado: usando mixed precision para acelerar GPUs modernas.")
    return True


# ============================================================================
# DATALOADER OTIMIZADO
# ----------------------------------------------------------------------------
# Cria DataLoaders com pin_memory quando CUDA esta ativo. Isso deixa a copia dos
# batches da CPU para GPU mais eficiente quando combinado com non_blocking=True.
# ============================================================================
def make_loader(dataset, batch_size, shuffle, device):
    use_cuda = device.type == "cuda"
    workers = 0

    if use_cuda and os.name != "nt":
        workers = min(4, os.cpu_count() or 1)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=use_cuda,
        num_workers=workers,
        persistent_workers=workers > 0,
    )



# ============================================================================
# CALCULO DE STEPS DO OTIMIZADOR
# ----------------------------------------------------------------------------
# O scheduler precisa saber quantas vezes optimizer.step() acontece em cada
# epoca. Como existe gradient accumulation, esse numero nao e igual ao total de
# batches do DataLoader.
# ============================================================================
def optimizer_steps_per_epoch(train_batches, accumulation_steps):
    return max(1, math.ceil(train_batches / accumulation_steps))



# ============================================================================
# AUTOAJUSTE DO MICRO BATCH NA GPU
# ----------------------------------------------------------------------------
# Testa tamanhos de batch diretamente na GPU e fica com o maior que cabe sem
# estourar VRAM. Isso ajuda a usar melhor a placa sem precisar adivinhar valores.
# ============================================================================
def autotune_micro_batch_size(model, seq_len, vocab_size, device, use_amp, max_train_size):
    if device.type != "cuda":
        return 4

    model.train()
    candidates = [4, 8, 16, 24, 32, 48, 64, 96, 128]
    candidates = [batch for batch in candidates if batch <= max(1, max_train_size)]
    if not candidates:
        return 1

    best_batch = candidates[0]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("Autoajustando micro_batch_size na GPU...")
    for batch_size in candidates:
        try:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            mask = model.generate_causal_mask(seq_len).to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x, mask)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            scaler.scale(loss).backward()
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  batch {batch_size:>3}: OK | pico alocado {peak_gb:.2f} GB")
            best_batch = batch_size
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {batch_size:>3}: sem VRAM suficiente")
            torch.cuda.empty_cache()
            break

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return best_batch



# ============================================================================
# GRADIENT ACCUMULATION
# ----------------------------------------------------------------------------
# Define quantos micro batches serao acumulados antes de atualizar os pesos. O
# objetivo e manter batch efetivo razoavel mesmo quando a VRAM limita o batch
# real que cabe na GPU.
# ============================================================================
def choose_accumulation_steps(n_samples, micro_batch_size):
    target_effective_batch = min(256, max(32, n_samples // 8))
    return max(1, math.ceil(target_effective_batch / micro_batch_size))


# ============================================================================
# TRANSFERENCIA DE BATCH PARA GPU
# ----------------------------------------------------------------------------
# Move X e y para o device escolhido. Quando CUDA e pin_memory estao ativos,
# non_blocking=True permite transferencia mais eficiente.
# ============================================================================
def move_batch(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ============================================================================
# PONTO DE ENTRADA DO SCRIPT
# ----------------------------------------------------------------------------
# Tudo abaixo so executa quando o arquivo e chamado diretamente com python
# train.py. Isso evita que o treino comece por acidente ao importar funcoes.
# ============================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # Leitura do texto bruto usado para treinar o tokenizador e o modelo.
    # ------------------------------------------------------------------------
    with open("./dataset/dados.txt", "r", encoding="utf-8") as f:
        text = f.read()


    # ------------------------------------------------------------------------
    # Escolha do device e configuracao de AMP.
    # ------------------------------------------------------------------------
    device = get_device()
    use_amp = should_use_amp(device)

    # ------------------------------------------------------------------------
    # Criacao do dataset, tokenizacao BPE e materializacao das janelas X/y.
    # ------------------------------------------------------------------------
    dataset = TensorizedTextDataset(TextDataset(text, seq_len=SEQ_LEN))
    n_samples = len(dataset)
    vocab_size = dataset.vocab_size


    # ------------------------------------------------------------------------
    # Divisao em treino e validacao com seed fixa para reproducibilidade.
    # ------------------------------------------------------------------------
    train_size = max(1, int(0.8 * n_samples))
    val_size = max(0, n_samples - train_size)
    if val_size == 0 and n_samples > 1:
        train_size = n_samples - 1
        val_size = 1

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)



    # ------------------------------------------------------------------------
    # Criacao do modelo Transformer com parametros ajustados automaticamente.
    # ------------------------------------------------------------------------
    model_config = choose_model_config(n_samples, vocab_size, device)
    model = TransformerModel(
        vocab_size=vocab_size,
        max_seq_len=SEQ_LEN,
        **model_config,
    ).to(device)


    # ------------------------------------------------------------------------
    # Autoajuste de batch e acumulacao para usar melhor a GPU disponivel.
    # ------------------------------------------------------------------------
    micro_batch_size = autotune_micro_batch_size(
        model=model,
        seq_len=SEQ_LEN,
        vocab_size=vocab_size,
        device=device,
        use_amp=use_amp,
        max_train_size=train_size,
    )
    accumulation_steps = choose_accumulation_steps(n_samples, micro_batch_size)
    effective_batch_size = micro_batch_size * accumulation_steps

    # ------------------------------------------------------------------------
    # DataLoaders de treino e validacao.
    # ------------------------------------------------------------------------
    train_loader = make_loader(train_dataset, micro_batch_size, True, device)
    val_loader = make_loader(val_dataset, micro_batch_size, False, device)

    optimizer_updates_per_epoch = optimizer_steps_per_epoch(len(train_loader), accumulation_steps)
    print(
        f"Dataset: {n_samples} amostras | treino={train_size} | val={val_size} | "
        f"micro_batch={micro_batch_size} | accumulation={accumulation_steps} | "
        f"batch efetivo={effective_batch_size}"
    )
    print(f"Steps de otimizacao por epoca: {optimizer_updates_per_epoch}")



    # ------------------------------------------------------------------------
    # Loss, otimizador, scheduler e scaler de AMP.
    # ------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=optimizer_updates_per_epoch,
        epochs=EPOCHS,
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)



    # ------------------------------------------------------------------------
    # Estado de early stopping e mascara causal fixa para o tamanho de sequencia.
    # ------------------------------------------------------------------------
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    last_train_loss = float("nan")

    causal_mask = model.generate_causal_mask(SEQ_LEN).to(device)

    # ------------------------------------------------------------------------
    # Loop principal de treino e validacao.
    # ------------------------------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_total = 0.0
        optimizer_updates = 0

        # --------------------------------------------------------------------
        # Treino: forward, loss, backward e atualizacao com accumulation.
        # --------------------------------------------------------------------
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



        # --------------------------------------------------------------------
        # Validacao: mede a loss em dados separados sem calcular gradientes.
        # --------------------------------------------------------------------
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


        # --------------------------------------------------------------------
        # Logs periodicos para acompanhar loss, learning rate e updates.
        # --------------------------------------------------------------------
        if epoch % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | lr={lr:.2e} | updates={optimizer_updates}"
            )


        # --------------------------------------------------------------------
        # Early stopping: encerra quando a validacao para de melhorar.
        # --------------------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping na epoca {epoch + 1}: val_loss nao melhorou por {patience} epocas")
                break


    # ------------------------------------------------------------------------
    # Metadados do treino salvos junto com a configuracao para inferencia.
    # ------------------------------------------------------------------------
    training_args = {
        "seq_len": SEQ_LEN,
        "micro_batch_size": micro_batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "learning_rate": LEARNING_RATE,
        "max_lr": MAX_LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": epoch + 1,
        "device": device,
        "use_amp": use_amp,
        "target_gpu_memory_fraction": TARGET_GPU_MEMORY_FRACTION,
        "train_size": train_size,
        "val_size": val_size,
        "best_val_loss": best_val_loss,
        "timestamp": datetime.now().isoformat(),
    }





    # ------------------------------------------------------------------------
    # Salvamento da configuracao, vocabulario/tokenizador e pesos do modelo.
    # ------------------------------------------------------------------------
    save_training_config(
        num_heads=model.num_heads,
        device=device,
        model=model,
        dataset=dataset,
        args=training_args,
        output_path="config.json",
    )
    torch.save(model.state_dict(), "model_weights.pth")
    print("Pesos salvos em: model_weights.pth")
