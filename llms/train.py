from torch.utils.data import random_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TransformerModel
from tokenizer import TextDataset
from save_config import save_training_config

with open("./dataset/dados.txt", "r", encoding="utf-8") as f:
    text = f.read()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    
    # ── Bônus: ajusta hiperparâmetros ao tamanho do dataset ──────────────────
    dataset = TextDataset(text, seq_len=64)   # sua classe de dataset
    n_samples = len(dataset)
    vocab_size = dataset.vocab_size
    
    # Regras simples: dataset pequeno → modelo menor; grande → modelo maior
    if n_samples < 500:
        d_model, num_heads, d_ff, num_layers = 64,  2,  256, 2
    elif n_samples < 5_000:
        d_model, num_heads, d_ff, num_layers = 128, 4,  512, 4
    else:
        d_model, num_heads, d_ff, num_layers = 256, 8, 1024, 6

    print(f"Dataset: {n_samples} amostras → d_model={d_model}, layers={num_layers}")


    
    # ── Split treino / validação ──────────────────────────────────────────────
    train_size = int(0.8 * n_samples)
    val_size   = n_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    
    # ── Configurações para infraestrutura pequena ─────────────────────────────
    micro_batch_size    = 4          # cabe na GPU/CPU pequena
    accumulation_steps  = 8          # batch efetivo = 4 × 8 = 32
    effective_batch_size = micro_batch_size * accumulation_steps

    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=micro_batch_size, shuffle=False)



    
    # ── Modelo ────────────────────────────────────────────────────────────────
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    epochs = 1000

    # Alternativa: scheduler baseado em validação (recomendado para datasets pequenos)
    # Calcular corretamente o número de steps de otimização por época
    steps_per_optimizer_step = len(train_loader) // accumulation_steps
    
    # Se for 0, ajustamos o accumulation_steps ou usamos 1 como mínimo
    if steps_per_optimizer_step == 0:
        print(f"AVISO: accumulation_steps ({accumulation_steps}) > batches ({len(train_loader)})")
        print(f"Ajustando accumulation_steps para {len(train_loader)}")
        accumulation_steps = len(train_loader)
        steps_per_optimizer_step = 1
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        steps_per_epoch=steps_per_optimizer_step,  # agora será pelo menos 1
        epochs=epochs,
        pct_start=0.1,
    )
    

    
    # ── Early Stopping (sem salvar checkpoint) ────────────────────────────────
    best_val_loss    = float("inf")
    patience         = 10
    patience_counter = 0

    

    # ── Loop de treino ────────────────────────────────────────────────────────
    for epoch in range(epochs):

        # — Treino com gradient accumulation —
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            X, y = batch
            X, y = X.to(device), y.to(device)

            mask   = model.generate_causal_mask(X.shape[1]).to(device)
            logits = model(X, mask)
            loss   = criterion(logits.view(-1, vocab_size), y.view(-1))

            # Normaliza pelo nº de passos acumulados
            (loss / accumulation_steps).backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # — Validação —
        model.eval()
        val_loss_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                X, y = batch
                X, y = X.to(device), y.to(device)

                mask   = model.generate_causal_mask(X.shape[1]).to(device)
                logits = model(X, mask)
                val_loss_total += criterion(logits.view(-1, vocab_size), y.view(-1)).item()

        val_loss = val_loss_total / len(val_loader)

        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f}")

        # — Early Stopping —
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0          # melhora → reseta contador
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping na época {epoch+1} (val_loss não melhorou por {patience} épocas)")
                break

    # ── Salvar config do treino para inferência futura ────────────────────────
    from datetime import datetime
    import json
    import os

    # Salvar configuração para inferência
    training_args = {
        "seq_len": 64,
        "micro_batch_size": micro_batch_size,
        "accumulation_steps": accumulation_steps,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "epochs": epoch + 1,  # número real de épocas treinadas
        "device": device,
        "train_size": train_size,
        "val_size": val_size,
        "best_val_loss": best_val_loss,
        "timestamp": datetime.now().isoformat()
    }

    save_training_config(num_heads=model.num_heads, device=device, model=model, dataset=dataset, args=training_args, output_path="config.json")

    # Salvar os pesos também
    torch.save(model.state_dict(), "model_weights.pth")


    # save_training_config(device, model, dataset, {
    #     "seq_len": dataset.seq_len,
    #     "train_size": train_size,
    #     "val_size": val_size,
    #     "best_val_loss": best_val_loss,
    #     "timestamp": torch.tensor(epoch).item()  # Exemplo de timestamp (pode ser data/hora real)
    # }, output_path="config.json")  

