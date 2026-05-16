
import json
import math
import torch

def _json_safe(value):
    """Converte valores comuns do PyTorch/Python para tipos aceitos por JSON."""
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _get_vocab_mappings(dataset):
    tokenizer = getattr(dataset, "tokenizer", None)

    if tokenizer is not None:
        return {
            "idx_to_word": _json_safe(tokenizer.idx_to_word),
            "word_to_idx": _json_safe(tokenizer.word_to_idx),
            "merges": _json_safe(getattr(tokenizer, "merges", [])),
            "special_tokens": _json_safe(getattr(tokenizer, "SPECIAL_TOKENS", [])),
        }

    return {
        "idx_to_word": _json_safe(getattr(dataset, "idx_to_word", {})),
        "word_to_idx": _json_safe(getattr(dataset, "word_to_idx", {})),
    }


def save_training_config(num_heads, device, model, dataset, args, output_path="config.json"):
    """
    Salva configurações do treino em um arquivo JSON para inferência posterior
    
    Args:
        num_heads: Número de cabeças de atenção
        device: Dispositivo usado para treino (torch.device)
        model: Modelo treinado (TransformerModel)
        dataset: Dataset usado no treino (TextDataset)
        args: Dicionário com argumentos do treino
        output_path: Caminho para salvar o config.json
    """
    
    config = {
        # Configurações do modelo
        "model_config": {
            "vocab_size": dataset.vocab_size,
            "d_model": model.d_model,
            "num_heads": num_heads,
            "d_ff": model.d_ff,
            "num_layers": model.num_layers,
            "max_seq_len": model.max_seq_len #args.get("seq_len", 64)
        },
        
        # Mapeamentos de tokens (vocabulário)
        "vocab_mappings": {
            **_get_vocab_mappings(dataset)
        },
        
        # Hiperparâmetros do treino
        "training_config": {
            "micro_batch_size": args.get("micro_batch_size", 4),
            "accumulation_steps": args.get("accumulation_steps", 8),
            "learning_rate": args.get("learning_rate", 1e-3),
            "weight_decay": args.get("weight_decay", 1e-5),
            "epochs": args.get("epochs", 1000),
            "effective_batch_size": args.get("micro_batch_size", 4) * args.get("accumulation_steps", 8)
        },
        
        # Informações do dataset
        "dataset_info": {
            "n_samples": len(dataset),
            "seq_len": args.get("seq_len", 64),
            "train_size": args.get("train_size", 0),
            "val_size": args.get("val_size", 0),
            "vocab_size": dataset.vocab_size
        },
        
        # Configurações de dispositivo
        "device_config": {
            "training_device": str(device),
            "inference_device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        
        # Metadados
        "metadata": {
            "best_val_loss": _json_safe(args.get("best_val_loss", None)),
            "timestamp": args.get("timestamp", None),
            "model_version": "1.0"
        }
    }
    config = _json_safe(config)
    
    # Salvar como JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False, allow_nan=False)
    
    print(f"Config saved to: {output_path}")
    return config



def load_inference_config(config_path="config.json"):
    """
    Carrega configuração para inferência
    
    Args:
        config_path: Caminho do arquivo config.json
        
    Returns:
        Dicionário com configuração
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"✅ Configuração carregada de: {config_path}")
    return config
