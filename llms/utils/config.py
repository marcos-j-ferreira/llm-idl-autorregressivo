import json
import torch
from pathlib import Path

def save_training_config(model, dataset, args, output_path="config.json"):
    """
    Salva configurações do treino em um arquivo JSON para inferência posterior
    
    Args:
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
            "num_heads": model.num_heads,
            "d_ff": model.d_ff,
            "num_layers": model.num_layers,
            "max_seq_len": args.get("seq_len", 64)
        },
        
        # Mapeamentos de tokens (vocabulário)
        "vocab_mappings": {
            "itos": dataset.itos,  # índice → token
            "stoi": dataset.stoi   # token → índice
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
            "training_device": str(args.get("device", torch.device("cpu"))),
            "inference_device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        
        # Metadados
        "metadata": {
            "best_val_loss": args.get("best_val_loss", None),
            "timestamp": args.get("timestamp", None),
            "model_version": "1.0"
        }
    }
    
    # Salvar como JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Configuração salva em: {output_path}")
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


# ==============================================
# EXEMPLO DE USO INTEGRADO AO SEU CÓDIGO DE TREINO
# ==============================================

def train_and_save_config():
    """Função completa integrando treino e salvamento de configuração"""
    import time
    from torch.utils.data import DataLoader, random_split
    
    # Seu código de treino aqui...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    
    # Assumindo que você já tem seus dados e dataset
    # text = seu_texto_aqui
    # dataset = TextDataset(text, seq_len=64)
    
    # Exemplo de configuração (substitua pelos seus valores reais)
    class DummyDataset:
        def __init__(self):
            self.vocab_size = 1000
            self.itos = {i: f"token_{i}" for i in range(1000)}
            self.stoi = {f"token_{i}": i for i in range(1000)}
            
        def __len__(self):
            return 5000
    
    dataset = DummyDataset()
    n_samples = len(dataset)
    vocab_size = dataset.vocab_size
    
    # Regras simples para tamanho do modelo
    if n_samples < 500:
        d_model, num_heads, d_ff, num_layers = 64, 2, 256, 2
    elif n_samples < 5_000:
        d_model, num_heads, d_ff, num_layers = 128, 4, 512, 4
    else:
        d_model, num_heads, d_ff, num_layers = 256, 8, 1024, 6
    
    print(f"Dataset: {n_samples} amostras → d_model={d_model}, layers={num_layers}")
    
    # Split
    train_size = int(0.8 * n_samples)
    val_size = n_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Configurações
    micro_batch_size = 4
    accumulation_steps = 8
    
    # Dicionário com argumentos para salvar
    training_args = {
        "seq_len": 64,
        "micro_batch_size": micro_batch_size,
        "accumulation_steps": accumulation_steps,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "epochs": 1000,
        "device": device,
        "train_size": train_size,
        "val_size": val_size,
        "best_val_loss": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Criar modelo (substitua pela sua classe TransformerModel)
    class DummyModel:
        def __init__(self):
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_ff = d_ff
            self.num_layers = num_layers
    
    model = DummyModel()
    
    # 🔥 SALVAR CONFIGURAÇÃO PARA INFERÊNCIA 🔥
    config = save_training_config(
        model=model,
        dataset=dataset,
        args=training_args,
        output_path="model_config.json"
    )
    
    # Salvar também os pesos do modelo
    # torch.save(model.state_dict(), "model_weights.pth")
    
    return config


# ==============================================
# SCRIPT DE INFERÊNCIA (USANDO O CONFIG.JSON)
# ==============================================

class InferenceModel:
    """Classe para carregar modelo a partir do config.json"""
    
    def __init__(self, config_path="model_config.json", weights_path=None):
        # Carregar configuração
        self.config = load_inference_config(config_path)
        
        # Extrair configurações do modelo
        model_config = self.config["model_config"]
        self.vocab_size = model_config["vocab_size"]
        self.d_model = model_config["d_model"]
        self.num_heads = model_config["num_heads"]
        self.d_ff = model_config["d_ff"]
        self.num_layers = model_config["num_layers"]
        self.max_seq_len = model_config["max_seq_len"]
        
        # Mapeamentos de vocabulário
        self.itos = self.config["vocab_mappings"]["itos"]
        self.stoi = self.config["vocab_mappings"]["stoi"]
        
        # Configurar dispositivo
        device_config = self.config["device_config"]
        self.device = torch.device(device_config["inference_device"])
        
        # Inicializar modelo (substitua pela sua classe TransformerModel)
        # self.model = TransformerModel(...)
        # self.model.load_state_dict(torch.load(weights_path))
        # self.model.to(self.device)
        # self.model.eval()
        
        print(f"✅ Modelo carregado para inferência em: {self.device}")
        print(f"📊 Vocabulário: {self.vocab_size} tokens")
        print(f"🔧 Arquitetura: d_model={self.d_model}, layers={self.num_layers}")
    
    def generate(self, prompt, max_length=100, temperature=1.0):
        """
        Gera texto a partir de um prompt
        
        Args:
            prompt: String de entrada
            max_length: Número máximo de tokens a gerar
            temperature: Temperatura para sampling (1.0 = normal, <1.0 = determinístico)
        """
        # Implemente sua lógica de geração aqui
        # Usando as configurações carregadas
        print(f"🎨 Gerando texto a partir de: '{prompt}'")
        print(f"⚙️ Configurações: max_length={max_length}, temp={temperature}")
        
        # Exemplo de placeholder
        return f"{prompt} [texto gerado pelo modelo com {self.d_model} dimensões]"


# ==============================================
# USO PRÁTICO
# ==============================================

if __name__ == "__main__":
    # 1. Treinar e salvar configuração
    print("="*50)
    print("FASE 1: TREINAMENTO")
    print("="*50)
    config = train_and_save_config()
    
    # 2. Carregar para inferência
    print("\n" + "="*50)
    print("FASE 2: INFERÊNCIA")
    print("="*50)
    inference_model = InferenceModel(
        config_path="model_config.json",
        weights_path="model_weights.pth"  # Opcional
    )
    
    # 3. Gerar texto
    generated = inference_model.generate(
        prompt="Era uma vez",
        max_length=200,
        temperature=0.8
    )
    
    print(f"\n📝 Resultado:\n{generated}")