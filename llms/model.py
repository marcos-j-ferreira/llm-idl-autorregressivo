import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Posicional Encoding seno/cosseno do Transformer original"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Criar matriz de positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Termos de frequência
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        # Seno para posições pares, cosseno para ímpares
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention implementada corretamente"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projeções
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # 1. Projetar
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Reshape para heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Calcular atenção
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4. Aplicar aos valores
        output = torch.matmul(attn_weights, V)

        # 5. Juntar heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 6. Projeção final
        output = self.out_proj(output)

        return output, attn_weights


class FeedForward(nn.Module):
    """Feed-Forward Network com GELU"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU é melhor que ReLU para transformers

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Bloco Transformer completo (Pre-Norm)"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Subcamada 1: Atenção
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Subcamada 2: FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-Norm (estilo moderno)
        # 1. Atenção com residual
        attn_output, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # 2. FFN com residual
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)

        return x


class TransformerModel(nn.Module):
    """Modelo Transformer completo para geração de texto"""
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_seq_len=512, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # 1. Embedding de tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding (ISSO ESTAVA FALTANDO!)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # 3. Blocos Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 4. Output Head
        self.output_head = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Inicialização dos pesos (importante!)
        self._init_weights()

    def _init_weights(self):
        """Inicialização Xavier/Glorot para melhor convergência"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        """
        x: tokens (batch, seq_len)
        mask: máscara causal para geração (batch, seq_len, seq_len)
        """
        # 1. Token embedding + scaling
        x = self.token_embedding(x) * math.sqrt(self.d_model)

        # 2. Adicionar positional encoding
        x = self.pos_encoding(x)

        # 3. Passar pelos blocos Transformer
        for block in self.transformer_blocks:
            x = block(x, mask)

        # 4. Projetar para vocabulário
        logits = self.output_head(x)

        return logits

    def generate_causal_mask(self, size):
        """Criar máscara causal (triangular inferior) para geração"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        mask = ~mask  # Invert for 1 = can see, 0 = cannot see
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Gerar texto token por token
        input_ids: (batch, seq_len) - tokens iniciais
        """
        self.eval()
        batch_size = input_ids.shape[0]

        for _ in range(max_new_tokens):
            # Criar máscara causal para os tokens atuais
            seq_len = input_ids.shape[1]
            mask = self.generate_causal_mask(seq_len).to(input_ids.device)

            # Forward pass
            logits = self(input_ids, mask)

            # Pegar logits do último token
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k sampling (se especificado)
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample próximo token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Adicionar à sequência
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

