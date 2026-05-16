from collections import Counter

# ----------- Char level ---------
# class TextDataset(Dataset):
#     def __init__(self, text, seq_len=64):
#         self.seq_len = seq_len
        
#         # Cria vocabulário de caracteres
#         chars = sorted(list(set(text)))
#         self.char_to_idx = {ch: i+1 for i, ch in enumerate(chars)}
#         self.char_to_idx['<PAD>'] = 0
        
#         self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
#         self.vocab_size = len(self.char_to_idx)
        
#         # Converte para índices
#         self.tokens = [self.char_to_idx.get(ch, 0) for ch in text]
        
#     def __len__(self):
#         return max(0, len(self.tokens) - self.seq_len)
    
#     def __getitem__(self, idx):
#         X = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
#         y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
#         return X, y


# -------------- Versão alternativa com tokenizador word-level:
 
# class TextDataset(Dataset):
#     def __init__(self, text, seq_len=64, min_freq=1):
#         self.seq_len = seq_len
#         self.min_freq = min_freq
#         self.build_vocab(text)
        
#     def build_vocab(self, text):
#         # Tokenização
#         words = text.lower().split()
        
#         # Conta frequência das palavras
#         word_counts = Counter(words)
        
#         # Filtra palavras com baixa frequência (opcional)
#         # Adiciona tokens especiais
#         self.word_to_idx = {
#             '<PAD>': 0,  # padding
#             '<UNK>': 1,  # unknown
#             '<BOS>': 2,  # begin of sequence
#             '<EOS>': 3,  # end of sequence
#         }
        
#         # Adiciona palavras do vocabulário
#         idx = 4
#         for word, count in word_counts.items():
#             if count >= self.min_freq:
#                 self.word_to_idx[word] = idx
#                 idx += 1
                
#         self.vocab_size = len(self.word_to_idx)
#         self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        
#         # Converte texto para tokens
#         self.tokens = []
#         for word in words:
#             token = self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
#             self.tokens.append(token)
            
#         print(f"Vocabulário: {self.vocab_size} tokens")
#         print(f"Total de tokens no texto: {len(self.tokens)}")
        
#     def __len__(self):
#         # Número de sequências possíveis
#         return max(1, len(self.tokens) - self.seq_len)
    
#     def __getitem__(self, idx):
#         # Garante que não ultrapassa os limites
#         end_idx = min(idx + self.seq_len, len(self.tokens) - 1)
#         start_idx = end_idx - self.seq_len
        
#         X = torch.tensor(self.tokens[start_idx:end_idx], dtype=torch.long)
#         y = torch.tensor(self.tokens[start_idx + 1:end_idx + 1], dtype=torch.long)
        
#         # Se a sequência for menor que seq_len, faz padding
#         if len(X) < self.seq_len:
#             pad_size = self.seq_len - len(X)
#             X = torch.cat([X, torch.zeros(pad_size, dtype=torch.long)])
#             y = torch.cat([y, torch.zeros(pad_size, dtype=torch.long)])
            
#         return X, y

# ---- BPE ------------

from collections import Counter, defaultdict
import re
import torch
from torch.utils.data import Dataset

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer treinado diretamente no texto fornecido.
    Compatível com o TextDataset abaixo.
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self, vocab_size: int = 512, min_freq: int = 2):
        """
        vocab_size : tamanho máximo do vocabulário (especiais + chars + merges)
        min_freq   : par precisa aparecer >= min_freq para ser fundido
        """
        self.target_vocab_size = vocab_size
        self.min_freq = min_freq

        self.word_to_idx: dict[str, int] = {}
        self.idx_to_word: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []   # ordem dos merges aprendidos
        self.vocab_size = 0

    # ------------------------------------------------------------------ #
    #  Treinamento                                                         #
    # ------------------------------------------------------------------ #

    def train(self, text: str) -> None:
        """Aprende o vocabulário BPE a partir de `text`."""
        # 1. Pré-tokeniza em palavras e representa cada uma como lista de chars
        #    O marcador "Ġ" (U+0120) indica início de palavra — estilo GPT-2
        words_raw = text.lower().split()
        word_freq: Counter = Counter(words_raw)

        # Representação inicial: cada palavra = tupla de chars + </w>
        vocab: dict[tuple, int] = {
            tuple(self._word_to_chars(w)): freq
            for w, freq in word_freq.items()
        }

        # 2. Vocabulário base = chars únicos
        base_chars: set[str] = set()
        for token_seq in vocab:
            base_chars.update(token_seq)

        # 3. Inicializa word_to_idx com especiais + chars base
        self.word_to_idx = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        for ch in sorted(base_chars):
            if ch not in self.word_to_idx:
                self.word_to_idx[ch] = len(self.word_to_idx)

        # 4. Loop BPE
        self.merges = []
        while len(self.word_to_idx) < self.target_vocab_size:
            pairs = self._get_pair_freqs(vocab)
            if not pairs:
                break

            best_pair, best_freq = max(pairs.items(), key=lambda x: x[1])
            if best_freq < self.min_freq:
                break

            # Funde o par e adiciona ao vocabulário
            new_token = "".join(best_pair)
            self.merges.append(best_pair)
            self.word_to_idx[new_token] = len(self.word_to_idx)

            # Aplica o merge em todo o vocab
            vocab = self._apply_merge(vocab, best_pair, new_token)

        self.vocab_size = len(self.word_to_idx)
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        print(f"BPE treinado | vocab: {self.vocab_size} tokens "
              f"| merges: {len(self.merges)}")

    # ------------------------------------------------------------------ #
    #  Encode / Decode                                                     #
    # ------------------------------------------------------------------ #

    def encode(self, text: str) -> list[int]:
        """Texto → lista de ids."""
        ids = []
        for word in text.lower().split():
            subwords = self._tokenize_word(word)
            for sw in subwords:
                ids.append(self.word_to_idx.get(sw, self.word_to_idx["<UNK>"]))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Lista de ids → texto (remove marcador Ġ)."""
        tokens = [self.idx_to_word.get(i, "<UNK>") for i in ids]
        # Junta e trata o marcador de início de palavra
        text = " ".join(t for t in tokens
                        if t not in self.SPECIAL_TOKENS)
        return text.replace("Ġ", " ").replace("</w>", "").strip()

    # ------------------------------------------------------------------ #
    #  Helpers internos                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _word_to_chars(word: str) -> list[str]:
        """'hello' → ['h','e','l','l','o</w>']"""
        if not word:
            return []
        chars = list(word)
        chars[-1] = chars[-1] + "</w>"   # marca fim de palavra
        return chars

    @staticmethod
    def _get_pair_freqs(vocab: dict) -> Counter:
        pairs: Counter = Counter()
        for token_seq, freq in vocab.items():
            for a, b in zip(token_seq, token_seq[1:]):
                pairs[(a, b)] += freq
        return pairs

    @staticmethod
    def _apply_merge(vocab: dict, pair: tuple[str, str],
                     new_token: str) -> dict:
        new_vocab: dict = {}
        a, b = pair
        for token_seq, freq in vocab.items():
            new_seq: list[str] = []
            i = 0
            while i < len(token_seq):
                if (i < len(token_seq) - 1
                        and token_seq[i] == a
                        and token_seq[i + 1] == b):
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(token_seq[i])
                    i += 1
            new_vocab[tuple(new_seq)] = freq
        return new_vocab

    def _tokenize_word(self, word: str) -> list[str]:
        """Aplica os merges aprendidos a uma única palavra."""
        chars = self._word_to_chars(word)
        if not chars:
            return []
        # Aplica merges na ordem em que foram aprendidos
        for a, b in self.merges:
            new_chars: list[str] = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                    new_chars.append(a + b)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        return chars


# ────────────────────────────────────────────────────────────────────────────
#  Dataset compatível com o restante do pipeline
# ────────────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 64,
                 bpe_vocab_size: int = 512, min_freq: int = 2):
        self.seq_len = seq_len

        # Treina o tokenizador BPE no texto fornecido
        self.tokenizer = BPETokenizer(vocab_size=bpe_vocab_size,
                                      min_freq=min_freq)
        self.tokenizer.train(text)

        # Expõe vocab_size para o modelo
        self.vocab_size = self.tokenizer.vocab_size

        # Tokeniza o texto completo
        self.tokens: list[int] = self.tokenizer.encode(text)
        print(f"Total de tokens no texto: {len(self.tokens)}")

    # Mantém a mesma interface de antes ──────────────────────────────────
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __len__(self) -> int:
        return max(1, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int):
        end_idx   = min(idx + self.seq_len, len(self.tokens) - 1)
        start_idx = end_idx - self.seq_len

        X = torch.tensor(self.tokens[start_idx:end_idx],     dtype=torch.long)
        y = torch.tensor(self.tokens[start_idx + 1:end_idx + 1], dtype=torch.long)

        if len(X) < self.seq_len:
            pad = self.seq_len - len(X)
            X = torch.cat([X, torch.zeros(pad, dtype=torch.long)])
            y = torch.cat([y, torch.zeros(pad, dtype=torch.long)])

        return X, y

    def itos(self, idx: int) -> str:
        """Retorna a sequência de tokens como string (para debug)."""
        token_ids = self.tokens[idx:idx + self.seq_len]
        return self.decode(token_ids)

    def stoi(self, text: str) -> list[int]:
        """Converte texto para lista de ids (para debug)."""
        return self.encode(text)

    def itos_vocab(self, idx: int) -> str:
        """Retorna o token correspondente a um id do vocabulário."""
        return self.tokenizer.idx_to_word.get(idx, "<UNK>")    
    
print("Rodou com sucesso")