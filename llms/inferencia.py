import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model import TransformerModel


class SavedBPETokenizer:
    """Tokenizer BPE reconstruido a partir do config.json salvo no treino."""

    def __init__(self, vocab_mappings):
        self.word_to_idx = vocab_mappings["word_to_idx"]
        self.idx_to_word = {
            int(idx): token
            for idx, token in vocab_mappings["idx_to_word"].items()
        }
        self.merges = [tuple(pair) for pair in vocab_mappings.get("merges", [])]
        self.special_tokens = set(
            vocab_mappings.get("special_tokens", ["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
        )
        self.unk_id = self.word_to_idx.get("<UNK>", 1)

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
        return text.replace("Ġ", " ").replace("</w>", "").strip()

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


class LLMInference:
    """Carrega config + pesos e gera texto sem precisar recriar o dataset."""

    def __init__(self, config_path, weights_path, device=None):
        self.config_path = Path(config_path)
        self.weights_path = Path(weights_path)
        self.config = self._load_config(self.config_path)
        self.device = self._resolve_device(device)
        self.tokenizer = SavedBPETokenizer(self.config["vocab_mappings"])
        self.seq_len = self.config["model_config"]["max_seq_len"]
        self.state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model = self._load_model()

    @staticmethod
    def _load_config(config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _resolve_device(device):
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            print("CUDA indisponivel. Usando CPU.")
            device = "cpu"

        return torch.device(device)

    def _load_model(self):
        model_config = self.config["model_config"]
        model_max_seq_len = self._get_model_max_seq_len(model_config["max_seq_len"])
        model = TransformerModel(
            vocab_size=model_config["vocab_size"],
            d_model=model_config["d_model"],
            num_heads=model_config["num_heads"],
            d_ff=model_config["d_ff"],
            num_layers=model_config["num_layers"],
            max_seq_len=model_max_seq_len,
        )

        model.load_state_dict(self.state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _get_model_max_seq_len(self, fallback):
        positional_encoding = self.state_dict.get("pos_encoding.pe")
        if positional_encoding is None:
            return fallback

        return positional_encoding.shape[1]

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=None,
    ):
        if temperature <= 0:
            raise ValueError("temperature precisa ser maior que zero.")

        token_ids = self.tokenizer.encode(prompt)
        bos_id = self.tokenizer.word_to_idx.get("<BOS>")

        if bos_id is not None:
            token_ids = [bos_id] + token_ids

        if not token_ids:
            token_ids = [self.tokenizer.unk_id]

        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            input_tokens = tokens[:, -self.seq_len:]
            mask = self.model.generate_causal_mask(input_tokens.shape[1]).to(self.device)
            logits = self.model(input_tokens, mask)
            next_token_logits = logits[:, -1, :] / temperature
            next_token_logits = self._apply_top_k(next_token_logits, top_k)
            next_token_logits = self._apply_top_p(next_token_logits, top_p)

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = int(next_token.item())
            next_token_text = self.tokenizer.idx_to_word.get(next_token_id, "<UNK>")

            if next_token_text in {"<EOS>", "<PAD>"}:
                break

            tokens = torch.cat([tokens, next_token], dim=1)

        generated_ids = tokens[0].tolist()
        if bos_id is not None and generated_ids and generated_ids[0] == bos_id:
            generated_ids = generated_ids[1:]

        return self.tokenizer.decode(generated_ids)

    @staticmethod
    def _apply_top_k(logits, top_k):
        if top_k is None or top_k <= 0:
            return logits

        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        return logits.masked_fill(logits < threshold, -float("inf"))

    @staticmethod
    def _apply_top_p(logits, top_p):
        if top_p is None or top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove_mask = cumulative_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False

        filtered_logits = logits.clone()
        filtered_logits.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_logits.masked_fill(remove_mask, -float("inf")),
        )
        return filtered_logits


def generate_from_files(
    config_path,
    weights_path,
    prompt,
    device=None,
    max_new_tokens=50,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    inference = LLMInference(
        config_path=config_path,
        weights_path=weights_path,
        device=device,
    )
    return inference.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
