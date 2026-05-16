from inferencia import LLMInference


def chat_loop(
    config_path="config.json",
    weights_path="model_weights.pth",
    device="auto",
    max_new_tokens=80,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
):
    chat = LLMInference(
        config_path=config_path,
        weights_path=weights_path,
        device=device,
    )

    print("Chat iniciado. Digite 'sair', 'exit' ou 'q' para encerrar.")
    print(f"Device: {chat.device}")

    while True:
        prompt = input("\nVoce: ").strip()

        if prompt.lower() in {"sair", "exit", "q", "quit"}:
            print("Encerrado.")
            break

        if not prompt:
            continue

        resposta = chat.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        print(f"Modelo: {resposta}")


if __name__ == "__main__":
    chat_loop()
