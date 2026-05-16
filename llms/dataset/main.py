import random

def gerar_dataset_txt(num_funcoes=800, nome_arquivo="dataset_python.txt"):
    """
    Gera um arquivo .txt grande com muitas funções e classes em Python.
    """
    conteudo = "# Dataset grande para treinamento de modelo de código Python\n\n"
    
    # Funções matemáticas básicas
    funcoes_base = [
        "def soma(a, b):\n    return a + b\n",
        "def subtracao(a, b):\n    return a - b\n",
        "def multiplicar(x, y):\n    return x * y\n",
        "def divisao(a, b):\n    if b == 0:\n        return 'Erro: divisao por zero'\n    return a / b\n",
        "def potencia(base, expoente):\n    return base ** expoente\n",
        "def fatorial(n):\n    if n < 0:\n        return None\n    resultado = 1\n    for i in range(1, n + 1):\n        resultado *= i\n    return resultado\n",
        "def fibonacci(n):\n    if n <= 0:\n        return []\n    seq = [0, 1]\n    for _ in range(2, n):\n        seq.append(seq[-1] + seq[-2])\n    return seq\n",
        "def eh_par(numero):\n    return numero % 2 == 0\n",
        "def eh_impar(numero):\n    return numero % 2 != 0\n",
        "def eh_primo(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
    ]
    
    # Funções de string
    funcoes_string = [
        "def inverter_string(texto):\n    return texto[::-1]\n",
        "def contar_vogais(texto):\n    vogais = 'aeiouAEIOU'\n    return sum(1 for letra in texto if letra in vogais)\n",
        "def eh_palindromo(palavra):\n    p = palavra.lower()\n    return p == p[::-1]\n",
        "def maiusculas(texto):\n    return texto.upper()\n",
        "def minusculas(texto):\n    return texto.lower()\n",
    ]
    
    # Funções de lista
    funcoes_lista = [
        "def soma_lista(numeros):\n    return sum(numeros)\n",
        "def media_lista(numeros):\n    return sum(numeros) / len(numeros) if numeros else 0\n",
        "def maior_numero(lista):\n    return max(lista) if lista else None\n",
        "def menor_numero(lista):\n    return min(lista) if lista else None\n",
        "def filtrar_positivos(lista):\n    return [x for x in lista if x > 0]\n",
    ]
    
    # Classes
    classes = [
        """class Pessoa:
    def __init__(self, nome, idade):
        self.nome = nome
        self.idade = idade
    
    def apresentar(self):
        return f"Olá, meu nome é {self.nome} e tenho {self.idade} anos."
    
    def aniversariar(self):
        self.idade += 1
        return f"Parabéns! Agora você tem {self.idade} anos."
""",
        """class Carro:
    def __init__(self, marca, modelo, ano):
        self.marca = marca
        self.modelo = modelo
        self.ano = ano
        self.velocidade = 0
    
    def acelerar(self, valor):
        self.velocidade += valor
        return f"Velocidade atual: {self.velocidade} km/h"
    
    def frear(self, valor):
        self.velocidade = max(0, self.velocidade - valor)
        return f"Velocidade atual: {self.velocidade} km/h"
""",
        """class ContaBancaria:
    def __init__(self, titular, saldo=0):
        self.titular = titular
        self.saldo = saldo
        self.historico = []
    
    def depositar(self, valor):
        if valor > 0:
            self.saldo += valor
            self.historico.append(f"+{valor}")
            return f"Depósito de R${valor} realizado."
        return "Valor inválido"
    
    def sacar(self, valor):
        if 0 < valor <= self.saldo:
            self.saldo -= valor
            self.historico.append(f"-{valor}")
            return f"Saque de R${valor} realizado."
        return "Saldo insuficiente"
""",
    ]
    
    # Gerar o arquivo com muitas variações
    for i in range(num_funcoes):
        if i % 8 == 0 and classes:
            conteudo += random.choice(classes) + "\n"
        elif i % 5 == 0:
            conteudo += random.choice(funcoes_string) + "\n"
        elif i % 4 == 0:
            conteudo += random.choice(funcoes_lista) + "\n"
        else:
            base = random.choice(funcoes_base)
            # Pequenas variações no nome
            if "soma" in base and random.random() > 0.7:
                base = base.replace("soma", f"somar_{random.randint(1,99)}")
            conteudo += base + "\n"
        
        # Adiciona uma função extra de vez em quando
        if random.random() > 0.85:
            conteudo += f"""def funcao_extra_{i}(x, y):
    resultado = x * y + {random.randint(1,50)}
    return resultado
"""
    
    # Salvar no arquivo
    with open(nome_arquivo, "w", encoding="utf-8") as f:
        f.write(conteudo)
    
    print(f"✅ Arquivo gerado com sucesso!")
    print(f"   Nome: {nome_arquivo}")
    print(f"   Total aproximado de funções + classes: {num_funcoes}")

# ===================== EXECUTAR =====================
if __name__ == "__main__":
    gerar_dataset_txt(num_funcoes=3000)   # Mude o número se quiser mais ou menos