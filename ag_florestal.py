# -*- coding: utf-8 -*-
"""
Algoritmo Genetico (AG) para o problema de Planejamento Florestal.

Implementacao fiel ao artigo "Utilizacao de Heuristicas para Solucao de um
Problema de Planejamento Florestal" (secao 3.5 - Algoritmo Genetico), usando
a base de dados Base120.xlsx (120 talhoes x 81 prescricoes, horizonte de 16 anos).

Objetivo: escolher UMA prescricao para cada talhao de modo a maximizar o VPL
total, penalizando a producao anual de madeira que ficar fora dos limites
[Dmin, Dmax] (Eq. 6 do artigo).

Autor do artigo: implementacao reescrita em Python (versao simples e editavel).
"""

import numpy as np
import pandas as pd

# =============================================================================
# PARAMETROS EDITAVEIS  -- altere aqui para seus experimentos
# =============================================================================

ARQUIVO_BASE   = "Base120.xlsx"  # caminho da base de dados

# --- Restricoes do problema (Eq. 3, 4 e 6 do artigo) ---
D_MIN     = 140000   # producao anual minima de madeira (m3)
D_MAX     = 160000   # producao anual maxima de madeira (m3)
PENALIDADE = 500     # Pe: R$ por unidade de volume (m3) fora dos limites

# --- Parametros do Algoritmo Genetico (secao 3.5) ---
TAM_POPULACAO  = 40    # numero de individuos na populacao
N_FILHOS       = 40    # filhos gerados por geracao
TAM_TORNEIO    = 4     # individuos sorteados no torneio de selecao
TAXA_MUTACAO   = 0.01  # 1% de chance de mutar um individuo
N_ELITE        = 10    # melhores individuos preservados na substituicao

# --- Criterio de parada ---
MAX_CALCULOS   = 50000  # numero maximo de calculos da funcao objetivo
                        # (no artigo: 5000, 10000, 25000 ou 50000)

# --- Reprodutibilidade ---
SEMENTE        = 42     # semente do gerador aleatorio (None para aleatorio)

# =============================================================================
# CARGA E PREPARACAO DOS DADOS
# =============================================================================

def carregar_dados(arquivo):
    """Le a base e monta as matrizes de VPL e de volumes.

    Retorna:
      VPL : matriz (n_talhoes, n_prescricoes) com o VPL de cada par talhao/prescricao
      VOL : matriz (n_talhoes, n_prescricoes, 16) com o volume colhido em cada ano
    """
    df = pd.read_excel(arquivo)
    df.columns = ["talhao", "idade", "prescricao"] + [f"ano{k}" for k in range(1, 17)] + ["VPL"]

    # Ordena por (talhao, prescricao) para que o reshape agrupe corretamente
    # cada talhao em um bloco contiguo com suas prescricoes em ordem crescente.
    df = df.sort_values(["talhao", "prescricao"]).reset_index(drop=True)

    n_t = df["talhao"].nunique()
    n_p = df["prescricao"].nunique()
    anos = [f"ano{k}" for k in range(1, 17)]

    # A base e regular (todo talhao tem todas as prescricoes), entao um simples
    # reshape substitui o antigo laco linha-a-linha (muito mais rapido).
    if len(df) != n_t * n_p:
        raise ValueError(
            f"Base irregular: {len(df)} linhas != {n_t} talhoes x {n_p} prescricoes."
        )

    VPL = df["VPL"].to_numpy(dtype=float).reshape(n_t, n_p)
    VOL = df[anos].to_numpy(dtype=float).reshape(n_t, n_p, 16)

    return VPL, VOL


# =============================================================================
# FUNCAO OBJETIVO (Eq. 6 do artigo)
# =============================================================================

class FuncaoObjetivo:
    """Avalia um individuo e conta quantas vezes a funcao objetivo foi calculada."""

    def __init__(self, VPL, VOL):
        self.VPL = VPL
        self.VOL = VOL
        self.n_talhoes = VPL.shape[0]
        self.idx = np.arange(self.n_talhoes)
        self.calculos = 0  # contador do criterio de parada

    def avaliar(self, individuo):
        """individuo: vetor de tamanho n_talhoes com o indice da prescricao escolhida."""
        self.calculos += 1

        # Soma dos VPLs das prescricoes escolhidas (Eq. 1)
        vpl_total = self.VPL[self.idx, individuo].sum()

        # Volume anual total (soma sobre os talhoes) para os 16 anos
        volume_por_ano = self.VOL[self.idx, individuo, :].sum(axis=0)

        # Desvio absoluto fora dos limites [D_MIN, D_MAX] em cada ano (dk)
        desvio = (np.maximum(0, D_MIN - volume_por_ano) +
                  np.maximum(0, volume_por_ano - D_MAX))

        # VPL penalizado (Eq. 6)
        return vpl_total - PENALIDADE * desvio.sum()


# =============================================================================
# OPERADORES DO ALGORITMO GENETICO
# =============================================================================

def populacao_inicial(rng, n_talhoes, n_prescricoes):
    """Inicializacao (secao 3.5, passo 1).

    No artigo usa-se a heuristica construtiva com alpha=1 (totalmente aleatoria)
    e cada individuo recebe um embaralhamento. Na pratica, isso equivale a gerar
    TAM_POPULACAO solucoes aleatorias: cada talhao recebe uma prescricao sorteada.
    """
    return [rng.integers(0, n_prescricoes, size=n_talhoes) for _ in range(TAM_POPULACAO)]


def selecao_torneio(rng, fitness, excluir=None):
    """Selecao por torneio (passo 3): sorteia TAM_TORNEIO individuos e
    retorna o indice daquele com maior VPL. 'excluir' remove temporariamente
    um individuo (usado para o 2o pai nao ser igual ao 1o)."""
    candidatos = list(range(len(fitness)))
    if excluir is not None:
        candidatos.remove(excluir)
    sorteados = rng.choice(candidatos, size=TAM_TORNEIO, replace=False)
    return max(sorteados, key=lambda i: fitness[i])


def cruzamento(rng, pai1, pai2):
    """Recombinacao de 1 ponto (passo 4): gera dois filhos p1|p2' e p2|p1'."""
    ponto = rng.integers(1, len(pai1))  # ponto de corte entre 1 e n_talhoes-1
    filho1 = np.concatenate([pai1[:ponto], pai2[ponto:]])
    filho2 = np.concatenate([pai2[:ponto], pai1[ponto:]])
    return filho1, filho2


def mutacao(rng, individuo):
    """Mutacao (passo 5): com prob. TAXA_MUTACAO, troca a prescricao de dois
    talhoes aleatorios entre si."""
    if rng.random() < TAXA_MUTACAO:
        a, b = rng.choice(len(individuo), size=2, replace=False)
        individuo[a], individuo[b] = individuo[b], individuo[a]
    return individuo


# =============================================================================
# LACO PRINCIPAL DO AG
# =============================================================================

def algoritmo_genetico(VPL, VOL, verbose=True, registrar_historico=False):
    """Executa uma rodada do AG.

    Retorna (melhor_individuo, melhor_fitness, n_calculos, historico).
    Quando registrar_historico=True, 'historico' e uma lista de pares
    (n_calculos, melhor_fitness_ate_o_momento) apos cada geracao — usada para
    tracar as curvas de convergencia. Caso contrario, 'historico' e None.
    """
    rng = np.random.default_rng(SEMENTE)
    fobj = FuncaoObjetivo(VPL, VOL)
    n_talhoes, n_prescricoes = VPL.shape

    # 1) Inicializacao + 2) Avaliacao
    populacao = populacao_inicial(rng, n_talhoes, n_prescricoes)
    fitness = [fobj.avaliar(ind) for ind in populacao]

    melhor_idx = int(np.argmax(fitness))
    melhor_individuo = populacao[melhor_idx].copy()
    melhor_fitness = fitness[melhor_idx]

    historico = [(fobj.calculos, melhor_fitness)] if registrar_historico else None

    geracao = 0
    while fobj.calculos < MAX_CALCULOS:
        geracao += 1

        # 3) Selecao + 4) Recombinacao  -> gera N_FILHOS filhos
        filhos = []
        while len(filhos) < N_FILHOS:
            i1 = selecao_torneio(rng, fitness)
            i2 = selecao_torneio(rng, fitness, excluir=i1)
            f1, f2 = cruzamento(rng, populacao[i1], populacao[i2])
            filhos.append(f1)
            if len(filhos) < N_FILHOS:
                filhos.append(f2)

        # 5) Mutacao
        filhos = [mutacao(rng, f) for f in filhos]

        # 2) Avaliacao dos filhos
        fitness_filhos = [fobj.avaliar(f) for f in filhos]

        # 6) Substituicao: une populacao + filhos, mantem os N_ELITE melhores
        #    e preenche o restante por torneio sobre o conjunto unido.
        pool        = populacao + filhos
        pool_fit    = fitness + fitness_filhos
        ordem       = np.argsort(pool_fit)[::-1]      # do melhor para o pior

        nova_pop, nova_fit = [], []
        for k in ordem[:N_ELITE]:                     # elitismo: 10 melhores
            nova_pop.append(pool[k].copy())
            nova_fit.append(pool_fit[k])

        while len(nova_pop) < TAM_POPULACAO:           # restante por torneio
            vencedor = selecao_torneio(rng, pool_fit)
            nova_pop.append(pool[vencedor].copy())
            nova_fit.append(pool_fit[vencedor])

        populacao, fitness = nova_pop, nova_fit

        # Atualiza o melhor global encontrado
        idx = int(np.argmax(fitness))
        if fitness[idx] > melhor_fitness:
            melhor_fitness = fitness[idx]
            melhor_individuo = populacao[idx].copy()

        if registrar_historico:
            historico.append((fobj.calculos, melhor_fitness))

        if verbose and geracao % 20 == 0:
            print(f"Geracao {geracao:4d} | calculos {fobj.calculos:6d} | "
                  f"melhor VPL = R$ {melhor_fitness:,.2f}")

    return melhor_individuo, melhor_fitness, fobj.calculos, historico


# =============================================================================
# EXECUCAO
# =============================================================================

if __name__ == "__main__":
    print("Carregando base de dados...")
    VPL, VOL = carregar_dados(ARQUIVO_BASE)
    print(f"Base carregada: {VPL.shape[0]} talhoes x {VPL.shape[1]} prescricoes\n")

    print(f"Executando AG (max {MAX_CALCULOS} calculos da funcao objetivo)...\n")
    melhor, valor, n_calc, _ = algoritmo_genetico(VPL, VOL)

    # VPL de referencia do artigo para calculo de eficiencia (JUNIOR et al., 2021)
    VPL_REFERENCIA = 32170883
    eficiencia = valor / VPL_REFERENCIA

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print(f"Calculos da funcao objetivo : {n_calc}")
    print(f"Melhor VPL (penalizado)     : R$ {valor:,.2f}")
    print(f"Eficiencia (VPL/VPL_ref)    : {eficiencia:.3f}  ({eficiencia*100:.1f}%)")
    print(f"Prescricoes escolhidas (talhao 1..120, valores 1..81):")
    print((melhor + 1).tolist())  # +1 para exibir no padrao do artigo (1..81)
