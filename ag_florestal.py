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

# --- Estrategia de inicializacao da populacao (foco da comparacao) ---
#   "aleatoria" : solucoes completamente aleatorias  (heuristica construtiva alpha=1)
#   "grasp"     : construtiva gulosa-aleatoria tipo GRASP com o ALPHA_GRASP abaixo
ESTRATEGIA_INICIAL = "aleatoria"
ALPHA_GRASP        = 0.5   # alpha da Eq. 7 do artigo (0=guloso, 1=aleatorio)

# --- Hibridizacao com Busca Local: AG Memetico (P1) ---
#   Quando MEMETICO=True, a cada geracao aplica-se uma busca local de
#   vizinhanca sistematica (primeira-melhora) ao(s) melhor(es) individuo(s),
#   gastando ate ORCAMENTO_BL avaliacoes. O AG diversifica (exploracao global)
#   e a BL intensifica (exploracao local / refinamento).
MEMETICO      = False
ORCAMENTO_BL  = 120   # avaliacoes maximas gastas na BL por geracao
N_ELITE_BL    = 1     # quantos dos melhores individuos recebem a BL (1..N_ELITE)

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

    talhoes     = sorted(df["talhao"].unique())
    prescricoes = sorted(df["prescricao"].unique())
    n_t, n_p    = len(talhoes), len(prescricoes)

    idx_t = {t: i for i, t in enumerate(talhoes)}
    idx_p = {p: j for j, p in enumerate(prescricoes)}

    VPL = np.zeros((n_t, n_p))
    VOL = np.zeros((n_t, n_p, 16))
    anos = [f"ano{k}" for k in range(1, 17)]

    for _, linha in df.iterrows():
        i = idx_t[linha["talhao"]]
        j = idx_p[linha["prescricao"]]
        VPL[i, j] = linha["VPL"]
        VOL[i, j, :] = linha[anos].values

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

def populacao_inicial_aleatoria(rng, n_talhoes, n_prescricoes):
    """Inicializacao ALEATORIA (secao 3.5, passo 1 com alpha=1).

    No artigo usa-se a heuristica construtiva com alpha=1 (totalmente aleatoria)
    e cada individuo recebe um embaralhamento. Na pratica, isso equivale a gerar
    TAM_POPULACAO solucoes aleatorias: cada talhao recebe uma prescricao sorteada.
    """
    return [rng.integers(0, n_prescricoes, size=n_talhoes) for _ in range(TAM_POPULACAO)]


def _listas_candidatas(VPL, alpha):
    """Monta a Lista Restrita de Candidatos (RCL) de cada talhao (Eq. 7 do artigo).

    Para o talhao i, calcula o limiar
        mu_i = VPLmax_i - alpha * (VPLmax_i - VPLmin_i)
    e mantem como candidatas apenas as prescricoes cujo VPL >= mu_i.

      alpha = 0   -> so a(s) prescricao(oes) de maior VPL (estritamente guloso)
      alpha = 1   -> todas as prescricoes (equivale a totalmente aleatorio)
      alpha = 0.5 -> metade superior da faixa de VPL de cada talhao (GRASP)

    Retorna uma lista de arrays de indices de prescricao (uma por talhao).
    """
    vpl_max = VPL.max(axis=1)
    vpl_min = VPL.min(axis=1)
    mu = vpl_max - alpha * (vpl_max - vpl_min)
    return [np.where(VPL[i] >= mu[i])[0] for i in range(VPL.shape[0])]


def populacao_inicial_grasp(rng, VPL, alpha):
    """Inicializacao GULOSA-ALEATORIA tipo GRASP (heuristica construtiva, secao 3.1).

    Cada individuo e construido de forma independente: para cada talhao sorteia-se
    uniformemente uma prescricao dentro da sua RCL (prescricoes de VPL mais alto).
    Isso enviesa a populacao inicial para regioes mais promissoras do espaco de
    busca, mantendo diversidade entre os TAM_POPULACAO individuos.
    """
    rcls = _listas_candidatas(VPL, alpha)
    n_talhoes = VPL.shape[0]
    populacao = []
    for _ in range(TAM_POPULACAO):
        individuo = np.array([rng.choice(rcls[i]) for i in range(n_talhoes)])
        populacao.append(individuo)
    return populacao


def gerar_populacao_inicial(rng, VPL, estrategia, alpha):
    """Despacha para a estrategia de inicializacao escolhida."""
    n_talhoes, n_prescricoes = VPL.shape
    if estrategia == "grasp":
        return populacao_inicial_grasp(rng, VPL, alpha)
    elif estrategia == "aleatoria":
        return populacao_inicial_aleatoria(rng, n_talhoes, n_prescricoes)
    raise ValueError(f"Estrategia desconhecida: {estrategia!r} (use 'aleatoria' ou 'grasp')")


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
# BUSCA LOCAL (componente memetico - vizinhanca sistematica, primeira-melhora)
# =============================================================================

def busca_local_sistematica(rng, fobj, individuo, fitness_ind, orcamento, n_prescricoes):
    """Refina um individuo por Busca Local de vizinhanca sistematica (secao 3.2
    do artigo) com o metodo de primeira-melhora, no estilo Lamarckiano.

    Vizinhanca sistematica: cada vizinho e uma copia da solucao atual com apenas
    UM talhao alterado para outra prescricao, mantendo os demais inalterados.
    Primeira-melhora: percorre os talhoes em ordem; ao encontrar o primeiro
    vizinho melhor que a solucao atual, aceita a troca e recomeca a varredura a
    partir da solucao melhorada. Encerra ao esgotar 'orcamento' avaliacoes da
    funcao objetivo ou ao completar uma varredura inteira sem melhora.

    Cada avaliacao conta no orcamento GLOBAL do AG (fobj.calculos), garantindo
    comparacao justa pelo mesmo criterio de parada. Retorna (individuo, fitness).
    """
    atual = individuo.copy()
    f_atual = fitness_ind
    n_talhoes = len(atual)
    usados = 0
    melhorou = True

    while melhorou and usados < orcamento:
        melhorou = False
        for t in range(n_talhoes):                 # varredura sistematica
            if usados >= orcamento:
                break
            # gera um vizinho trocando a prescricao do talhao t por outra
            nova = int(rng.integers(0, n_prescricoes))
            if nova == atual[t]:
                nova = (nova + 1) % n_prescricoes
            vizinho = atual.copy()
            vizinho[t] = nova

            f_viz = fobj.avaliar(vizinho)
            usados += 1
            if f_viz > f_atual:                    # primeira-melhora: aceita e reinicia
                atual, f_atual = vizinho, f_viz
                melhorou = True
                break

    return atual, f_atual


# =============================================================================
# LACO PRINCIPAL DO AG
# =============================================================================

class Resultado:
    """Saida de uma execucao do AG, com dados para analise e graficos."""

    def __init__(self, melhor_individuo, melhor_fitness, n_calculos,
                 fitness_inicial, historico, estrategia, alpha):
        self.melhor_individuo = melhor_individuo   # vetor de prescricoes (0..80)
        self.melhor_fitness   = melhor_fitness     # VPL penalizado do melhor
        self.n_calculos       = n_calculos         # calculos da func. objetivo
        self.fitness_inicial  = fitness_inicial    # fitness da populacao inicial
        self.historico        = historico          # lista de (n_calculos, melhor)
        self.estrategia       = estrategia
        self.alpha            = alpha

    # compatibilidade com codigo antigo que desempacotava a tupla:
    def __iter__(self):
        return iter((self.melhor_individuo, self.melhor_fitness, self.n_calculos))


def algoritmo_genetico(VPL, VOL, verbose=True, estrategia=None, alpha=None,
                       memetico=None, orcamento_bl=None, n_elite_bl=None):
    """Executa uma rodada completa do AG.

    estrategia   : "aleatoria" ou "grasp" (default: ESTRATEGIA_INICIAL global).
    alpha        : alpha do GRASP         (default: ALPHA_GRASP global).
    memetico     : se True, hibridiza com Busca Local a cada geracao (P1).
    orcamento_bl : avaliacoes gastas na BL por geracao (default: ORCAMENTO_BL).
    n_elite_bl   : quantos melhores individuos recebem a BL (default: N_ELITE_BL).

    Retorna um objeto Resultado (iteravel como a tupla antiga
    (melhor_individuo, melhor_fitness, n_calculos)).
    """
    if estrategia is None:
        estrategia = ESTRATEGIA_INICIAL
    if alpha is None:
        alpha = ALPHA_GRASP
    if memetico is None:
        memetico = MEMETICO
    if orcamento_bl is None:
        orcamento_bl = ORCAMENTO_BL
    if n_elite_bl is None:
        n_elite_bl = N_ELITE_BL

    rng = np.random.default_rng(SEMENTE)
    fobj = FuncaoObjetivo(VPL, VOL)
    n_prescricoes = VPL.shape[1]

    # 1) Inicializacao + 2) Avaliacao
    populacao = gerar_populacao_inicial(rng, VPL, estrategia, alpha)
    fitness = [fobj.avaliar(ind) for ind in populacao]
    fitness_inicial = list(fitness)   # snapshot da qualidade da pop. inicial

    melhor_idx = int(np.argmax(fitness))
    melhor_individuo = populacao[melhor_idx].copy()
    melhor_fitness = fitness[melhor_idx]

    # historico (n_calculos, melhor_fitness) para as curvas de convergencia
    historico = [(fobj.calculos, melhor_fitness)]

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

        # 6b) Intensificacao (AG Memetico): busca local nos melhores individuos.
        #     Refina os n_elite_bl melhores da nova populacao (Lamarckiano),
        #     respeitando o orcamento global de calculos da funcao objetivo.
        if memetico:
            melhores = np.argsort(nova_fit)[::-1][:n_elite_bl]
            for b in melhores:
                orc = min(orcamento_bl, MAX_CALCULOS - fobj.calculos)
                if orc <= 0:
                    break
                nova_pop[b], nova_fit[b] = busca_local_sistematica(
                    rng, fobj, nova_pop[b], nova_fit[b], orc, n_prescricoes)

        populacao, fitness = nova_pop, nova_fit

        # Atualiza o melhor global encontrado
        idx = int(np.argmax(fitness))
        if fitness[idx] > melhor_fitness:
            melhor_fitness = fitness[idx]
            melhor_individuo = populacao[idx].copy()

        # registra o melhor-ate-agora vs. numero de calculos (curva de convergencia)
        historico.append((fobj.calculos, melhor_fitness))

        if verbose and geracao % 20 == 0:
            print(f"Geracao {geracao:4d} | calculos {fobj.calculos:6d} | "
                  f"melhor VPL = R$ {melhor_fitness:,.2f}")

    return Resultado(melhor_individuo, melhor_fitness, fobj.calculos,
                     fitness_inicial, historico, estrategia, alpha)


# =============================================================================
# EXECUCAO
# =============================================================================

if __name__ == "__main__":
    print("Carregando base de dados...")
    VPL, VOL = carregar_dados(ARQUIVO_BASE)
    print(f"Base carregada: {VPL.shape[0]} talhoes x {VPL.shape[1]} prescricoes\n")

    print(f"Executando AG (estrategia inicial = {ESTRATEGIA_INICIAL}"
          + (f", alpha={ALPHA_GRASP}" if ESTRATEGIA_INICIAL == "grasp" else "")
          + (f", MEMETICO orc={ORCAMENTO_BL} elite_bl={N_ELITE_BL}" if MEMETICO else "")
          + f", max {MAX_CALCULOS} calculos)...\n")
    res = algoritmo_genetico(VPL, VOL)

    # VPL de referencia do artigo para calculo de eficiencia (JUNIOR et al., 2021)
    VPL_REFERENCIA = 32170883
    eficiencia = res.melhor_fitness / VPL_REFERENCIA

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print(f"Estrategia inicial          : {res.estrategia}"
          + (f" (alpha={res.alpha})" if res.estrategia == "grasp" else ""))
    print(f"Fitness medio da pop inicial: R$ {np.mean(res.fitness_inicial):,.2f}")
    print(f"Fitness max da pop inicial  : R$ {np.max(res.fitness_inicial):,.2f}")
    print(f"Calculos da funcao objetivo : {res.n_calculos}")
    print(f"Melhor VPL (penalizado)     : R$ {res.melhor_fitness:,.2f}")
    print(f"Eficiencia (VPL/VPL_ref)    : {eficiencia:.3f}  ({eficiencia*100:.1f}%)")
    print(f"Prescricoes escolhidas (talhao 1..120, valores 1..81):")
    print((res.melhor_individuo + 1).tolist())  # +1 para exibir no padrao do artigo
