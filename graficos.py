# -*- coding: utf-8 -*-
"""
Gera as figuras da apresentacao (slides) para a *proposta de melhoria* do
Algoritmo Genetico no problema de planejamento florestal.

A ideia central do trabalho: o AG do artigo original (SILVA & ROCHA) teve o
PIOR desempenho entre as heuristicas (eficiencia 0.204 -> 0.801) e os autores
concluiram que "uma revisao aprofundada dos metodos implementados se faz
necessaria". Esta versao revisada do AG atinge eficiencia ~0.94. As figuras
comparam as duas versoes e mostram a convergencia.

Uso:
    python graficos.py            # roda o experimento e salva as figuras em figuras/

Saidas (pasta figuras/):
    fig1_convergencia.png      curva de convergencia (VPL medio +/- desvio)
    fig2_comparacao_efic.png   barras: eficiencia AG-artigo vs AG-revisado
    fig3_boxplot.png           distribuicao das 15 execucoes por ponto de parada
    fig4_vpl_maximo.png        evolucao do VPL maximo vs calculos (artigo vs revisado)
    resultados_ag.csv          tabela de resultados (estilo Tabela 2 do artigo)
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import ag_florestal as ag

# =============================================================================
# CONFIGURACAO DO EXPERIMENTO
# =============================================================================

N_EXECUCOES = 15                              # repeticoes (mesmo do artigo)
PONTOS_PARADA = [5000, 10000, 25000, 50000]   # criterios de parada avaliados
MAX_CALCULOS = 50000                          # rodamos ate o maior ponto...
                                              # ...e derivamos os menores do historico

VPL_REFERENCIA = 32_170_883                   # branch-and-bound (JUNIOR et al., 2021)
LIMITE_SUPERIOR = 36_334_070                  # limite do problema relaxado (artigo)

PASTA_SAIDA = "figuras"

# Resultados do AG *original* do artigo (Tabela 2) — baseline a ser superado.
AG_ARTIGO = {
    5000:  dict(media=6_554_814,  maximo=11_529_932, minimo=-14_926_232),
    10000: dict(media=13_404_776, maximo=22_868_922, minimo=5_297_959),
    25000: dict(media=21_682_053, maximo=28_287_942, minimo=2_927_771),
    50000: dict(media=25_773_563, maximo=29_245_765, minimo=18_315_377),
}

# Paleta / estilo
COR_REVISADO = "#1b7837"   # verde (nossa versao)
COR_ARTIGO   = "#b2182b"   # vermelho (artigo)
COR_REF      = "#2166ac"   # azul (referencia)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def milhoes(x, _pos=None):
    """Formata valores do eixo em milhoes de reais."""
    return f"{x/1e6:.0f}M"


# =============================================================================
# EXECUCAO DO EXPERIMENTO (com registro de historico)
# =============================================================================

def rodar_experimento():
    """Roda o AG N_EXECUCOES vezes ate MAX_CALCULOS, guardando o historico.

    Como o AG avalia a funcao objetivo em blocos fixos (40 iniciais + 40 por
    geracao), todas as execucoes compartilham exatamente a mesma grade de
    'numero de calculos'. Isso permite empilhar as curvas em uma matriz.

    Retorna:
      grade    : vetor (P,) com o numero de calculos em cada ponto do historico
      curvas   : matriz (N_EXECUCOES, P) com o melhor VPL-ate-o-momento
      tempos   : lista de tempos (s) de cada execucao
    """
    print("Carregando base de dados...")
    VPL, VOL = ag.carregar_dados(ag.ARQUIVO_BASE)
    print(f"Base: {VPL.shape[0]} talhoes x {VPL.shape[1]} prescricoes\n")

    ag.MAX_CALCULOS = MAX_CALCULOS
    curvas, tempos, grade = [], [], None

    for execucao in range(N_EXECUCOES):
        ag.SEMENTE = execucao                 # semente distinta por execucao
        t0 = time.time()
        _, valor, _, hist = ag.algoritmo_genetico(
            VPL, VOL, verbose=False, registrar_historico=True
        )
        tempos.append(time.time() - t0)

        calc = np.array([h[0] for h in hist])
        fit = np.array([h[1] for h in hist])
        if grade is None:
            grade = calc
        curvas.append(fit)
        print(f"  execucao {execucao+1:2d}/{N_EXECUCOES} | "
              f"VPL final = R$ {valor:15,.2f} | {tempos[-1]:5.1f}s")

    curvas = np.vstack(curvas)
    print(f"\nExperimento concluido em {sum(tempos):.1f}s "
          f"({sum(tempos)/N_EXECUCOES:.1f}s por execucao).\n")
    return grade, curvas, tempos


def valores_no_ponto(grade, curvas, ponto):
    """Extrai o VPL de cada execucao no ponto de parada dado (ultimo <= ponto)."""
    j = np.searchsorted(grade, ponto, side="right") - 1
    return curvas[:, j]


# =============================================================================
# FIGURA 1 — CURVA DE CONVERGENCIA (VPL medio +/- desvio padrao)
# =============================================================================

def fig_convergencia(grade, curvas):
    media = curvas.mean(axis=0)
    desvio = curvas.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(grade, media, color=COR_REVISADO, lw=2.2,
            label="AG revisado — VPL medio (15 execucoes)")
    ax.fill_between(grade, media - desvio, media + desvio,
                    color=COR_REVISADO, alpha=0.20, label="+/- 1 desvio padrao")

    ax.axhline(VPL_REFERENCIA, color=COR_REF, ls="--", lw=1.5,
               label=f"VPL de referencia (B&B) = R$ {VPL_REFERENCIA/1e6:.1f}M")
    ax.axhline(LIMITE_SUPERIOR, color="0.4", ls=":", lw=1.5,
               label=f"Limite superior (relaxado) = R$ {LIMITE_SUPERIOR/1e6:.1f}M")

    for p in PONTOS_PARADA:
        ax.axvline(p, color="0.8", lw=0.8, zorder=0)

    ax.set_xlabel("Numero de calculos da funcao objetivo")
    ax.set_ylabel("VPL (R$)")
    ax.set_title("Convergencia do AG revisado", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(milhoes))
    ax.set_xlim(0, MAX_CALCULOS)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    fig.tight_layout()
    _salvar(fig, "fig1_convergencia.png")


# =============================================================================
# FIGURA 2 — COMPARACAO DE EFICIENCIA (barras: artigo vs revisado)
# =============================================================================

def fig_comparacao_eficiencia(grade, curvas):
    efic_revisado, efic_artigo = [], []
    for p in PONTOS_PARADA:
        v = valores_no_ponto(grade, curvas, p)
        efic_revisado.append(v.mean() / VPL_REFERENCIA)
        efic_artigo.append(AG_ARTIGO[p]["media"] / VPL_REFERENCIA)

    x = np.arange(len(PONTOS_PARADA))
    largura = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - largura/2, efic_artigo, largura,
                color=COR_ARTIGO, label="AG do artigo (original)")
    b2 = ax.bar(x + largura/2, efic_revisado, largura,
                color=COR_REVISADO, label="AG revisado (proposto)")

    for barras in (b1, b2):
        for b in barras:
            h = b.get_height()
            ax.annotate(f"{h*100:.1f}%", (b.get_x() + b.get_width()/2, h),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=9.5, fontweight="bold")

    ax.axhline(1.0, color=COR_REF, ls="--", lw=1.2,
               label="Referencia (100%)")
    ax.set_xticks(x, [f"{p:,}".replace(",", ".") for p in PONTOS_PARADA])
    ax.set_xlabel("Numero maximo de calculos da funcao objetivo")
    ax.set_ylabel("Eficiencia media (VPL / VPL_ref)")
    ax.set_title("Eficiencia do AG: artigo original vs. versao revisada",
                 fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    _salvar(fig, "fig2_comparacao_efic.png")


# =============================================================================
# FIGURA 3 — BOXPLOT DA DISTRIBUICAO DAS 15 EXECUCOES POR PONTO DE PARADA
# =============================================================================

def fig_boxplot(grade, curvas):
    dados = [valores_no_ponto(grade, curvas, p) for p in PONTOS_PARADA]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bp = ax.boxplot(dados, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", lw=1.6),
                    flierprops=dict(marker="o", markersize=4, alpha=0.6))
    for caixa in bp["boxes"]:
        caixa.set(facecolor=COR_REVISADO, alpha=0.35)

    # media do AG do artigo como marcador de comparacao
    medias_artigo = [AG_ARTIGO[p]["media"] for p in PONTOS_PARADA]
    ax.plot(np.arange(1, len(PONTOS_PARADA)+1), medias_artigo,
            "X", color=COR_ARTIGO, markersize=11,
            label="Media do AG do artigo", zorder=5)

    ax.axhline(VPL_REFERENCIA, color=COR_REF, ls="--", lw=1.3,
               label=f"VPL de referencia = R$ {VPL_REFERENCIA/1e6:.1f}M")

    ax.set_xticklabels([f"{p:,}".replace(",", ".") for p in PONTOS_PARADA])
    ax.set_xlabel("Numero maximo de calculos da funcao objetivo")
    ax.set_ylabel("VPL (R$)")
    ax.set_title("Distribuicao do VPL nas 15 execucoes (AG revisado)",
                 fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(milhoes))
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    _salvar(fig, "fig3_boxplot.png")


# =============================================================================
# FIGURA 4 — EVOLUCAO DO VPL MAXIMO vs CALCULOS (artigo vs revisado)
# =============================================================================

def fig_vpl_maximo(grade, curvas):
    max_revisado = [valores_no_ponto(grade, curvas, p).max() for p in PONTOS_PARADA]
    max_artigo = [AG_ARTIGO[p]["maximo"] for p in PONTOS_PARADA]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(PONTOS_PARADA, max_revisado, "o-", color=COR_REVISADO, lw=2.2,
            markersize=8, label="AG revisado (proposto)")
    ax.plot(PONTOS_PARADA, max_artigo, "s--", color=COR_ARTIGO, lw=2.0,
            markersize=7, label="AG do artigo (original)")
    ax.axhline(VPL_REFERENCIA, color=COR_REF, ls="--", lw=1.3,
               label=f"VPL de referencia = R$ {VPL_REFERENCIA/1e6:.1f}M")

    for xp, yp in zip(PONTOS_PARADA, max_revisado):
        ax.annotate(f"{yp/1e6:.1f}M", (xp, yp), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9,
                    color=COR_REVISADO, fontweight="bold")

    ax.set_xlabel("Numero maximo de calculos da funcao objetivo")
    ax.set_ylabel("VPL maximo (R$)")
    ax.set_title("Melhor VPL encontrado vs. esforco computacional",
                 fontweight="bold")
    ax.set_xticks(PONTOS_PARADA)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _p: f"{v/1000:.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(milhoes))
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    _salvar(fig, "fig4_vpl_maximo.png")


# =============================================================================
# TABELA DE RESULTADOS (CSV + impressao)
# =============================================================================

def tabela_resultados(grade, curvas):
    linhas = []
    print(f"{'Calculos':>9} | {'Media (R$)':>16} | {'Maximo (R$)':>16} | "
          f"{'Minimo (R$)':>16} | {'Efic.':>6}")
    print("-" * 76)
    for p in PONTOS_PARADA:
        v = valores_no_ponto(grade, curvas, p)
        media, maximo, minimo = v.mean(), v.max(), v.min()
        efic = media / VPL_REFERENCIA
        print(f"{p:>9} | {media:>16,.2f} | {maximo:>16,.2f} | "
              f"{minimo:>16,.2f} | {efic:>6.3f}")
        linhas.append((p, media, maximo, minimo, efic,
                       AG_ARTIGO[p]["media"] / VPL_REFERENCIA))

    caminho = os.path.join(PASTA_SAIDA, "resultados_ag.csv")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("calculos,media,maximo,minimo,eficiencia_revisado,eficiencia_artigo\n")
        for lin in linhas:
            f.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x)
                             for x in lin) + "\n")
    print(f"\nTabela salva em {caminho}")


# =============================================================================
# UTILITARIOS
# =============================================================================

def _salvar(fig, nome):
    caminho = os.path.join(PASTA_SAIDA, nome)
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura salva: {caminho}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    grade, curvas, _tempos = rodar_experimento()

    print("Gerando figuras...")
    fig_convergencia(grade, curvas)
    fig_comparacao_eficiencia(grade, curvas)
    fig_boxplot(grade, curvas)
    fig_vpl_maximo(grade, curvas)

    print()
    tabela_resultados(grade, curvas)
    print("\nConcluido. Figuras e tabela na pasta 'figuras/'.")
