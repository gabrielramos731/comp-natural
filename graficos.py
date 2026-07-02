# -*- coding: utf-8 -*-
"""
Gera os graficos e tabelas de comparacao do experimento fatorial 2x2
(inicializacao {aleatoria, GRASP} x busca local {sem, com}).

Le os CSV/JSON produzidos por experimento_comparacao.py (pasta resultados/)
e salva as figuras em resultados/figuras/ e tabelas em resultados/.

Figuras geradas:
  - convergencia.png        : evolucao do VPL/eficiencia media x nº de calculos
                              (painel completo + zoom na regiao positiva)
  - barras_50000.png        : eficiencia media das 4 configuracoes em 50000 calc.
  - efeito_busca_local.png  : efeito fatorial da BL (sem vs com) por inicializacao
  - boxplot_50000.png       : distribuicao do VPL final (15 execucoes) por config.
  - tabela_50000.png        : tabela-resumo (imagem) do ponto de 50000 calculos
Tabelas geradas:
  - tabela_comparacao.md    : tabela markdown com todos os pontos de parada
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTA = "resultados"
FIG   = os.path.join(PASTA, "figuras")
os.makedirs(FIG, exist_ok=True)

# ---- ordem e estilo visual consistentes entre todas as figuras ----
ORDEM = ["Aleatoria", "Aleatoria + BL", "GRASP (a=0.5)", "GRASP (a=0.5) + BL"]
ESTILO = {
    "Aleatoria":          dict(color="#1f77b4", ls="--", marker="o"),
    "Aleatoria + BL":     dict(color="#1f77b4", ls="-",  marker="o"),
    "GRASP (a=0.5)":      dict(color="#d62728", ls="--", marker="s"),
    "GRASP (a=0.5) + BL": dict(color="#d62728", ls="-",  marker="s"),
}

with open(os.path.join(PASTA, "experimentos.json"), encoding="utf-8") as f:
    META = json.load(f)
VREF = META["parametros"]["vpl_referencia"]
N_EXEC = META["parametros"]["n_execucoes"]
PONTOS = META["parametros"]["pontos_parada"]

conv   = pd.read_csv(os.path.join(PASTA, "convergencia.csv"))
resumo = pd.read_csv(os.path.join(PASTA, "resumo_comparacao.csv"))
runs   = pd.read_csv(os.path.join(PASTA, "runs_finais.csv"))

MILHOES = 1e6


# =============================================================================
# 1) CURVAS DE CONVERGENCIA (painel completo + zoom)
# =============================================================================
def grafico_convergencia():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    for rot in ORDEM:
        d = conv[conv["estrategia"] == rot].sort_values("num_calculos")
        x = d["num_calculos"].to_numpy()
        efic = d["eficiencia_media"].to_numpy()
        st = ESTILO[rot]
        # painel completo (mostra o "mergulho" das configuracoes GRASP)
        ax1.plot(x, efic, label=rot, color=st["color"], ls=st["ls"], lw=2)
        # painel de zoom (regiao positiva, onde as diferencas finas aparecem)
        ax2.plot(x, efic, label=rot, color=st["color"], ls=st["ls"], lw=2)

    for ax in (ax1, ax2):
        ax.axhline(1.0, color="gray", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("Nº de cálculos da função objetivo")
        ax.grid(alpha=0.3)
    ax1.set_ylabel("Eficiência média (VPL / VPL_ref)")
    ax1.set_title("(a) Convergência — visão completa")
    ax2.set_title("(b) Convergência — zoom na região positiva")
    ax2.set_ylim(0.70, 0.98)
    ax1.legend(fontsize=9, loc="lower right")

    fig.suptitle(f"Evolução do VPL médio ({N_EXEC} execuções) — fatorial "
                 f"inicialização × busca local", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG, "convergencia.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# 2) BARRAS: eficiencia media em 50000 calculos (com desvio-padrao)
# =============================================================================
def grafico_barras_50000():
    ponto = max(PONTOS)
    d = resumo[resumo["num_calculos"] == ponto].set_index("estrategia")
    efic = [d.loc[r, "eficiencia_media"] for r in ORDEM]
    cores = [ESTILO[r]["color"] for r in ORDEM]
    hatch = ["", "///", "", "///"]  # BL = hachurado

    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(range(len(ORDEM)), efic,
                    color=cores, edgecolor="black", alpha=0.85)
    for b, h in zip(barras, hatch):
        b.set_hatch(h)
    for i, e in enumerate(efic):
        ax.text(i, e + 0.002, f"{e:.3f}", ha="center", fontsize=11,
                fontweight="bold")

    ax.set_xticks(range(len(ORDEM)))
    ax.set_xticklabels(ORDEM, rotation=12)
    ax.set_ylabel("Eficiência média (VPL / VPL_ref)")
    ax.set_ylim(0.83, 0.975)
    ax.axhline(0.958, color="green", ls=":", lw=1.2,
               label="melhor do artigo (SA = 0.958)")
    ax.set_title(f"Eficiência média em {ponto:,} cálculos "
                 f"({N_EXEC} execuções; barra hachurada = com BL)\n"
                 f"dispersão detalhada no boxplot")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "barras_50000.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# 3) EFEITO FATORIAL DA BUSCA LOCAL (sem vs com, por inicializacao)
# =============================================================================
def grafico_efeito_bl():
    ponto = max(PONTOS)
    d = resumo[resumo["num_calculos"] == ponto].set_index("estrategia")
    grupos = ["Aleatória", "GRASP (α=0.5)"]
    sem = [d.loc["Aleatoria", "eficiencia_media"],
           d.loc["GRASP (a=0.5)", "eficiencia_media"]]
    com = [d.loc["Aleatoria + BL", "eficiencia_media"],
           d.loc["GRASP (a=0.5) + BL", "eficiencia_media"]]

    x = np.arange(len(grupos))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5))
    b1 = ax.bar(x - w/2, sem, w, label="AG puro (sem BL)",
                color="#bbbbbb", edgecolor="black")
    b2 = ax.bar(x + w/2, com, w, label="AG memético (com BL)",
                color="#2ca02c", edgecolor="black")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004,
                    f"{b.get_height():.3f}", ha="center", fontsize=10)
    # setas de ganho
    for i in range(len(grupos)):
        ax.annotate("", xy=(x[i] + w/2, com[i]), xytext=(x[i] - w/2, sem[i]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2,
                                    alpha=0.6))
        ax.text(x[i], max(sem[i], com[i]) + 0.02,
                f"+{(com[i]-sem[i]):.3f}", ha="center", color="#2ca02c",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(grupos)
    ax.set_ylabel("Eficiência média (VPL / VPL_ref)")
    ax.set_ylim(0.80, 1.0)
    ax.set_title(f"Efeito da Busca Local em {ponto:,} cálculos "
                 f"({N_EXEC} execuções)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "efeito_busca_local.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# 4) BOXPLOT da distribuicao do VPL final (50000 calc.) por configuracao
# =============================================================================
def grafico_boxplot():
    ponto = max(PONTOS)
    col = f"vpl_{ponto}"
    dados = [runs[runs["estrategia"] == r][col].to_numpy() / VREF for r in ORDEM]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bp = ax.boxplot(dados, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black"))
    for patch, r in zip(bp["boxes"], ORDEM):
        patch.set_facecolor(ESTILO[r]["color"])
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black")

    ax.set_xticklabels(ORDEM, rotation=12)
    ax.set_ylabel("Eficiência final (VPL / VPL_ref)")
    piso = 0.88
    ax.set_ylim(piso, 0.97)
    # anota configuracoes cuja distribuicao vaza abaixo do piso do eixo
    for i, (arr, r) in enumerate(zip(dados, ORDEM), start=1):
        if arr.min() < piso:
            ax.annotate(f"↓ mín {arr.min():.3f}", xy=(i, piso),
                        xytext=(i, piso + 0.006), ha="center", fontsize=9,
                        color="#7a1f1f", fontweight="bold")
    ax.set_title(f"Distribuição do VPL final em {ponto:,} cálculos "
                 f"({N_EXEC} execuções; ◇ = média)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "boxplot_50000.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# 5) TABELA-RESUMO (imagem) do ponto de 50000 calculos
# =============================================================================
def tabela_50000_png():
    ponto = max(PONTOS)
    d = resumo[resumo["num_calculos"] == ponto].set_index("estrategia")
    colunas = ["Média (R$)", "Máximo (R$)", "Mínimo (R$)",
               "Desvio (R$)", "Efic. média", "Efic. máx."]
    linhas = []
    for r in ORDEM:
        linhas.append([
            f"{d.loc[r, 'media']:,.0f}",
            f"{d.loc[r, 'maximo']:,.0f}",
            f"{d.loc[r, 'minimo']:,.0f}",
            f"{d.loc[r, 'desvio_padrao']:,.0f}",
            f"{d.loc[r, 'eficiencia_media']:.3f}",
            f"{d.loc[r, 'eficiencia_maxima']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.axis("off")
    tab = ax.table(cellText=linhas, rowLabels=ORDEM, colLabels=colunas,
                   cellLoc="center", rowLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.6)
    # destaca a melhor eficiencia media
    melhor = int(np.argmax([d.loc[r, "eficiencia_media"] for r in ORDEM]))
    for c in range(len(colunas)):
        tab[(melhor + 1, c)].set_facecolor("#d9f0d3")
    tab[(melhor + 1, -1)].set_facecolor("#d9f0d3")
    for c in range(len(colunas)):
        tab[(0, c)].set_facecolor("#40466e")
        tab[(0, c)].set_text_props(color="white", fontweight="bold")
    ax.set_title(f"Resultados em {ponto:,} cálculos da função objetivo "
                 f"({N_EXEC} execuções)", fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tabela_50000.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 6) TABELA MARKDOWN com todos os pontos de parada
# =============================================================================
def tabela_markdown():
    linhas = ["# Tabela de comparação — fatorial inicialização × busca local\n",
              f"_{N_EXEC} execuções por configuração; eficiência = VPL / "
              f"VPL_ref (R$ {VREF:,})._\n",
              "| Configuração | Cálculos | Média (R$) | Máximo (R$) | "
              "Mínimo (R$) | Desvio (R$) | Efic. média | Efic. máx |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ORDEM:
        for p in PONTOS:
            row = resumo[(resumo["estrategia"] == r) &
                         (resumo["num_calculos"] == p)].iloc[0]
            linhas.append(
                f"| {r} | {p} | {row['media']:,.0f} | {row['maximo']:,.0f} "
                f"| {row['minimo']:,.0f} | {row['desvio_padrao']:,.0f} "
                f"| {row['eficiencia_media']:.3f} | "
                f"{row['eficiencia_maxima']:.3f} |")
    with open(os.path.join(PASTA, "tabela_comparacao.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")


if __name__ == "__main__":
    grafico_convergencia()
    grafico_barras_50000()
    grafico_efeito_bl()
    grafico_boxplot()
    tabela_50000_png()
    tabela_markdown()
    print("Figuras salvas em:", FIG)
    for nome in sorted(os.listdir(FIG)):
        print("  -", nome)
    print("Tabela markdown:", os.path.join(PASTA, "tabela_comparacao.md"))
