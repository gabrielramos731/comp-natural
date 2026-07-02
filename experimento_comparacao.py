# -*- coding: utf-8 -*-
"""
Experimento COMPARATIVO de estrategias de inicializacao do AG florestal.

Compara duas formas de gerar a populacao inicial do Algoritmo Genetico,
mantendo TODO o restante do AG identico (selecao, cruzamento, mutacao,
substituicao e criterio de parada):

  A) "aleatoria" : solucoes completamente aleatorias        (construtiva alpha=1)
  B) "grasp"     : construtiva gulosa-aleatoria tipo GRASP   (Eq. 7, alpha=0.5)

Metodologia:
  - Cada estrategia e executada N_EXECUCOES vezes (sementes distintas), ate
    MAX_CALCULOS_ALVO calculos da funcao objetivo.
  - Como o "melhor-ate-agora" e monotonico, os resultados nos pontos de parada
    intermediarios (5000/10000/25000/50000) sao lidos diretamente do historico
    de cada execucao -- equivalente a rodar experimentos independentes em cada
    ponto, porem mais barato e ja fornecendo as curvas de convergencia.

Saidas (pasta resultados/):
  - resumo_comparacao.csv : tabela agregada (estilo Tabela 2 do artigo)
  - runs_finais.csv       : VPL final de cada execucao (para boxplots/dispersao)
  - convergencia.csv      : curva media de VPL x calculos (para graficos de evolucao)
  - qualidade_inicial.csv : qualidade da populacao inicial por estrategia
  - experimentos.json     : dump estruturado completo (reprodutibilidade)
  - ../EXPERIMENTOS.md     : log legivel, com um bloco por rodada de experimento
"""

import os
import json
import time
from datetime import datetime

import numpy as np

import ag_florestal as ag

# =============================================================================
# PARAMETROS DO EXPERIMENTO (editaveis)
# =============================================================================

N_EXECUCOES        = 15                          # repeticoes por estrategia
MAX_CALCULOS_ALVO  = 50000                       # calculos maximos por execucao
PONTOS_PARADA      = [5000, 10000, 25000, 50000] # checkpoints reportados
VPL_REFERENCIA     = 32170883                    # JUNIOR et al. (2021)

# Desenho fatorial 2x2: {inicializacao} x {com/sem Busca Local}.
# Cada configuracao altera SO a forma de gerar/refinar a populacao; o restante
# do AG (selecao, cruzamento, mutacao, substituicao, criterio de parada) e
# identico entre elas.
#   Inicializacao : "aleatoria" (alpha=1)  ou  "grasp" (RCL gulosa-aleatoria, Eq.7)
#   Busca Local   : sem BL (AG puro)        ou  com BL (AG memetico, P1)
ORCAMENTO_BL = 120   # avaliacoes gastas pela BL por geracao (memetico)
N_ELITE_BL   = 2     # quantos melhores individuos recebem a BL por geracao

ESTRATEGIAS = [
    {"rotulo": "Aleatoria",         "estrategia": "aleatoria", "alpha": None,
     "memetico": False},
    {"rotulo": "Aleatoria + BL",    "estrategia": "aleatoria", "alpha": None,
     "memetico": True},
    {"rotulo": "GRASP (a=0.5)",     "estrategia": "grasp",     "alpha": 0.5,
     "memetico": False},
    {"rotulo": "GRASP (a=0.5) + BL", "estrategia": "grasp",    "alpha": 0.5,
     "memetico": True},
]

PASTA_SAIDA = "resultados"
GRID_PASSO  = 250   # passo (em calculos) da grade das curvas de convergencia


def melhor_ate(historico, ponto):
    """Melhor VPL encontrado ate `ponto` calculos, lido do historico monotonico.

    historico: lista de (n_calculos, melhor_fitness). Retorna o melhor_fitness
    da ultima entrada cujo n_calculos <= ponto.
    """
    calcs = [c for c, _ in historico]
    idx = np.searchsorted(calcs, ponto, side="right") - 1
    idx = max(0, idx)
    return historico[idx][1]


def curva_em_grade(historico, grade):
    """Reamostra o historico (funcao escada monotonica) na `grade` de calculos."""
    calcs = np.array([c for c, _ in historico])
    vals  = np.array([v for _, v in historico])
    idx = np.searchsorted(calcs, grade, side="right") - 1
    idx = np.clip(idx, 0, len(vals) - 1)
    return vals[idx]


# =============================================================================
# EXECUCAO DO EXPERIMENTO
# =============================================================================

def rodar():
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    VPL, VOL = ag.carregar_dados(ag.ARQUIVO_BASE)
    print(f"Base: {VPL.shape[0]} talhoes x {VPL.shape[1]} prescricoes")
    print(f"Estrategias: {[c['rotulo'] for c in ESTRATEGIAS]}")
    print(f"{N_EXECUCOES} execucoes x {MAX_CALCULOS_ALVO} calculos por estrategia\n")

    ag.MAX_CALCULOS = MAX_CALCULOS_ALVO
    grade = np.arange(0, MAX_CALCULOS_ALVO + 1, GRID_PASSO)

    # estruturas acumuladoras
    runs_por_estrategia = {}     # rotulo -> lista de dicts por execucao
    curvas_por_estrategia = {}   # rotulo -> array (N_EXECUCOES, len(grade))
    inicial_por_estrategia = {}  # rotulo -> lista de (media, maximo) da pop inicial

    for cfg in ESTRATEGIAS:
        rot = cfg["rotulo"]
        print(f">>> Estrategia: {rot}")
        runs = []
        curvas = np.zeros((N_EXECUCOES, len(grade)))
        iniciais = []
        t0 = time.perf_counter()

        for execucao in range(N_EXECUCOES):
            ag.SEMENTE = execucao   # semente distinta por execucao (reprodutivel)
            res = ag.algoritmo_genetico(
                VPL, VOL, verbose=False,
                estrategia=cfg["estrategia"], alpha=cfg["alpha"],
                memetico=cfg["memetico"],
                orcamento_bl=ORCAMENTO_BL, n_elite_bl=N_ELITE_BL,
            )

            iniciais.append((float(np.mean(res.fitness_inicial)),
                             float(np.max(res.fitness_inicial))))
            curvas[execucao] = curva_em_grade(res.historico, grade)

            run = {
                "estrategia": rot,
                "execucao": execucao,
                "semente": execucao,
                "vpl_final": float(res.melhor_fitness),
                "eficiencia_final": float(res.melhor_fitness / VPL_REFERENCIA),
                "fitness_inicial_medio": float(np.mean(res.fitness_inicial)),
                "fitness_inicial_max": float(np.max(res.fitness_inicial)),
            }
            # melhor VPL em cada ponto de parada intermediario
            for ponto in PONTOS_PARADA:
                run[f"vpl_{ponto}"] = float(melhor_ate(res.historico, ponto))
            runs.append(run)

        tempo_medio = (time.perf_counter() - t0) / N_EXECUCOES
        for r in runs:
            r["tempo_medio_s"] = tempo_medio

        runs_por_estrategia[rot] = runs
        curvas_por_estrategia[rot] = curvas
        inicial_por_estrategia[rot] = iniciais
        print(f"    concluido em {time.perf_counter() - t0:.1f}s "
              f"({tempo_medio:.2f}s por execucao)\n")

    # -------------------------------------------------------------------------
    # AGREGACAO: tabela-resumo por (estrategia, ponto de parada)
    # -------------------------------------------------------------------------
    linhas_resumo = []
    for cfg in ESTRATEGIAS:
        rot = cfg["rotulo"]
        runs = runs_por_estrategia[rot]
        for ponto in PONTOS_PARADA:
            vals = np.array([r[f"vpl_{ponto}"] for r in runs])
            linhas_resumo.append({
                "estrategia": rot,
                "num_calculos": ponto,
                "media": float(vals.mean()),
                "maximo": float(vals.max()),
                "minimo": float(vals.min()),
                "desvio_padrao": float(vals.std(ddof=1)),
                "eficiencia_media": float(vals.mean() / VPL_REFERENCIA),
                "eficiencia_maxima": float(vals.max() / VPL_REFERENCIA),
                "tempo_medio_s": float(runs[0]["tempo_medio_s"]),
            })

    # -------------------------------------------------------------------------
    # SALVAR ARQUIVOS DE RESULTADO
    # -------------------------------------------------------------------------
    _salvar_csv(os.path.join(PASTA_SAIDA, "resumo_comparacao.csv"),
                linhas_resumo,
                ["estrategia", "num_calculos", "media", "maximo", "minimo",
                 "desvio_padrao", "eficiencia_media", "eficiencia_maxima",
                 "tempo_medio_s"])

    runs_flat = [r for rot in runs_por_estrategia
                 for r in runs_por_estrategia[rot]]
    campos_runs = (["estrategia", "execucao", "semente", "vpl_final",
                    "eficiencia_final"]
                   + [f"vpl_{p}" for p in PONTOS_PARADA]
                   + ["fitness_inicial_medio", "fitness_inicial_max",
                      "tempo_medio_s"])
    _salvar_csv(os.path.join(PASTA_SAIDA, "runs_finais.csv"),
                runs_flat, campos_runs)

    # curvas de convergencia (media e desvio por estrategia na grade)
    linhas_conv = []
    for cfg in ESTRATEGIAS:
        rot = cfg["rotulo"]
        curvas = curvas_por_estrategia[rot]
        media = curvas.mean(axis=0)
        desvio = curvas.std(axis=0, ddof=1)
        for k, calc in enumerate(grade):
            linhas_conv.append({
                "estrategia": rot,
                "num_calculos": int(calc),
                "vpl_medio": float(media[k]),
                "vpl_desvio": float(desvio[k]),
                "eficiencia_media": float(media[k] / VPL_REFERENCIA),
            })
    _salvar_csv(os.path.join(PASTA_SAIDA, "convergencia.csv"),
                linhas_conv,
                ["estrategia", "num_calculos", "vpl_medio", "vpl_desvio",
                 "eficiencia_media"])

    # qualidade da populacao inicial
    linhas_ini = []
    for cfg in ESTRATEGIAS:
        rot = cfg["rotulo"]
        arr = np.array(inicial_por_estrategia[rot])
        linhas_ini.append({
            "estrategia": rot,
            "fitness_inicial_medio": float(arr[:, 0].mean()),
            "fitness_inicial_max_medio": float(arr[:, 1].mean()),
            "fitness_inicial_max_global": float(arr[:, 1].max()),
        })
    _salvar_csv(os.path.join(PASTA_SAIDA, "qualidade_inicial.csv"),
                linhas_ini,
                ["estrategia", "fitness_inicial_medio",
                 "fitness_inicial_max_medio", "fitness_inicial_max_global"])

    # dump JSON completo
    dump = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "parametros": {
            "n_execucoes": N_EXECUCOES,
            "max_calculos_alvo": MAX_CALCULOS_ALVO,
            "pontos_parada": PONTOS_PARADA,
            "vpl_referencia": VPL_REFERENCIA,
            "estrategias": [c["rotulo"] for c in ESTRATEGIAS],
            "memetico": {"orcamento_bl": ORCAMENTO_BL, "n_elite_bl": N_ELITE_BL},
            "ag": {
                "tam_populacao": ag.TAM_POPULACAO,
                "n_filhos": ag.N_FILHOS,
                "tam_torneio": ag.TAM_TORNEIO,
                "taxa_mutacao": ag.TAXA_MUTACAO,
                "n_elite": ag.N_ELITE,
                "d_min": ag.D_MIN, "d_max": ag.D_MAX,
                "penalidade": ag.PENALIDADE,
            },
        },
        "resumo": linhas_resumo,
        "runs": runs_flat,
        "qualidade_inicial": linhas_ini,
    }
    with open(os.path.join(PASTA_SAIDA, "experimentos.json"), "w",
              encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)

    _imprimir_resumo(linhas_resumo)
    _anexar_log(dump, linhas_resumo, linhas_ini)
    print(f"\nArquivos salvos em '{PASTA_SAIDA}/' e log em 'EXPERIMENTOS.md'.")
    return dump


# =============================================================================
# UTILITARIOS DE SAIDA
# =============================================================================

def _salvar_csv(caminho, linhas, campos):
    import csv
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for linha in linhas:
            w.writerow({c: linha.get(c, "") for c in campos})


def _imprimir_resumo(linhas_resumo):
    print("\n" + "=" * 96)
    print("RESUMO COMPARATIVO (media de %d execucoes)" % N_EXECUCOES)
    print("=" * 96)
    cab = (f"{'Estrategia':<14} | {'Calc':>6} | {'Media (R$)':>16} | "
           f"{'Maximo (R$)':>16} | {'Minimo (R$)':>16} | {'Efic':>5}")
    print(cab)
    print("-" * len(cab))
    for l in linhas_resumo:
        print(f"{l['estrategia']:<14} | {l['num_calculos']:>6} | "
              f"{l['media']:>16,.0f} | {l['maximo']:>16,.0f} | "
              f"{l['minimo']:>16,.0f} | {l['eficiencia_media']:>5.3f}")


def _anexar_log(dump, linhas_resumo, linhas_ini):
    """Anexa um bloco legivel ao arquivo de experimentos EXPERIMENTOS.md."""
    p = dump["parametros"]
    linhas = []
    linhas.append(f"\n## Experimento {dump['gerado_em']}\n")
    linhas.append("**Comparacao de inicializacao do AG:** "
                  + " vs ".join(p["estrategias"]) + ".\n")
    linhas.append(f"- Execucoes por estrategia: {p['n_execucoes']} "
                  f"(sementes 0..{p['n_execucoes'] - 1})")
    linhas.append(f"- Calculos maximos: {p['max_calculos_alvo']} "
                  f"| Pontos de parada: {p['pontos_parada']}")
    linhas.append(f"- VPL de referencia: R$ {p['vpl_referencia']:,}")
    linhas.append(f"- AG: pop={p['ag']['tam_populacao']}, filhos={p['ag']['n_filhos']}, "
                  f"torneio={p['ag']['tam_torneio']}, mut={p['ag']['taxa_mutacao']}, "
                  f"elite={p['ag']['n_elite']}")
    linhas.append(f"- Busca Local (configs '+ BL'): vizinhanca sistematica, "
                  f"primeira-melhora, orcamento={p['memetico']['orcamento_bl']} "
                  f"aval./geracao, aplicada aos {p['memetico']['n_elite_bl']} "
                  f"melhores individuos\n")

    linhas.append("### Qualidade da populacao inicial\n")
    linhas.append("| Estrategia | Fitness medio (R$) | Melhor individuo (R$) |")
    linhas.append("|---|---:|---:|")
    for l in linhas_ini:
        linhas.append(f"| {l['estrategia']} | {l['fitness_inicial_medio']:,.0f} "
                      f"| {l['fitness_inicial_max_medio']:,.0f} |")
    linhas.append("")

    linhas.append("### Resultado por ponto de parada\n")
    linhas.append("| Estrategia | Calculos | Media (R$) | Maximo (R$) | "
                  "Minimo (R$) | Desvio (R$) | Efic. media | Efic. max |")
    linhas.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for l in linhas_resumo:
        linhas.append(
            f"| {l['estrategia']} | {l['num_calculos']} | {l['media']:,.0f} "
            f"| {l['maximo']:,.0f} | {l['minimo']:,.0f} "
            f"| {l['desvio_padrao']:,.0f} | {l['eficiencia_media']:.3f} "
            f"| {l['eficiencia_maxima']:.3f} |")
    linhas.append("")

    cabecalho_arquivo = (
        "# Registro de Experimentos - AG Planejamento Florestal\n\n"
        "Log automatico gerado por `experimento_comparacao.py`. Cada bloco "
        "corresponde a uma rodada completa de experimentos. Dados brutos na "
        "pasta `resultados/`.\n"
    )
    caminho = "EXPERIMENTOS.md"
    novo = not os.path.exists(caminho)
    with open(caminho, "a", encoding="utf-8") as f:
        if novo:
            f.write(cabecalho_arquivo)
        f.write("\n".join(linhas))


if __name__ == "__main__":
    rodar()
