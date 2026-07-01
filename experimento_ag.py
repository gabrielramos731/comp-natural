# -*- coding: utf-8 -*-
"""
Reproduz a tabela de resultados do AG (estilo Tabela 2 do artigo):
executa o AG N vezes para cada ponto de parada e reporta media, maximo,
minimo e eficiencia.
"""

import time
import numpy as np

import ag_florestal as ag

# --- Parametros do experimento (editaveis) ---
N_EXECUCOES   = 15                              # repeticoes por ponto de parada
PONTOS_PARADA = [5000, 10000, 25000, 50000]     # numeros maximos de calculos
VPL_REFERENCIA = 32170883                        # JUNIOR et al. (2021)

if __name__ == "__main__":
    VPL, VOL = ag.carregar_dados(ag.ARQUIVO_BASE)
    print(f"Base: {VPL.shape[0]} talhoes x {VPL.shape[1]} prescricoes\n")

    cabecalho = f"{'Calculos':>9} | {'Media (R$)':>16} | {'Maximo (R$)':>16} | {'Minimo (R$)':>16} | {'Efic.':>6} | {'Tempo(s)':>8}"
    print(cabecalho)
    print("-" * len(cabecalho))

    for max_calc in PONTOS_PARADA:
        ag.MAX_CALCULOS = max_calc
        resultados = []
        t0 = time.time()
        for execucao in range(N_EXECUCOES):
            # semente diferente por execucao para evitar vies (artigo: 15 execucoes)
            ag.SEMENTE = execucao
            _, valor, _ = ag.algoritmo_genetico(VPL, VOL, verbose=False)
            resultados.append(valor)
        tempo_medio = (time.time() - t0) / N_EXECUCOES

        media  = np.mean(resultados)
        maximo = np.max(resultados)
        minimo = np.min(resultados)
        efic   = media / VPL_REFERENCIA

        print(f"{max_calc:>9} | {media:>16,.2f} | {maximo:>16,.2f} | "
              f"{minimo:>16,.2f} | {efic:>6.3f} | {tempo_medio:>8.1f}")
