# AG - Planejamento Florestal (Base120)

Implementação em Python do **Algoritmo Genético** descrito na seção 3.5 do
artigo, para o problema de planejamento florestal: escolher uma prescrição
para cada um dos 120 talhões maximizando o VPL e penalizando a produção anual
de madeira fora dos limites `[140000, 160000] m³` (Eq. 6).

## Arquivos
- `ag_florestal.py` — implementação do AG (uma execução).
- `experimento_ag.py` — roda o AG 15× em cada ponto de parada e gera a tabela de resultados (estilo Tabela 2).
- `graficos.py` — roda o experimento registrando o histórico e gera as **figuras da apresentação** (pasta `figuras/`).
- `Base120.xlsx` — base de dados (120 talhões × 81 prescrições × 16 anos).

## Como executar
```bash
pip install pandas numpy openpyxl matplotlib
python ag_florestal.py        # uma execução, mostra a evolução e o melhor VPL
python experimento_ag.py      # reproduz a tabela (15 execuções × 4 pontos)
python graficos.py            # gera as figuras dos slides em figuras/
```

## Figuras geradas (`graficos.py`)
| Arquivo | Conteúdo |
|---|---|
| `fig1_convergencia.png` | Curva de convergência do VPL médio (± desvio) vs nº de cálculos, com o VPL de referência e o limite superior. |
| `fig2_comparacao_efic.png` | **Gráfico-chave:** barras de eficiência do AG do artigo vs. AG revisado nos 4 pontos de parada. |
| `fig3_boxplot.png` | Distribuição das 15 execuções por ponto de parada, com a média do AG do artigo marcada. |
| `fig4_vpl_maximo.png` | Melhor VPL encontrado vs. esforço computacional (artigo vs. revisado). |
| `resultados_ag.csv` | Tabela de resultados (estilo Tabela 2). |

## Parâmetros editáveis (topo do `ag_florestal.py`)

| Parâmetro | Valor (artigo) | Significado |
|---|---|---|
| `D_MIN`, `D_MAX` | 140000, 160000 | limites de produção anual de madeira (m³) |
| `PENALIDADE` | 500 | R$ por m³ fora dos limites (Pe) |
| `TAM_POPULACAO` | 40 | indivíduos na população |
| `N_FILHOS` | 40 | filhos gerados por geração |
| `TAM_TORNEIO` | 4 | indivíduos por torneio na seleção |
| `TAXA_MUTACAO` | 0.01 | probabilidade de mutação (1%) |
| `N_ELITE` | 10 | melhores preservados na substituição |
| `MAX_CALCULOS` | 5000/10000/25000/50000 | critério de parada (nº de avaliações) |
| `SEMENTE` | 42 | semente aleatória (use `None` para variar) |

## Mapeamento artigo → código
- **Inicialização** (passo 1): `populacao_inicial` — 40 soluções aleatórias (construtiva α=1 + embaralhamento).
- **Avaliação** (passo 2 / Eq. 6): `FuncaoObjetivo.avaliar` — VPL total − penalidade do desvio anual.
- **Seleção** (passo 3): `selecao_torneio` — torneio de 4; 2º pai sem repetir o 1º.
- **Recombinação** (passo 4): `cruzamento` — 1 ponto de corte, gera 2 filhos.
- **Mutação** (passo 5): `mutacao` — 1%, troca a prescrição de 2 talhões.
- **Substituição** (passo 6): laço principal — 10 elite + restante por torneio.

## Observação sobre os resultados (proposta de melhoria)
O próprio artigo concluiu que, para o AG, "uma revisão aprofundada dos métodos
implementados se faz necessária" — esta versão é justamente essa revisão. A
implementação revisada supera o AG do artigo em **todos** os pontos de parada,
com o maior ganho justamente quando há pouco orçamento computacional:

| Cálculos | Eficiência (artigo) | Eficiência (revisado) |
|---:|---:|---:|
| 5 000  | 0.204 | **0.750** |
| 10 000 | 0.417 | **0.883** |
| 25 000 | 0.674 | **0.932** |
| 50 000 | 0.801 | **0.939** |

(15 execuções, sementes 0–14; referência B&B = R\$ 32 170 883.)
