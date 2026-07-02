# AG - Planejamento Florestal (Base120)

Implementação em Python do **Algoritmo Genético** descrito na seção 3.5 do
artigo, para o problema de planejamento florestal: escolher uma prescrição
para cada um dos 120 talhões maximizando o VPL e penalizando a produção anual
de madeira fora dos limites `[140000, 160000] m³` (Eq. 6).

## Arquivos
- `ag_florestal.py` — implementação do AG (uma execução), com **duas estratégias de inicialização da população**.
- `experimento_ag.py` — roda o AG 15× em cada ponto de parada e gera a tabela de resultados (estilo Tabela 2).
- `experimento_comparacao.py` — **experimento fatorial 2×2** (inicialização {aleatória, GRASP} × busca local {sem, com}), salva resultados em `resultados/` e registra em `EXPERIMENTOS.md`.
- `graficos.py` — gera os **gráficos e tabelas** de comparação a partir dos resultados (`resultados/figuras/`).
- `Base120.xlsx` — base de dados (120 talhões × 81 prescrições × 16 anos).

## Como executar
```bash
pip install pandas numpy openpyxl matplotlib
python ag_florestal.py           # uma execução, mostra a evolução e o melhor VPL
python experimento_ag.py         # reproduz a tabela (15 execuções × 4 pontos)
python experimento_comparacao.py # experimento fatorial 2×2 (15 execuções cada)
python graficos.py               # gera figuras e tabelas em resultados/figuras/
```

## Comparação de estratégias (foco do estudo)
Três formas de gerar/refinar a população do AG, mantendo **todo o restante idêntico**:

| Estratégia | O que muda | Parâmetro |
|---|---|---|
| **Aleatória** | população inicial completamente aleatória (construtiva α=1) | `ESTRATEGIA_INICIAL = "aleatoria"` |
| **GRASP (gulosa-aleatória)** | população inicial via RCL por talhão, Eq. 7 (`µ = VPLmax − α·(VPLmax − VPLmin)`); sorteia entre prescrições de VPL ≥ µ | `ESTRATEGIA_INICIAL = "grasp"`, `ALPHA_GRASP = 0.5` |
| **AG Memético (BL)** | init aleatória **+ Busca Local sistemática (primeira-melhora)** aplicada aos melhores indivíduos a cada geração (AG diversifica, BL intensifica) | `MEMETICO = True`, `ORCAMENTO_BL = 120`, `N_ELITE_BL = 1` |

As 4 configurações (2×2) são: **Aleatória**, **Aleatória + BL**, **GRASP (α=0.5)** e **GRASP (α=0.5) + BL**.
`experimento_comparacao.py` executa cada uma 15× até 50000 cálculos, lê os pontos de parada
(5000/10000/25000/50000) do histórico e grava:
- `resultados/resumo_comparacao.csv` — tabela agregada (média/máx/mín/desvio/eficiência).
- `resultados/runs_finais.csv` — VPL de cada execução (boxplots/dispersão).
- `resultados/convergencia.csv` — curva média de VPL × nº de cálculos (evolução).
- `resultados/qualidade_inicial.csv` — qualidade da população inicial por estratégia.
- `resultados/experimentos.json` — dump completo; `EXPERIMENTOS.md` — log legível.

## Gráficos e tabelas (`graficos.py` → `resultados/figuras/`)
- `convergencia.png` — evolução do VPL/eficiência × nº de cálculos (visão completa + zoom).
- `barras_50000.png` — eficiência média das 4 configurações em 50000 cálculos.
- `efeito_busca_local.png` — efeito fatorial da BL (sem vs com) por inicialização.
- `boxplot_50000.png` — distribuição do VPL final (15 execuções) por configuração.
- `tabela_50000.png` — tabela-resumo (imagem) do ponto de 50000 cálculos.
- `resultados/tabela_comparacao.md` — tabela markdown com todos os pontos de parada.

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

## Observação sobre os resultados
Esta implementação limpa converge bem (eficiência ~0.95 com 50000 cálculos),
acima dos números relatados no artigo para o AG (~0.80). O próprio artigo
concluiu que "uma revisão aprofundada dos métodos implementados se faz
necessária" — esta versão pode ser apresentada como essa revisão.
