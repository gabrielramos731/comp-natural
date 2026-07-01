# Revisão técnica, verificação e proposta de melhoria — AG para Planejamento Florestal

> Documento de apoio ao trabalho final. Registra **o que existia**, **o que foi feito**,
> **o que melhorou (com causas e consequências)**, a **verificação de correção** e a
> **proposta de melhoria de fato**. Escrito para ser honesto sobre o que é ganho real
> e o que é apenas reimplementação correta.

---

## 1. O problema (contexto)

Reproduz o Algoritmo Genético (AG) da seção 3.5 do artigo *"Utilização de Heurísticas
para Solução de um Problema de Planejamento Florestal"* (Silva & Rocha, IFNMG), sobre a
base `Base120.xlsx` de JUNIOR et al. (2021).

- **Instância:** 120 talhões × 81 prescrições, horizonte de 16 anos (9 720 linhas).
- **Decisão:** escolher **uma** prescrição inteira por talhão (Eq. 2 e 5).
- **Objetivo (Eq. 6):** maximizar o VPL total, **penalizando** em `Pe = R$ 500` cada m³ de
  produção anual fora de `[Dmin, Dmax] = [140 000, 160 000] m³`:

  ```
  max  Σ_i VPL[i, x_i]  −  Pe · Σ_k d_k
  ```
  onde `d_k` é o volume absoluto desviado dos limites no ano `k`.

- **Referências (do artigo):** limite superior relaxado = R$ 36 334 070; VPL de
  referência (branch-and-bound) = **R$ 32 170 883**. Eficiência = `VPL / VPL_ref`.
- **No artigo, o AG foi o pior método:** eficiência de **0,204** (5 000 cálculos) a
  **0,801** (50 000). Os próprios autores concluíram que *"uma revisão aprofundada dos
  métodos implementados se faz necessária"*. **Este é o gancho do trabalho.**

---

## 2. O que existia (código original)

Três arquivos:

| Arquivo | Papel |
|---|---|
| `ag_florestal.py` | AG completo (uma execução). |
| `experimento_ag.py` | Roda o AG 15× em cada ponto de parada e imprime a tabela (estilo Tabela 2). |
| `Base120.xlsx` | Base de dados. |

O AG já era uma implementação **fiel** ao artigo:
inicialização aleatória, avaliação pela Eq. 6, seleção por torneio (4), recombinação de
1 ponto (2 filhos), mutação por troca (1%) e substituição com elitismo (10 melhores +
torneio). O critério de parada é o **número de cálculos da função objetivo**.

**Limitações do código original:**
1. **Carga lenta:** `carregar_dados` montava as matrizes com `df.iterrows()` linha a
   linha (≈ 9 720 iterações Python) — ~2,5 s só no laço, além do `read_excel`.
2. **Sem histórico de convergência:** a função só devolvia o VPL final, impossibilitando
   traçar curvas de evolução e derivar múltiplos pontos de parada de uma só execução.
3. **Sem geração de figuras** para a apresentação.

---

## 3. O que foi feito

### 3.1. `ag_florestal.py`
- **Carga vetorizada.** O laço `iterrows` foi substituído por ordenação
  `sort_values(["talhao","prescricao"])` + `reshape` direto para `VPL (120×81)` e
  `VOL (120×81×16)`. Incluí uma guarda `len(df) == n_t·n_p` que aborta se a base não for
  regular.
  - **Causa:** o antigo laço era o gargalo de tempo e não agregava robustez real.
  - **Consequência:** o custo de montagem das matrizes caiu de ~2,5 s para praticamente
    zero (o tempo restante é o `read_excel`, ~1,6 s). **Nenhuma mudança nos valores** —
    verificado como idêntico ao método antigo (ver §5). A guarda torna explícita a
    premissa de base regular.
- **Registro de histórico.** `algoritmo_genetico(..., registrar_historico=True)` passa a
  devolver `historico` = lista de pares `(nº_de_cálculos, melhor_VPL_até_agora)` após cada
  geração.
  - **Consequência (API):** a função agora retorna **4 valores**
    `(melhor_individuo, melhor_fitness, n_calculos, historico)`. Todos os chamadores foram
    atualizados. Custo de memória: ~1 250 floats por execução (desprezível).

### 3.2. `experimento_ag.py`
- Ajustado para a nova assinatura de 4 valores. Comportamento idêntico.

### 3.3. `graficos.py` (novo)
Roda 15 execuções **até 50 000 cálculos registrando o histórico** e, a partir dele,
deriva os 4 pontos de parada e gera as figuras dos slides (300 dpi, em português).

> **Nota metodológica importante:** como o AG só depende de `MAX_CALCULOS` para *decidir
> quando parar* (e nada mais), o melhor-VPL-até-agora em 5 000 cálculos dentro de uma
> execução de 50 000 é **exatamente igual** ao resultado de uma execução que pararia em
> 5 000 (mesma semente). Isso foi verificado numericamente (§5). Ou seja: **uma execução
> por semente fornece os 4 pontos, de forma exata** — mais rápido que o artigo (que rodava
> cada ponto separado) e sem introduzir viés.

### 3.4. `README.md`
Documentação dos scripts, das figuras e da tabela de resultados atualizada.

---

## 4. Resultados

15 execuções, sementes 0–14, `VPL_ref = 32 170 883`.

| Cálculos | Média (R$) | Máximo (R$) | Mínimo (R$) | Eficiência (revisado) | Eficiência (artigo) |
|---:|---:|---:|---:|---:|---:|
| 5 000  | 24 119 493 | 29 352 411 | 18 083 160 | **0,750** | 0,204 |
| 10 000 | 28 403 237 | 30 218 182 | 23 055 490 | **0,883** | 0,417 |
| 25 000 | 29 987 156 | 30 338 064 | 28 846 510 | **0,932** | 0,674 |
| 50 000 | 30 204 245 | 30 440 853 | 29 217 413 | **0,939** | 0,801 |

**Figuras geradas** (`figuras/`):

| Arquivo | Conteúdo |
|---|---|
| `fig1_convergencia.png` | VPL médio ± desvio vs nº de cálculos, com VPL de referência e limite superior. |
| `fig2_comparacao_efic.png` | **Gráfico-chave:** barras de eficiência artigo vs. revisado nos 4 pontos. |
| `fig3_boxplot.png` | Distribuição das 15 execuções por ponto, com a média do AG do artigo marcada. |
| `fig4_vpl_maximo.png` | Melhor VPL vs. esforço computacional (artigo vs. revisado). |
| `resultados_ag.csv` | Tabela acima em CSV. |

---

## 5. Verificação de correção

Bateria de **20 checagens independentes** (script separado, ground-truth alternativo).
Todas passaram:

- **Base regular:** 9 720 = 120 × 81; todo talhão tem 81 prescrições únicas (1..81).
- **Carga vetorizada == ground truth:** `VPL` idêntico ao `pivot` do pandas; `VOL` idêntico
  a 200 lookups diretos aleatórios. (Confirma que a otimização de §3.1 não alterou dados.)
- **Função objetivo == Eq. 6:** `avaliar()` recomputada "na mão" bate com diferença < 1e-6;
  o contador incrementa exatamente 1 por avaliação.
- **Contagem de cálculos legítima:** 40 (população inicial) + 40 por geração; a grade do
  histórico começa em 40, anda de 40 em 40 e **contém exatamente** 5 000, 10 000, 25 000 e
  50 000 (sem arredondamento).
- **Monotonicidade:** o melhor-até-agora é não-decrescente.
- **Determinismo:** mesma semente → mesmo VPL; sementes diferentes → VPLs diferentes.
- **Extração de ponto exata:** valor em 5 000 tirado do histórico de uma execução de
  50 000 == valor de uma execução limitada a 5 000 (5 sementes, igualdade exata).

Conclusão: **a ideia e a execução estão corretas**, e a geração dos gráficos usa dados
válidos e um método de amostragem sem viés.

---

## 6. O que realmente melhorou — causas e consequências (análise honesta)

Aqui é preciso separar dois tipos de "melhoria":

### 6.1. Melhorias de engenharia (inequívocas)
- Carga ~vetorizada (tempo de montagem ~2,5 s → ~0).
- Histórico de convergência + geração automática de figuras.
- Uma execução por semente rende os 4 pontos de parada, exatamente.

Essas são ganhos reais de **ferramental e reprodutibilidade**, não de qualidade da solução.

### 6.2. A grande diferença de eficiência vs. o artigo — investigação
O ponto delicado: o AG **revisado** obtém 0,750→0,939, contra 0,204→0,801 do artigo. De
onde vem essa diferença tão grande, se o algoritmo é o *mesmo* descrito na seção 3.5?

Investiguei a hipótese mais provável de diferença de projeto — a **inicialização**. O
artigo descreve gerar **uma** solução-base (construtiva α=1, i.e. aleatória) e produzir os
40 indivíduos por **embaralhamento** dessa base (todos compartilham o mesmo multiconjunto
de prescrições). O código atual gera **40 indivíduos independentes**. Comparei as duas
(8 execuções):

| Inicialização | Efic. @ 5 000 | Efic. @ 50 000 |
|---|---:|---:|
| Independente (atual) | 0,770 | 0,942 |
| Embaralhar-base (literal do artigo) | 0,620 | 0,946 |

**Leitura honesta dos números:**
- A inicialização **explica parte da vantagem em orçamento baixo** (0,77 vs 0,62 em 5 000):
  indivíduos independentes dão mais diversidade e melhores blocos construtivos cedo.
- Mas em 50 000 **as duas convergem para ~0,94** — e, crucialmente, **mesmo reproduzindo a
  inicialização do artigo, a nossa implementação limpa atinge 0,946**, muito acima dos
  0,801 relatados por eles.

**Conclusão:** a diferença dramática vs. o artigo **não** é explicada pelas diferenças de
projeto que conseguimos identificar. O mais provável é que o código original deles tivesse
um **defeito de implementação** (contagem de cálculos, operador mal aplicado, avaliação da
penalidade, etc.) — exatamente o que os autores admitiram ao pedir "uma revisão aprofundada
dos métodos implementados".

> **Portanto, o que hoje chamamos de "AG revisado" é, com honestidade, uma
> _reimplementação correta e fiel_ do próprio AG do artigo — a "revisão aprofundada" que
> eles pediram. É um resultado forte e apresentável, mas _não é um algoritmo novo_.** Para
> uma **melhoria algorítmica de fato**, ver a §7.

---

## 7. Proposta de melhoria **de fato**

Meta: ir além de corrigir o AG e **superar o melhor resultado global do artigo** — que não
foi o AG, e sim o **Simulated Annealing (vizinhança aleatória, 0,958)** e a **Busca Local
sistemática + primeira-melhora (0,954)**. Hoje o AG revisado está em 0,939; o alvo é
≥ 0,95, idealmente batendo o SA.

As propostas abaixo são concretas, justificadas pelo comportamento observado e diretamente
implementáveis sobre `ag_florestal.py`.

### P1 — AG **memético** (hibridizar com Busca Local) — *maior potencial*
A cada geração, aplicar uma **busca local de vizinhança sistemática (primeira-melhora)** ao
melhor indivíduo (ou aos `N_ELITE`), gastando um pequeno orçamento de avaliações antes de
seguir. O AG fornece **diversificação** (exploração global) e a BL fornece
**intensificação** — e a BL sistemática-pm foi justamente o melhor método não-SA do artigo.
- *Justificativa:* o AG puro estagna perto de 0,94; a intensificação local tende a "polir"
  a melhor solução em direção ao ótimo, como o SA/BL fazem.
- *Cuidado:* balancear o orçamento gasto na BL para não reduzir demais as gerações.

### P2 — Inicialização **gulosa-aleatória** (tipo GRASP, α ≈ 0,5) — *ganho em orçamento baixo*
Em vez de puramente aleatória, construir cada indivíduo escolhendo, por talhão, uma
prescrição de uma **lista restrita de candidatos** (as de maior VPL, limiar µ da Eq. 7 do
artigo com α ≈ 0,5).
- *Justificativa:* hoje estamos em **0,750 em 5 000 cálculos** — o ponto mais fraco.
  Começar de soluções já boas melhora fortemente o regime de baixo orçamento (medido em §6:
  a inicialização domina a fase inicial).

### P3 — Mutação mais expressiva / **adaptativa**
A mutação atual é uma **troca** (swap) entre dois talhões — preserva o multiconjunto de
prescrições e **não introduz material genético novo**. Propor:
- uma mutação de **reatribuição** (dar a um talhão uma prescrição aleatória nova); e/ou
- **taxa adaptativa**: aumentar a mutação quando a população estagna (queda de diversidade),
  reduzir quando está evoluindo.
- *Justificativa:* combate a convergência prematura e amplia o alcance do espaço de busca.

### P4 — **Crossover uniforme** por talhão (no lugar do 1-ponto)
Os talhões são **independentes** (não há estrutura sequencial), então o corte de 1 ponto
impõe um viés posicional artificial. O crossover uniforme (cada gene vindo de um dos pais
com prob. 0,5) mistura melhor os blocos.

### P5 — Vizinhança **guiada pela restrição** (mutação localizada)
Direcionar parte das mutações para o **ano de maior desvio volumétrico** (a "vizinhança
localizada" do artigo), trocando prescrições de talhões que colhem naquele ano — reduzindo
a penalidade de forma dirigida.

**Plano de avaliação sugerido:** implementar P1 + P2 primeiro (maior retorno esperado),
rodar o mesmo protocolo (15 execuções × 4 pontos), e **adicionar as linhas do SA e da BL do
artigo** nas figuras 2/4 para mostrar se o AG memético alcança/supera 0,95. As demais
(P3–P5) entram como estudo de ablação.

---

## 8. Resumo para os slides

- **Problema:** escolher 1 de 81 manejos para cada um dos 120 talhões, maximizando o VPL
  com penalidade por produção anual fora de `[140k, 160k] m³`.
- **O que fizemos:** revisamos e reimplementamos corretamente o AG que o artigo apontou como
  deficiente; instrumentamos convergência e automatizamos as figuras.
- **Resultado:** eficiência sobe de **0,204–0,801** (AG do artigo) para **0,750–0,939**
  (AG revisado), com verificação formal de correção.
- **Honestidade:** a revisão é uma implementação correta do mesmo AG — o ganho vem de
  corrigir a implementação (a fase inicial também se beneficia de inicialização
  independente). Uma **melhoria algorítmica de fato** (AG memético + init gulosa-aleatória)
  é a proposta para superar também o SA (0,958), o melhor método do artigo.

---

*Arquivos: `ag_florestal.py`, `experimento_ag.py`, `graficos.py`, `figuras/`, `README.md`.
Reproduzível com `python graficos.py` (matplotlib, numpy, pandas, openpyxl).*
