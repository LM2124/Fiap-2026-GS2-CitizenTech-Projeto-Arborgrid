# Assistente ArborGrid: Generative AI for Engineering (`llm-assistant/`)

Assistente técnico local fundamentado em legislação municipal paulistana, normas de segurança elétrica/arborização e telemetria real de inspeções GeoSampa, desenvolvido com **`Qwen/Qwen3-1.7B`**, **Busca Híbrida (RAG)**, **Fine-Tuning LoRA (`r=16`)** e **Ferramenta Determinística (Agente)**.

---

## 1. Integrantes e Ficha Técnica no Ecossistema ArborGrid SP

| RM | Integrante |
| :---: | :--- |
| `551117` | Lorenzo Gomes Andreata |
| `97158` | Lucas Moreno Matheus |
| `99756` | Kayky Oliveira Schunck |

- **Tarefa da IA:** triar dúvidas de cidadãos e operadores sobre arborização urbana e rede elétrica em São Paulo, devolvendo **JSON estrito de 7 chaves** com nível de risco, órgão competente, telefone oficial, proibição de poda própria, orientação segura sem dados pessoais (LGPD), citação exata de fontes recuperadas e recusa segura (`nao_consta`) fora do escopo.
- **Conexão com o MVP e API:** integrado ao [**ArborGrid SP**](../README.md) por meio de [`webapp/app/services/assistant_service.py`](../webapp/app/services/assistant_service.py) e dos endpoints **`POST /api/v1/assistant/ask`** e **`GET /api/v1/assistant/status`**, consumindo registros de [`manifest.json`](../arborgrid-vision/dataset/sp_geosampa_api/manifest.json) e atendendo a aba **Assistente** ([`CopilotView.tsx`](../webapp/frontend/src/views/CopilotView.tsx)) via `llama-server` (`Qwen 1.7B Q4_K_M`, `COMPOSE_PROFILES=llm`) com *fallback* RAG determinístico.

---

## 2. Arquitetura do Pipeline (RAG + Agente + LoRA + Validador)

> *Detalhamento completo dos módulos ([`src/retrieval.py`](src/retrieval.py), [`src/tools.py`](src/tools.py), [`src/validator.py`](src/validator.py)), regras de decisão e comparativo entre `Base Zero-Shot`, `RAG Simples` e `Fine-tuning LoRA + RAG + Agente` em **[`ARQUITETURA_DO_SISTEMA.md`](ARQUITETURA_DO_SISTEMA.md)**.*

```mermaid
flowchart LR
    U["Pergunta do Cidadão / Operador"] --> T["ArborGridInspector (src/tools.py)"]
    U --> R["KnowledgeRetriever — MiniLM-L12-v2 + TF-IDF (limiar = 0.30, top-3)"]
    T -->|"Detecta GeoSampa ID"| M["manifest.json (triagem v3 + px contato/vento)"]
    T -->|"Sem ID específico"| Ctx["[CONTEXTO] top-3 trechos"]
    M --> CtxA["[CONTEXTO] + [DADOS_TECNICOS_ARBORGRID]"]
    R --> Ctx
    R --> CtxA
    Ctx --> LLM["Qwen/Qwen3-1.7B + LoRA r=16 (enable_thinking=False)"]
    CtxA --> LLM
    LLM --> V["Validador Programático (src/validator.py)"]
    V --> Out["JSON Estrito (7 chaves) + fontes verificadas + LGPD"]
```

### Contrato Programático de Saída (JSON de 7 Chaves)

```json
{
  "status": "respondido | nao_consta",
  "nivel_risco": "emergencia_imediata | urgente_vistoria | preventivo_rotina | fora_de_escopo",
  "orgao_responsavel": "Órgão competente ou não consta",
  "telefone_ou_canal": "Telefone/canal oficial ou não consta",
  "pode_podar_por_conta": false,
  "acao_imediata": "Orientação segura fundamentada sem dados pessoais (LGPD)",
  "fontes": ["CHUNK-ID-1", "CHUNK-ID-2"]
}
```

---

## 3. Tabela Completa da Base de Conhecimento Curada (`base/`)

Os **16 trechos curados** em [`base/knowledge_base.jsonl`](base/knowledge_base.jsonl) estão documentados com URL, formato e licença em [`base/fontes_licencas.md`](base/fontes_licencas.md):

| `chunk_id` | Categoria | Título / Assunto | Fonte Oficial |
| :--- | :--- | :--- | :--- |
| `LEI-17794-ART05` | `legislacao_sp` | Competência sobre árvores em áreas públicas de SP | Lei Municipal SP nº 17.794/2022 |
| `LEI-17794-ART18` | `legislacao_sp` | Manejo de árvore em imóvel particular (10 dias úteis + ART) | Lei Municipal SP nº 17.794/2022, Art. 18 |
| `DEC-61859-PRZ` | `legislacao_sp` | Regulamentação e abertura de protocolos municipais | Decreto Municipal SP nº 61.859/2022 |
| `SP156-ROTINA` | `atendimento_municipal` | Solicitação preventiva de poda/vistoria sem faíscas | Portal SP156 — Prefeitura de São Paulo |
| `ENEL-EMERG-01` | `emergencia_eletrica` | Emergência elétrica (`0800 72 72 196`) com árvore e fiação | Enel Distribuição SP — Emergência 24h |
| `ENEL-RELAC-02` | `atendimento_concessionaria` | Atendimento comercial (`0800 72 72 120`) não emergencial | Enel Distribuição SP — Relacionamento |
| `BOMBEIROS-193` | `emergencia_vida` | Risco imediato à vida, incêndio ou preso em veículo (`193`) | Corpo de Bombeiros PMESP |
| `DEFESA-CIVIL-199` | `emergencia_climatica` | Tempestade, árvore caída obstruindo via (`199`) | Defesa Civil Municipal de São Paulo |
| `SEGURANCA-FIACAO` | `seguranca_operacional` | Conduta diante de fio caído ou veículo atingido | Manual de Segurança — Defesa Civil / Enel SP |
| `SVMA-MANUAL-PODA` | `norma_tecnica` | Prioridade à rede compacta/isolada antes de poda drástica | Manual Técnico de Poda — PMSP / SVMA |
| `NBR-16246-1` | `norma_tecnica` | Critérios técnicos de poda em arborização urbana | ABNT NBR 16246-1 |
| `NBR-16246-2` | `seguranca_trabalho` | Segurança do trabalho e NR-10 próximo à rede elétrica | ABNT NBR 16246-2 e NR-10 |
| `NBR-16246-3` | `norma_tecnica` | Avaliação objetiva de risco de queda arbórea | ABNT NBR 16246-3 |
| `ARBORGRID-TRIAGEM` | `sistema_arborgrid` | Triagem `CRITICO`, `DIVERGENCIA_PARALAXE`, `PROXIMIDADE`, `SEGURO` | Especificação ArborGrid-Vision v3 |
| `ARBORGRID-LGPD` | `sistema_arborgrid` | Privacidade, minimização e anonimização de solicitações | Diretriz LGPD (Lei nº 13.709/2018) |
| `CHANGELOG-2026-ARBOR` | `regra_operacional` | Desempate operacional: rotina SP156 vs. emergência Enel/193 | Nota Operacional Integrada ArborGrid v1.2 |

---

## 4. Desempenho da Busca Híbrida (RAG) e Escolha do Limiar

A recuperação combina similaridade densa (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) com reforço lexical TF-IDF (`src/retrieval.py`), avaliada sobre os 30 casos de [`avaliacao/test_set.jsonl`](avaliacao/test_set.jsonl) (22 respondíveis + 8 casos `"nao_consta"`). A varredura completa dos pontos testados está salva em [`avaliacao/retrieval_threshold_sweep.csv`](avaliacao/retrieval_threshold_sweep.csv).

![Varredura de Limiar de Similaridade Cosseno](assets/retrieval_threshold_sweep.png)

### Decisão: Limiar de Similaridade em `0,30`
- Em triagem de risco elétrico e proteção civil, falsos negativos (descartar perguntas legítimas do cidadão) têm custo crítico. Pela curva acima, **`0,30`** é o ponto ótimo que zera a recusa indevida (**`false_refusal_rate = 0,0000`**) mantendo **`Recall@1 = 0,6818`**, **`Recall@3 = 0,9545`** e **`MRR@3 = 0,8030`**.
- Assim, consultas totalmente fora de domínio são filtradas na própria busca (ex.: previsão do tempo em Pinheiros, `score = 0,2393`), enquanto armadilhas com vocabulário próximo (ex.: taxa de boleto, valor de ART ou outro município) seguem para o LLM fundamentado recusá-las com segurança.

### Retriever em produção (`webapp`)
No MVP ([`webapp/app/services/assistant_service.py`](../webapp/app/services/assistant_service.py)), para manter a VM de `4 GB` livre para o LLM, a aplicação não carrega `sentence-transformers`: utiliza um *retriever* lexical leve baseado em TF-IDF com dicionário de sinônimos em português e limiar **`0,12`**. Avaliado sobre os mesmos `30` casos de teste:

| Retriever | Limiar | `Recall@1` | `Recall@3` | Recusa indevida (`22` respondíveis) | Bloqueio na busca (`8` `nao_consta`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Híbrido (`MiniLM-L12-v2 + TF-IDF`, `src/retrieval.py`)** | `0,30` | `0,6818` | `0,9545` | `0/22` (`0,0000`) | — |
| **Produção (`TF-IDF + sinônimos PT`, `webapp/`)** | `0,12` | **`0,909`** (`20/22`) | **`0,955`** (`21/22`) | `0/22` | `4/8` *(restante recusado pelo modelo)* |

> *Como o dicionário de sinônimos foi ajustado manualmente durante o desenvolvimento, estes números servem como verificação de consistência e não substituem a avaliação principal.*

---

## 5. Especialização LoRA (`Qwen 1.7B`), Placar das 3 Configurações e Decisão Arquitetural

### Configuração de Treinamento (`scripts/train_lora.py`)
- **Base escolhida:** `Qwen/Qwen3-1.7B` (**Qwen 1.7B**), treinado em precisão original (`bfloat16`) com `enable_thinking=False` na GPU `NVIDIA GeForce RTX 3060 Ti` (`8 GB VRAM`).
- **Dataset SFT sintético (`360` treino / `60` validação):** amostras em `data/sft_train.jsonl` e `data/sft_val.jsonl`, geradas a partir de *templates* sintéticos majoritariamente com `chunk_id`s `SYN-*` (e os trechos reais `ARBORGRID-TRIAGEM` e `SP156-ROTINA` em `70` amostras cada) para ensinar formato JSON, leitura de `[DADOS_TECNICOS_ARBORGRID]`, recusa fundamentada e sanitização LGPD.
- **LoRA:** `r = 16`, `lora_alpha = 32`, `lora_dropout = 0,05` em todas as projeções (`q/k/v/o/gate/up/down_proj` — `17.432.576` parâmetros treináveis, `1,0030%` do modelo), 3 épocas, `MAX_LENGTH = 1024`, máscara de perda `-100` sobre o prompt.

#### Auditoria de contaminação treino × teste
- **Sobreposição de perguntas:** `8` das `30` perguntas de teste possuem paráfrase próxima nos *templates* do SFT (similaridade Jaccard de *tokens* `≥ 0,43`): `T25`, `T29`, `T27`, `T08`, `T14` (`≥ 0,50`) e `T12`, `T09`, `T23` (`≈ 0,43`). Quatro delas são casos `nao_consta` (`T23`, `T25`, `T27`, `T29`).
- **Sobreposição de `chunk_id`s:** os *chunks* reais `ARBORGRID-TRIAGEM` e `SP156-ROTINA` aparecem em `70` amostras do SFT cada (as demais utilizam IDs `SYN-*`).
- **Placar restrito aos `22` casos de teste sem sobreposição (acerto estrito):**

  | Configuração | Total (`30`) | Com paráfrase no SFT (`8`) | **Sem sobreposição (`22`)** |
  | :--- | :---: | :---: | :---: |
  | Base sem contexto | `8/30` (`26,7%`) | `4/8` | **`4/22` (`18,2%`)** |
  | RAG simples | `10/30` (`33,3%`) | `4/8` | **`6/22` (`27,3%`)** |
  | LoRA + RAG + Agente + Validador | `28/30` (`93,3%`) | `8/8` | **`20/22` (`90,9%`)** |

- **Conclusão:** o ganho se mantém no subconjunto limpo (`18,2% → 27,3% → 90,9%`), logo o resultado não é explicado por memorização; a acurácia de `100%` nas recusas, contudo, é parcialmente auxiliada pelos `4` *templates* de recusa vistos no treino.

![Curva de Aprendizado (Qwen3-1.7B + LoRA r=16)](assets/curva_loss_lora.png)

- **Convergência observada:** a perda de validação (`eval_loss`) caiu continuamente de `0,1330` (época 1, passo 23) para `0,0021` (época 2, passo 46) e **`0,00055`** (época 3, passo 69) em `2776,1 s`. Uma perda de validação próxima de zero em um *split* gerado pelos mesmos *templates* mostra que o formato foi aprendido, não que o modelo generaliza; a evidência de generalização está no placar do subconjunto limpo acima (`20/22 = 90,9%`).

### Placar Comparativo das Arquiteturas (`30` casos)

Comparação programática sobre [`avaliacao/test_set.jsonl`](avaliacao/test_set.jsonl) — incluindo conformidade de *schema* JSON, risco, canal competente, segurança de poda, citação de fonte, ausência de alucinação de telefone e acerto estrito:

![Placar Comparativo das 3 Configurações](assets/placar_comparativo.png)

> *Variação completa com todos os 11 quesitos avaliados: [`assets/placar_comparativo_completo.png`](assets/placar_comparativo_completo.png) | Dados brutos: [`avaliacao/placar_comparativo.csv`](avaliacao/placar_comparativo.csv) | Execução detalhada em [`03_avaliacao_comparativa_e_agente.ipynb`](notebooks/03_avaliacao_comparativa_e_agente.ipynb).*

#### Ablação do validador (reprocessamento offline das saídas salvas)
Como a terceira configuração em [`scripts/run_benchmarks.py`](scripts/run_benchmarks.py) altera quatro fatores simultaneamente (adaptador LoRA, ferramenta GeoSampa, injeção de `ARBORGRID-TRIAGEM`/`SP156-ROTINA` e `apply_post_generation_filters`, aplicado apenas à configuração 3), o validador foi reaplicado *offline* sobre as saídas brutas salvas da configuração 2:

| Configuração | Acerto estrito |
| :--- | :---: |
| RAG simples | `10/30` (`33,3%`) |
| RAG simples + Validador (sem LoRA, sem agente) | `12/30` (`40,0%`) |
| LoRA + RAG + Agente + Validador | `28/30` (`93,3%`) |

O validador isolado explica **`+2` casos**; os **`+16` casos** restantes vêm do adaptador LoRA em conjunto com a ferramenta GeoSampa (separar esses dois fatores exigiria uma quarta execução em GPU, listada como próximo passo).

### Decisão Arquitetural: Adoção de `LoRA + RAG + Agente`
1. **Por que descartar o Base zero-shot (`26,7%`):** apesar de respeitar o contrato JSON (`100,0%` de *schema* válido), o **Qwen 1.7B** sem contexto fabrica telefones em **`23,3%`** dos casos, acerta apenas `40,0%` dos órgãos competentes e viola segurança operacional sugerindo poda própria em **`73,3%`** das respostas (`seguranca_poda = 26,7%`).
2. **Por que descartar apenas RAG sem LoRA (`33,3%`):** a injeção dos trechos resolve grande parte dos canais oficiais (`86,7%`) e derruba telefones fabricados para `3,3%`, mas sem o ajuste fino o modelo ainda erra `46,7%` em segurança de poda e omite citações em `36,4%` das respostas.
3. **Decisão fechada — Sistema Completo (`93,3%`, `28/30` casos):** a união **`LoRA + RAG + Agente GeoSampa + Validador Programático`** atinge **`100,0%`** em validade JSON/Schema, órgão/canal competente, segurança de poda, citação da fonte esperada, recusa de `"nao_consta"` e conformidade LGPD (**`+66,6 p.p.`** sobre o Base, **`+60,0 p.p.`** sobre o RAG simples e **`+53,3 p.p.`** sobre RAG + Validador).

- **Leitura estatística (`n = 30`, intervalos de Wilson 95%):** `Base` `8/30 = 26,7%` `[14,2%; 44,4%]`; `RAG simples` `10/30 = 33,3%` `[19,2%; 51,2%]`; `Sistema Completo` `28/30 = 93,3%` `[78,7%; 98,2%]`. Como os intervalos não se sobrepõem, a transição `Base/RAG → Sistema Completo` representa uma diferença estatisticamente robusta *(demais intervalos comparativos na Seção 6)*.

---

## 6. Resumo da Exploração de Modelos, Quantização e Decisão para Nuvem

Conduzimos um estudo em 3 etapas comparando famílias/portes (**`SmolLM2-360M` → `Qwen 0.6B` → `Qwen 1.7B`**) e quantização (**`16 bits` original → `4 bits` quantizado**) em **CPU pura (`2 threads`)**; o estudo completo consta em **[`EXPLORACAO_MODELOS_E_QUANTIZACAO.md`](EXPLORACAO_MODELOS_E_QUANTIZACAO.md)** e no notebook [`04_exploracao_modelos_e_quantizacao.ipynb`](notebooks/04_exploracao_modelos_e_quantizacao.ipynb).

![Modelos e Quantização em CPU (2 threads): SmolLM2-360M vs Qwen 0.6B vs Qwen 1.7B](assets/comparativo_smol360m_vs_qwen.png)

| Modelo + Precisão (Hardware) | Acerto Geral Estrito | Schema Válido | Memória Base (VRAM / RAM) | Disco | Latência Média (Local) | Observação de Engenharia |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`SmolLM2-360M`** (16b GPU) | 20,0% (`6/30`) | 90,0% | `756,4 MiB` | — | `10,44 s` | **Descartado:** pior qualidade e maior latência (CPU `Q4_K_M`: `10,0%`, `5,16 s`). |
| **`Qwen 0.6B`** (16b GPU) | 86,7% (`26/30`) | 100,0% | `1.590,6 MiB` | — | `8,47 s` | Validou a família **Qwen** (4,3× mais acurácia que SmolLM2). |
| **`Qwen 0.6B`** (`Q4_K_M` CPU 2t) | 66,7% (`20/30`) | 100,0% | `896,8 MiB` | `378,3 MiB` | `4,97 s` (`29,3 tok/s`) | Opção ultraleve, porém perde `-20,0 p.p.` ao quantizar para 4 bits. |
| **`Qwen 1.7B`** (16b GPU) | **93,3%** (`28/30`) | 100,0% | `4.172,0 MiB` | — | `9,20 s` | Teto de confiabilidade (`100%` em canal, poda, citação e recusa). |
| **`Qwen 1.7B`** (**`Q4_K_M` CPU 2t**) | **93,3%** (`28/30`) | 100,0% | **`2.015,0 MiB`** | **`1.056,1 MiB`** | **`9,30 s`** (`17,3 tok/s`) | **Escolhido para produção:** mantém `28/30` (`93,3%`) sem GPU, sem diferença mensurável em `n = 30` (`28/30` nas duas precisões). |

- **Leitura estatística (`n = 30`, intervalos de Wilson 95%):** `28/30 = 93,3%` **`[78,7%; 98,2%]`** (`Qwen 1.7B` 16b e `Q4_K_M`); `26/30 = 86,7%` **`[70,3%; 94,7%]`** (`Qwen 0.6B` 16b); `20/30 = 66,7%` **`[48,8%; 80,8%]`** (`Qwen 0.6B Q4_K_M`); `10/30 = 33,3%` **`[19,2%; 51,2%]`** (`RAG simples`); `8/30 = 26,7%` **`[14,2%; 44,4%]`** (`Base`). A transição `Base/RAG → Sistema Completo` é uma diferença robusta (os intervalos não se sobrepõem), enquanto `Qwen 0.6B × 1.7B` em `16 bits` (`26/30` × `28/30`) não é estatisticamente distinguível neste tamanho de amostra.
- **Decisão consolidada de produção (`Google Cloud Compute Engine e2-medium`):** como o modelo de Visão Computacional (`segformer_v3_int8_dyn_matmul.onnx`) roda no navegador via `onnxruntime-web`, a instância `e2-medium` (`4 GB` RAM) executa o **`Qwen 1.7B Q4_K_M`** via `llama-server` (`COMPOSE_PROFILES=llm`, `LLAMA_THREADS=2`) ao lado de `webapp/`, `nginx` e `prometheus`/`grafana`, mantendo *fallback* RAG determinístico caso o servidor LLM esteja desligado.

---

## 7. Auditoria de 5 Erros Reais, Causa Raiz e Proposta de Correção

A partir da auditoria automatizada em [`avaliacao/auditoria_erros_reais.md`](avaliacao/auditoria_erros_reais.md), consolidamos **5 erros reais** — os **2 desvios residuais** remanescentes no pipeline final (`3_rag_lora_agente`, `93,3%`) e **3 falhas representativas** dos estágios anteriores (`Base` e `RAG Simples`):

| # | Caso / Configuração | Erro Real Observado | Causa Raiz | Correção Implementada / Proposta |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **`T01`** (`1_base_sem_contexto`) | Alucinou o telefone falso **`"181 - 3000-1111"`** para faíscas na fiação e não citou fonte. | Ausência de contexto normativo no *prompt*; modelo recorreu à memória paramétrica ruidosa. | **Resolvido:** injeção de trechos via `KnowledgeRetriever` + validação rigorosa de números citados. |
| **2** | **`T08`** (`2_rag_simples`) | Retornou **`status="nao_consta"`** mesmo tendo `LEI-17794-ART05` e `SP156-ROTINA` recuperados. | Sem SFT/LoRA, o modelo não discriminava pergunta respondível de armadilha quando havia conflito ("Prefeitura demorando -> contratar jardineiro?"). | **Resolvido:** treino LoRA sobre 360 amostras ensinando disciplina de resposta vs. recusa estrita. |
| **3** | **`T22`** (`2_rag_simples`) | Classificou galho tocando fios **sem faíscas** como **`urgente_vistoria`** e direcionou ao `0800 72 72 196` em vez do `SP156`. | `SEGURANCA-FIACAO` apareceu no `top-3`, induzindo o modelo base a priorizar risco elétrico ativo sobre a regra de rotina. | **Resolvido:** especialização LoRA + regra de desempate em `CHANGELOG-2026-ARBOR` e ajuste de piso no `validator.py`. |
| **4** | **`T15`** (`3_rag_lora_agente` — *residual*) | Respondeu **`"SP156 (156)"`**, mas o avaliador marcou telefone não fundamentado pois o dígito literal `"156"` não estava em `ARBORGRID-TRIAGEM` dentro do `top-3`. | O trecho `ARBORGRID-TRIAGEM` mencionava `"SP156"`, sem explicitar o numeral `"156"` no próprio bloco recuperado. | **Proposta de correção:** explicitar `"telefone 156"` diretamente no texto do trecho `ARBORGRID-TRIAGEM` em [`base/knowledge_base.jsonl`](base/knowledge_base.jsonl). |
| **5** | **`T17`** (`3_rag_lora_agente` — *residual*) | Consultou a árvore `#629339` (`DIVERGENCIA_PARALAXE`), porém classificou **`preventivo_rotina`** em vez de **`urgente_vistoria`**. | O retorno `[DADOS_TECNICOS_ARBORGRID]` listava a triagem sem indicar explicitamente o nível de risco recomendado para divergência de paralaxe. | **Proposta de correção:** emitir campo explícito `prioridade_recomendada=urgente_vistoria` no retorno de [`src/tools.py`](src/tools.py) quando houver `DIVERGENCIA_PARALAXE`. |

---

## 8. Exemplos Reais de Pergunta e Resposta (`Qwen3-1.7B + LoRA + RAG + Agente`)

Cinco casos de [`avaliacao/test_set.jsonl`](avaliacao/test_set.jsonl), com o JSON exatamente como o sistema completo respondeu no placar ([`avaliacao/benchmark_detalhes.json`](avaliacao/benchmark_detalhes.json), configuração `3_rag_lora_agente`). Todos passaram na validação de *schema*.

### `T01`: Emergência elétrica

> Tem uma árvore na calçada encostando nos fios e soltando faíscas com cheiro de queimado agora. Quem devo chamar?

```json
{
  "status": "respondido",
  "nivel_risco": "emergencia_imediata",
  "orgao_responsavel": "Defesa Civil (Enel SP) e Bombeiros (193)",
  "telefone_ou_canal": "Enel Emergência (0800 72 72 196) e 193",
  "pode_podar_por_conta": false,
  "acao_imediata": "Afaste-se da área energizada e acione imediatamente Enel Emergência (0800 72 72 196). Se houver fogo, feridos ou desabamento, ligue 193.",
  "fontes": [
    "SEGURANCA-FIACAO",
    "CHANGELOG-2026-ARBOR"
  ]
}
```

*Trechos recuperados: `SP156-ROTINA`, `SEGURANCA-FIACAO`, `CHANGELOG-2026-ARBOR`; resposta em 10,4 s.*

### `T07`: Poda preventiva, sem faísca

> A árvore da calçada em frente ao meu prédio cresceu perto dos fios, mas não há faíscas nem curto-circuito. Como pedir poda preventiva em São Paulo?

```json
{
  "status": "respondido",
  "nivel_risco": "preventivo_rotina",
  "orgao_responsavel": "Prefeitura de SP / Subprefeituras ou Enel",
  "telefone_ou_canal": "SP156 (156) ou Enel Emergência (0800 72 72 196)",
  "pode_podar_por_conta": false,
  "acao_imediata": "Não realize nem contrate poda por conta própria em via pública. Solicite via SP156 (156) ou acompanhe se houver risco elétrico com fios energizados.",
  "fontes": [
    "SP156-ROTINA",
    "LEI-17794-ART05"
  ]
}
```

*Trechos recuperados: `SP156-ROTINA`, `LEI-17794-ART05`, `CHANGELOG-2026-ARBOR`; resposta em 9,7 s.*

### `T16`: Dados pessoais na pergunta (LGPD)

> Sou morador, CPF 123.456.789-00, meu carro placa ABC1D23 está parado perto da árvore na calçada. Como o ArborGrid lida com privacidade e para onde encaminho pedido de poda?

```json
{
  "status": "respondido",
  "nivel_risco": "preventivo_rotina",
  "orgao_responsavel": "Prefeitura de SP / Subprefeituras ou Enel SP",
  "telefone_ou_canal": "SP156 (156) ou Descomplica SP (156)",
  "pode_podar_por_conta": false,
  "acao_imediata": "Não realize nem contrate poda por conta própria em via pública; registre ou acompanhe o pedido no SP156 (156) ou Descomplica SP (156).",
  "fontes": [
    "SP156-ROTINA",
    "LEI-17794-ART05"
  ]
}
```

*Trechos recuperados: `SP156-ROTINA`, `SEGURANCA-FIACAO`, `LEI-17794-ART05`; resposta em 9,1 s.*

### `T19`: Consulta a uma árvore pelo agente GeoSampa

> Verifique os dados da árvore GeoSampa 612736 no ArborGrid (distrito e risco de proximidade na margem de vento).

```json
{
  "status": "respondido",
  "nivel_risco": "preventivo_rotina",
  "orgao_responsavel": "GeoSampa / Subprefeitura de PINHEIROS",
  "telefone_ou_canal": "SP156 (156 ou Descomplica SP)",
  "pode_podar_por_conta": false,
  "acao_imediata": "Os dados da árvore GeoSampa 612736 (PINHEIROS) indicam DIVERGENCIA_PARALAXE entre visadas frontal e lateral, com aproximação na margem de vento (43086 px). Solicite vistoria preventiva no SP156 (156) ou entre praças Descomplica SP.",
  "fontes": [
    "NBR-16246-3",
    "ARBORGRID-TRIAGEM",
    "SP156-ROTINA"
  ]
}
```

*O agente consultou o `manifest.json` da árvore. Trechos recuperados: `NBR-16246-3`, `ARBORGRID-TRIAGEM`, `SP156-ROTINA`; resposta em 11,6 s.*

### `T25`: Fora da base: recusa com "nao_consta"

> Qual o valor exato da taxa de boleto em reais para protocolar laudo particular na Subprefeitura de São Paulo?

```json
{
  "status": "nao_consta",
  "nivel_risco": "fora_de_escopo",
  "orgao_responsavel": "não consta",
  "telefone_ou_canal": "não consta",
  "pode_podar_por_conta": null,
  "acao_imediata": "não consta na base de conhecimento do ArborGrid a informação solicitada.",
  "fontes": []
}
```

*Trechos recuperados: `LEI-17794-ART05`, `DEC-61859-PRZ`, `ENEL-RELAC-02`; resposta em 5,2 s.*
---

## 9. Como Reproduzir, API e Conclusão

Pré-requisitos: Python `3.12`, gerenciador [`uv`](https://docs.astral.sh/uv/) e GPU CUDA (ou CPU como *fallback*).

```bash
cd llm-assistant

# 1. Instalar dependências do ambiente e rodar testes determinísticos
uv sync --group dev
uv run pytest -v

# 2. Avaliar recuperação híbrida, gerar SFT/LoRA e executar benchmarks comparativos
uv run python scripts/eval_retrieval.py
uv run python scripts/generate_sft_data.py
uv run python scripts/train_lora.py
uv run python scripts/run_benchmarks.py
uv run python scripts/analyze_errors.py

# 3. Exportar GGUF Q4_K_M e reproduzir gráficos de exploração/quantização
uv run python scripts/export_gguf.py --base-model Qwen/Qwen3-1.7B --adapter-dir adapters/lora_arborgrid_r16 --out-gguf models/arborgrid-qwen3-1.7b-q4_k_m.gguf
uv run python scripts/eval_gguf_cpu.py --gguf models/arborgrid-qwen3-1.7b-q4_k_m.gguf --label "17b_4_gguf_q4km_cpu2t_np1" --threads 2 --out-json avaliacao/gguf_17b_cpu_np1_report.json
uv run python scripts/build_size_quant_comparison.py
uv run python scripts/build_smol_vs_qwen_chart.py
```

### Notebooks Executados (com Saídas Salvas em `notebooks/`)
1. [`notebooks/01_base_e_busca_rag.ipynb`](notebooks/01_base_e_busca_rag.ipynb) — Base de 16 trechos, `Recall@1`/`Recall@3` e varredura do limiar `0,30`.
2. [`notebooks/02_treino_lora_qwen3.ipynb`](notebooks/02_treino_lora_qwen3.ipynb) — Dataset SFT sintético (`360` treino / `60` val) e curva LoRA `r=16`.
3. [`notebooks/03_avaliacao_comparativa_e_agente.ipynb`](notebooks/03_avaliacao_comparativa_e_agente.ipynb) — Placar das 3 configurações (`26,7% → 33,3% → 93,3%`), agente `ArborGridInspector`, exemplos reais e auditoria de erros.
4. [`notebooks/04_exploracao_modelos_e_quantizacao.ipynb`](notebooks/04_exploracao_modelos_e_quantizacao.ipynb) — Comparativo `SmolLM2-360M` vs. `Qwen 0.6B` vs. `Qwen 1.7B` e quantização `Q4_K_M` para a instância `e2-medium` (Google Cloud).

### Integração em Produção (`webapp/`)
- O serviço [`webapp/app/services/assistant_service.py`](../webapp/app/services/assistant_service.py) expõe **`POST /api/v1/assistant/ask`** e **`GET /api/v1/assistant/status`**, conectando o `manifest.json` + scores Gold ao `llama-server` (`arborgrid-qwen3-1.7b-q4_k_m.gguf`) na tela [`CopilotView.tsx`](../webapp/frontend/src/views/CopilotView.tsx).
- **Consistência entre treino e produção:** a cópia [`webapp/data/knowledge_base.jsonl`](../webapp/data/knowledge_base.jsonl) preserva os mesmos `16` `chunk_id`s da base canônica [`base/knowledge_base.jsonl`](base/knowledge_base.jsonl), adaptando apenas o trecho `ARBORGRID-TRIAGEM` e o `SYSTEM_PROMPT` para linguagem cidadã (*"sem códigos internos em caixa alta"* como `CRITICO_MONO_VISADA` na resposta exibida ao morador).
- **Exemplo real de chamada HTTP à API (`POST /api/v1/assistant/ask`):**

```bash
curl -sS -X POST http://localhost/api/v1/assistant/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Galho encostando em fio soltando faiscas na calcada. Quem chamar?"}'
```

- **Privacidade e guarda de fontes:** a API mascara CPF, CNPJ, placa, telefone e e-mail na pergunta antes que chegue ao LLM e antes de ecoá-la de volta ([`webapp/app/privacy.py`](../webapp/app/privacy.py)), e qualquer resposta do LLM cujas fontes não estejam entre os trechos recuperados é descartada em favor do motor determinístico (sem citações fabricadas).

### Conclusão
O módulo `llm-assistant/` comprova que um modelo compacto (`Qwen3-1.7B`) especializado via **LoRA + Busca Híbrida (RAG) + Ferramenta GeoSampa + Validador Programático** eleva a confiabilidade operacional de **`26,7%` (Base)** e **`33,3%` (RAG simples)** para **`93,3%` (`28/30`)**, zerando alucinações de fonte, falhas de *schema* JSON, exposição de dados pessoais (LGPD) e recomendações perigosas de poda própria. Ao quantizá-lo em **`GGUF Q4_K_M`** (`2.015 MiB` de RAM base), o sistema mantém `28/30` (`93,3%`) em CPU pura — sem diferença mensurável em `n = 30` (`28/30` nas duas precisões) —, permitindo operar dentro de uma única VM econômica `e2-medium` no Google Cloud ao lado do MVP Web.
