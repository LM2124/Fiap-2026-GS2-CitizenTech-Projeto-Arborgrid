# Arquitetura do Sistema (`llm-assistant/`)

> Documento complementar ao [`README.md`](README.md) e ao [`EXPLORACAO_MODELOS_E_QUANTIZACAO.md`](EXPLORACAO_MODELOS_E_QUANTIZACAO.md). Detalha os módulos em [`src/`](src/) e [`scripts/run_benchmarks.py`](scripts/run_benchmarks.py) que diferenciam as 3 configurações comparadas na ablação arquitetural.

---

## 1. Evolução Progressiva das 3 Configurações Testadas

```mermaid
flowchart TB
    Q["Pergunta do Cidadão / Operador"] --> C1["1. Base Zero-Shot (SYSTEM_PROMPT + Pergunta)"]
    Q --> C2["2. RAG Simples (+ KnowledgeRetriever)"]
    Q --> C3["3. Fine-tuning LoRA + RAG + Agente (Completa)"]

    subgraph CFG1["1. Base Zero-Shot"]
        B1["Modelo original (sem LoRA)"]
    end

    subgraph CFG2["2. RAG Simples"]
        R2["KnowledgeRetriever (top_k=3, limiar=0.30)"] --> B2["Modelo original (sem LoRA)"]
    end

    subgraph CFG3["3. Fine-tuning LoRA + RAG + Agente"]
        R3["KnowledgeRetriever (top_k=3, limiar=0.30)"] --> P3["[CONTEXTO] + [DADOS_TECNICOS_ARBORGRID]"]
        T3["ArborGridInspector (manifest.json)"] --> P3
        P3 --> L3["Modelo + LoRA r=16 (enable_thinking=False)"]
        L3 --> V3["apply_post_generation_filters() (Citações + Risco)"]
    end

    C1 --> CFG1 --> O1["JSON bruto (sem guardrails)"]
    C2 --> CFG2 --> O2["JSON com [CONTEXTO] (sem guardrails)"]
    C3 --> CFG3 --> O3["JSON estrito 7 chaves + fontes + piso de segurança"]
```

| Configuração (`scripts/run_benchmarks.py`) | `KnowledgeRetriever` (RAG) | `ArborGridInspector` (Agente) | Adaptador LoRA (`r=16`) | `apply_post_generation_filters()` |
| :--- | :---: | :---: | :---: | :---: |
| **`1_base_sem_contexto`** (`Base Zero-Shot`) | — | — | — | — |
| **`2_rag_simples`** (`RAG Simples`) | Sim (`[CONTEXTO]`) | — | — | — |
| **`3_rag_lora_agente`** (`Fine-tuning LoRA + RAG + Agente`) | Sim (`[CONTEXTO]`) | Sim (`manifest.json` + normas) | Sim | Sim |

---

## 2. Fluxo Ponta a Ponta da Arquitetura Final (`src/`)

```mermaid
flowchart LR
    U["Pergunta do Usuário"] --> R["KnowledgeRetriever (src/retrieval.py)"]
    U --> T["ArborGridInspector (src/tools.py)"]

    subgraph RAG["Recuperação Híbrida (base/knowledge_base.jsonl)"]
        R --> S["score = dense_sims + 0.35 * lexical_sims"]
        S --> F{"score >= 0.30?"}
        F -->|"Sim (até top_k=3)"| K1["Inclui trechos em [CONTEXTO]"]
        F -->|"Não"| K0["CONTEXTO vazio (força nao_consta)"]
    end

    subgraph AGENTE["Ferramenta Determinística"]
        T --> RE{"_TREE_ID_RE no manifest.json?"}
        RE -->|"Sim"| MD["Extrai distrito, SISGAU, v3 + contact/wind px + ARBORGRID-TRIAGEM e SP156-ROTINA"]
        RE -->|"Não"| NO["Sem bloco [DADOS_TECNICOS_ARBORGRID]"]
    end

    K1 --> PROMPT["format_chat_prompt() (src/common.py)"]
    K0 --> PROMPT
    MD --> PROMPT
    NO --> PROMPT

    PROMPT --> LLM["LLM + LoRA r=16 (enable_thinking=False)"]
    LLM --> VAL["apply_post_generation_filters() (src/validator.py)"]
    VAL --> OUT["JSON de 7 chaves"]
```

---

## 3. Detalhamento dos Módulos de Controle

### 3.1 Busca Híbrida e Gate de Domínio ([`src/retrieval.py`](src/retrieval.py))
- **`KnowledgeRetriever`:** combina *embeddings* densos normalizados (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) com TF-IDF (`TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)`) sobre os 25 trechos curados de [`base/knowledge_base.jsonl`](base/knowledge_base.jsonl).
- **Ranqueamento e filtragem:** `score = dense_sims + (0.35 * lexical_sims)`. Apenas trechos com `score >= 0.30` (`DEFAULT_RETRIEVAL_THRESHOLD`, `top_k = 3`) entram no prompt via `format_context_block()`. Quando nenhum trecho atinge o limiar, o prompt informa explicitamente ausência de contexto confiável.

### 3.2 Agente Determinístico GeoSampa ([`src/tools.py`](src/tools.py))
- **`ArborGridInspector.maybe_build_tool_context()`:** identifica menções numéricas por `_TREE_ID_RE` e consulta `by_id` carregado de [`manifest.json`](../arborgrid-vision/dataset/sp_geosampa_api/manifest.json).
- **Agregação conservadora e injeção normativa:** consolida a gravidade entre vistas pela ordem **`CRITICO > PROXIMIDADE > SEGURO`**, calcula picos de pixel de contato direto (`max_contact_direct_px`) e margem de vento (`max_wind_buffer_px`), e injeta o bloco `[DADOS_TECNICOS_ARBORGRID]`.
- Quando encontra uma árvore válida, [`scripts/run_benchmarks.py`](scripts/run_benchmarks.py#L266-L273) também garante que os trechos normativos `ARBORGRID-TRIAGEM` e `SP156-ROTINA` componham os *chunks* disponíveis para citação exata.

### 3.3 Validador Programático e Filtros Pós-Geração ([`src/validator.py`](src/validator.py))

```mermaid
flowchart TD
    RAW["Saída bruta do LLM"] --> PARSE["extract_json_object()"]
    PARSE -->|"JSON inválido"| PASS["Retorna bruto (falha em validate_schema)"]
    PARSE -->|"dict válido"| ST{"status == 'nao_consta'?"}
    ST -->|"Sim"| CLR["Força fontes = []"]
    ST -->|"Não (respondido)"| SRC["Filtra fontes apenas em allowed_ids (retrieved_chunks)"]
    SRC --> PH{"Cita telefone 08007272196, 193 ou 199?"}
    PH -->|"Não"| OUT["Serializa JSON final"]
    PH -->|"Sim"| HAZ{"_ACTIVE_HAZARD_RE ativo sem _NEGATED_HAZARD_RE?"}
    HAZ -->|"Sim"| E1["Piso: nivel_risco = 'emergencia_imediata'"]
    HAZ -->|"Não"| E2{"Era 'fora_de_escopo' ou 'preventivo_rotina' sem 156?"}
    E2 -->|"Sim"| E3["Eleva piso: nivel_risco = 'urgente_vistoria'"]
    E2 -->|"Não"| OUT
    CLR --> OUT
    E1 --> OUT
    E3 --> OUT
```

- **Avaliação determinística (`evaluate_prediction()`):** verifica conformidade de esquema com `validate_schema()` sobre as 7 chaves de `RESPONSE_FIELDS` ([`src/common.py`](src/common.py)), valida telefones contra `allowed_phones` extraídos do contexto recuperado e garante ausência de dados pessoais na resposta (`_CPF_RE` e `_PLATE_RE` para LGPD).
