# ArborGrid — Módulo de Big Data & Data Integration (`data-processing/`)

Pipeline de dados (Airflow 3 + Databricks Lakehouse Bronze/Prata/Ouro) e documentação oficial do contrato REST oferecido pela aplicação ArborGrid (`webapp/`).

## Estrutura

```text
data-processing/
├── README.md                    # Este arquivo: contrato de integração da API ArborGrid
├── amostras/                    # Payloads de exemplo do contrato (API -> Big Data e Gold -> MVP)
└── arborgrid-bigdata/           # Pipeline implementado (Requisitos 01-09)
    ├── dags/                    # DAG arborgrid_coleta (Airflow 3, diária): clima, árvores e denúncias em paralelo + verificação
    ├── airflow/                 # docker-compose do Airflow (LocalExecutor)
    ├── chegada/                 # Um arquivo por coleta e por fonte, com a data no nome (JSON Lines)
    ├── databricks/              # 00 setup · 01 Bronze (Auto Loader) · 02 Prata (qualidade + MERGE) · 03 Ouro (estrela + LGPD) · 04 consultas · 05 contrato MVP
    ├── amostras/                # Amostras das três fontes e o lote Gold exportado no contrato do MVP
    ├── scripts/                 # enviar_ouro_mvp.py: publica o lote Gold em POST /api/v1/gold/risk-scores
    └── README.md                # Documentação completa do pipeline (Requisitos 01-09)
```

---

## Contrato Oficial da API ArborGrid (`v1`)

Todos os endpoints de ingestão utilizam autenticação via header **Bearer Token** (armazenar em **Airflow Variables**, nunca hardcoded no código da DAG):

```http
Authorization: Bearer <ARBORGRID_API_TOKEN>
```

### 1. Fonte Primária Tabular (Recomendada para Airflow/Spark): `GET /api/v1/inspections`

Retorna registros normalizados (1 linha por vista Street View analisada pelo modelo **SegFormer-B2 1024 px V3 Gold**, total de `536` vistas cobrindo `271` árvores em 4 regiões de São Paulo), já preparados para particionamento diário incremental (`?date=AAAA-MM-DD`).

**Parâmetros de consulta suportados:**
- `date` (opcional, `AAAA-MM-DD`): filtra os registros do lote operacional daquela data. Retorna `items: []` quando não há captura na data.
- `region_key` (opcional): `pinheiros`, `alto_da_mooca`, `vila_mariana`, `avenida_paulista`.
- `triage` (opcional): `CRITICO`, `PROXIMIDADE`, `SEGURO`.
- `conflict_only` (opcional, `true|false`): filtra apenas `CRITICO` e `PROXIMIDADE`.
- `limit` (1..1000, padrão 200) e `offset` (padrão 0).

**Datas de lote já populadas (para carga incremental em >= 2 lotes no Databricks Volume):**
Consultáveis via `GET /api/v1/inspections/available-dates`:
- `2026-11-10` — Pinheiros
- `2026-11-11` — Alto da Mooca
- `2026-11-12` — Vila Mariana
- `2026-11-13` — Avenida Paulista (Controle Negativo - rede subterrânea)

> `captured_at` é uma partição operacional **sintética** (gerada de forma determinística por região) para simular coletas diárias incrementais; a data real da imagem está em `streetview_capture_month`.

**Campos garantidos por item (`items[]`):**
- `detection_id`: identificador único e determinístico (`arb-<sha256_16>`)
- `image_id`: identificador base da fotografia
- `geosampa_id`: ID oficial da árvore na PMSP/GeoSampa
- `region_key`, `region_label`, `distrito`, `subprefeitura`, `sisgau_last_inspection`
- `perspective`: `center_front`, `side_angle_A`, `side_angle_B`
- `streetview_capture_month`: mês real da captura Street View (`AAAA-MM`)
- `captured_at`: timestamp ISO 8601 com fuso (`-03:00`) do lote operacional
- `latitude`, `longitude`: coordenadas decimais (`5` casas)
- `triage_class`: `CRITICO` | `PROXIMIDADE` | `SEGURO`
- `tree_consolidated_triage`: classificação consolidada multi-ângulo da árvore
- `conflict_category`: `tree_wire_conflict` | `no_conflict`
- `confidence`: score contínuo (`0.0` a `1.0`) derivado da evidência física em pixels
- `model_version`: `segformer-b2-1024-arborgrid-v3-gold`
- `copa_px`, `wire_px`, `pole_px`, `contact_direct_px`, `wind_buffer_px`
- `pano_id_hash`: hash SHA-256 truncado (LGPD / sem expor identificadores brutos)

Veja exemplo completo em [`amostras/sample_inspections_response.json`](amostras/sample_inspections_response.json).

### 2. Manifesto Completo Hierárquico: `GET /api/v1/manifest` e `GET /api/v1/trees`

Mantém a estrutura original do `manifest.json` agrupada por `geosampa_id`, com `views[]` aninhadas e totalizadores dinâmicos corrigidos.

### 3. Contrato de Retorno Gold -> MVP: `POST /api/v1/gold/risk-scores`

Para fechar o ciclo exigido na rubrica de Big Data (mostrar qual tabela da camada Ouro o MVP consome e exibe), o Web App disponibiliza ingestão idempotente (`MERGE` por `geosampa_id`) da tabela **`workspace.gs.gold_fato_risco_arboreo`**:

- Payload de exemplo em [`amostras/sample_gold_upsert.json`](amostras/sample_gold_upsert.json).
- Assim que o lote é enviado (pelo Databricks exportado ou script de integração), o mapa interativo do Web App exibe automaticamente o **Score Composto Gold (0–100)**, rajada de vento e prioridade de poda (`URGENTE`, `ALTA`, `MEDIA`, `BAIXA`) no painel lateral e nos popups de cada árvore.

---

## Segunda Fonte Externa Sugerida (Airflow Paralelo)

Para cruzar conflito espacial de copa com severidade climática e validar os distritos de São Paulo:
- **Open-Meteo API** (`https://api.open-meteo.com/v1/forecast` ou `/v1/archive`): `wind_speed_10m`, `wind_gusts_10m`, `precipitation` por centroide de cada `region_key` (gratuita, sem necessidade de chave).
