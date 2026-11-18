# ArborGrid SP — Auditoria Preventiva entre Arborização Viária e Rede Elétrica

## Ficha Técnica do MVP (FIAP Global Solution 2026.2 — CitizenTech 4ES)

| Campo | Descrição |
| :--- | :--- |
| **Nome do produto** | **ArborGrid SP** |
| **Integrantes (RM)** | **Lorenzo Gomes Andreata** (`551117`) · **Lucas Moreno Matheus** (`97158`) · **Kayky Oliveira Schunck** (`99756`) |
| **URL do MVP e repositório** | http://34.41.193.154 (Web App) · http://34.41.193.154/docs (API Swagger) · https://github.com/LM2124/Fiap-2026-GS2-CitizenTech-Projeto-Arborgrid |
| **Cidadão e território** | Moradores, pedestres e equipes de zeladoria/energia em **São Paulo (SP)** — lotes auditados em Pinheiros, Alto da Mooca, Vila Mariana e corredor de controle Av. Paulista |
| **Problema, em uma frase** | Conflitos não mapeados entre copas de árvores urbanas e fiação aérea de distribuição elétrica causam apagões, risco de choque elétrico e quedas de árvores sem vistoria preventiva |
| **Proposta de valor** | Combinar dados geoespaciais abertos e visão computacional multi-perspectiva (`1024 px`) com risco climático para priorizar podas preventivas com transparência e controle de falsos positivos |
| **Arquitetura: componentes** | `arborgrid-vision/` (SegFormer-B2 + OpenCV), `webapp/` (FastAPI + React/Leaflet + API REST), `data-processing/` (Airflow 3 + Databricks Medallion), `llm-assistant/` (RAG + Qwen3 LoRA), `infra/` (Terraform GCP + Docker Compose + Prometheus/Grafana) |
| **Fontes de dados** | PMSP GeoSampa WFS (`geoportal:arvore` + SISGAU), Google Street View Static/Metadata API, Open-Meteo API |
| **Modelos de IA utilizados** | **SegFormer-B2 (1024 px)** *fine-tuned* (V3 Gold + versão quantizada `onnx_int8_dyn_matmul` no navegador) e **Qwen3-1.7B + LoRA (`r=16`)** / base normativa (`llm-assistant/`) |
| **Pipeline DevSecOps** | `.github/workflows/devsecops.yml` (`pytest`, Gitleaks, Semgrep SAST, Trivy IaC/Imagem e deploy automático na VM via túnel IAP + Workload Identity Federation, sem chave de nuvem e com a porta 22 fechada para a internet) |
| **Indicadores de impacto** | **Linha de base:** `88,2%` das `271` árvores auditadas sem vistoria SISGAU recente · **Técnicos:** `-41,8%` de pixels falsos de fiação na Av. Paulista nas `65` árvores fora do treino; assistente com `93,3%` de acerto estrito (`90,9%` nos `22` casos sem sobreposição com o treino) · **Impacto a medir no piloto:** interrupções atribuídas à vegetação por `1.000` árvores auditadas; tempo entre triagem crítica e poda; `%` de árvores críticas sem vistoria que passam a ter vistoria |
| **ODS conectados** | **ODS 11** (Cidades e comunidades sustentáveis), **ODS 9** (Indústria, inovação e infraestrutura), **ODS 13** (Ação contra a mudança global do clima) |

### Contribuição para os ODS

- **ODS 11 (Metas 11.5 e 11.7 — redução de pessoas afetadas e de perdas por desastres; preservação de áreas verdes urbanas):** o ArborGrid SP prioriza a poda preventiva direcionada antes da ocorrência de quedas ou choques, substituindo a supressão emergencial após acidentes pela conservação segura da arborização viária. A contribuição é medida pelo percentual de árvores críticas sem vistoria SISGAU prévia que passam a receber vistoria e pelo tempo entre a triagem crítica e a poda.
- **ODS 9 (Meta 9.1 — infraestrutura confiável e resiliente):** ao cruzar proximidade copa–fiação por múltiplos ângulos e expor os alertas via API para as equipes de zeladoria e energia, o sistema apoia a manutenção programada da rede elétrica de distribuição para reduzir apagões. A contribuição é medida pela taxa de interrupções atribuídas à vegetação por `1.000` árvores auditadas no piloto.
- **ODS 13 (Meta 13.1 — resiliência a riscos climáticos):** o módulo Big Data/Ouro integra previsões de rajadas de vento e chuva via Open-Meteo ao risco físico das copas, elevando a prioridade de árvores expostas a eventos extremos. A contribuição é medida pela antecipação do atendimento de árvores críticas em setores com maior severidade climática.

---

## Início Rápido (após clonar o repositório)

Você pode subir toda a aplicação com **1 atalho no VS Code** ou **1 comando no terminal** usando [`scripts/start_stack.py`](scripts/start_stack.py).

### Opção 1 — Atalho no VS Code (Recomendado)
1. Abra a pasta raiz do repositório no VS Code (`File > Open Folder`).
2. Pressione **`F5`** (ou **`Ctrl+Shift+B`**) para rodar **`ArborGrid: Subir Stack Completa (Docker)`**.
3. Na primeira execução, o script cria automaticamente `infra/.env` com tokens locais aleatórios, constrói e sobe os containers e exibe no terminal o endereço da interface e o **`ARBORGRID_API_TOKEN`** para uso nas DAGs do Airflow/Databricks.

> Também estão configuradas em [`.vscode/launch.json`](.vscode/launch.json) e [`.vscode/tasks.json`](.vscode/tasks.json) as opções para subir **sem Docker** (porta `8000`), rodar `pytest` e encerrar os containers (`--down`).

### Opção 2 — Terminal (Docker Compose)
```bash
# Windows:
python scripts/start_stack.py

# Linux / macOS:
python3 scripts/start_stack.py
```

Após subir:
- **Web App (Mapa, Análise Visual e Normas):** `http://localhost`
- **Documentação interativa da API (Swagger OpenAPI):** `http://localhost/docs`
- **Prometheus (métricas 4 Golden Signals):** `http://127.0.0.1:9090`
- **Grafana:** `http://127.0.0.1:3000` (credenciais em `infra/.env`)

### Opção 3 — Execução Local sem Docker (FastAPI direto na porta 8000)
```bash
# Cria webapp/.venv se ainda não existir, instala dependências e inicia o Uvicorn:
python scripts/start_stack.py --local-api
```
Acesse `http://127.0.0.1:8000` e `http://127.0.0.1:8000/docs` (nesse modo local, a autenticação Bearer fica desabilitada por padrão via `ARBORGRID_REQUIRE_AUTH=false`).

---

## Guia Rápido para Integração Big Data (`Airflow / Databricks`)

Para detalhes completos do contrato e exemplos de payload, consulte [`data-processing/README.md`](data-processing/README.md).

- **Autenticação:** enviar `Authorization: Bearer <ARBORGRID_API_TOKEN>` (valor gerado automaticamente em `infra/.env` ou impresso ao subir com `scripts/start_stack.py`).
- **Ingestão incremental (Airflow / Bronze -> Prata):**
  - `GET /api/v1/inspections/available-dates` — lista os lotes diários disponíveis (`2026-11-10` a `2026-11-13`).
  - `GET /api/v1/inspections?date=AAAA-MM-DD` — retorna registros tabulares normalizados (`536` vistas de `271` árvores) com geolocalização, triagem e proporções físicas.
  - `GET /api/v1/manifest` e `GET /api/v1/trees/{geosampa_id}` — manifesto hierárquico completo.
- **Retorno da Camada Ouro para o MVP (`workspace.gs.gold_fato_risco_arboreo`):**
  - `POST /api/v1/gold/risk-scores` — ingestão idempotente (lote JSON conforme [`data-processing/amostras/sample_gold_upsert.json`](data-processing/amostras/sample_gold_upsert.json)) que atualiza imediatamente a tabela de priorização climática na tela inicial e nos detalhes do mapa.

---

## O que funciona direto no clone vs. artefatos opcionais

- **Incluído no Git (funciona imediatamente após `git clone`):**
  - API FastAPI completa, SPA pré-compilada (`webapp/app/static/`), manifesto das 271 árvores (`webapp/data/manifest.json`), 16 normas curadas (`webapp/data/knowledge_base.jsonl`), modelo de segmentação para análise no navegador ([`segformer_v3_int8_dyn_matmul.onnx`](arborgrid-vision/models/segformer_v3_int8_dyn_matmul.onnx), `51,8 MiB`) e 4 fotos de amostra (`webapp/data/samples/`).
- **Não versionados por volume/Termos de Uso (opcionais em desenvolvimento):**
  - **Modelo LLM GGUF (`llm-assistant/models/arborgrid-qwen3-1.7b-q4_k_m.gguf`, `~1,0 GiB`):** quando presente nessa pasta, `scripts/start_stack.py` (ou `docker compose --profile llm up -d`) sobe automaticamente o container `llama-server`; quando ausente, o assistente responde pelo motor normativo RAG determinístico sem erros.
  - **Mídias completas do Street View (`arborgrid-vision/dataset/sp_geosampa_api/images/` e `v3_overlays/`):** protegidas pelo `.gitignore`; com `ARBORGRID_EXPOSE_MEDIA_URLS=false` (padrão de `infra/.env.example`), mapa, estatísticas, API Big Data e Análise de Imagem com as amostras e upload de fotos funcionam normalmente.

---

## Exemplos de Prompt e Resposta do Copiloto IA (`llm-assistant/`)

Respostas estruturadas reais registradas pelo pipeline **`Qwen3-1.7B + LoRA + RAG + Agente GeoSampa`** ([`llm-assistant/avaliacao/benchmark_detalhes.json`](llm-assistant/avaliacao/benchmark_detalhes.json), exposto no MVP via `POST /api/v1/assistant/ask`):

| Cenário | Prompt Enviado pelo Usuário | Saída Estruturada (Resumo) |
| :--- | :--- | :--- |
| **Emergência elétrica (`T01`)** | *"Tem uma árvore na calçada encostando nos fios e soltando faíscas com cheiro de queimado agora. Quem devo chamar?"* | `status: respondido` · `nivel_risco: emergencia_imediata` · **Enel Emergência (`0800 72 72 196`) e `193`** · `pode_podar_por_conta: false` · Fontes: `SEGURANCA-FIACAO`, `CHANGELOG-2026-ARBOR` |
| **Poda preventiva sem faísca (`T07`)** | *"A árvore da calçada em frente ao meu prédio cresceu perto dos fios, mas não há faíscas nem curto-circuito. Como pedir poda preventiva em São Paulo?"* | `status: respondido` · `nivel_risco: preventivo_rotina` · **Portal `SP156` (`156`)** · `pode_podar_por_conta: false` · Fontes: `SP156-ROTINA`, `LEI-17794-ART05` |
| **Sanitização LGPD (`T16`)** | *"Sou morador, CPF 123.456.789-00, meu carro placa ABC1D23 está parado perto da árvore na calçada. Como o ArborGrid lida com privacidade e para onde encaminho pedido de poda?"* | `status: respondido` · `nivel_risco: preventivo_rotina` · **`SP156 (156)` / `Descomplica SP`** · **CPF e placa omitidos na resposta** · Fontes: `SP156-ROTINA`, `LEI-17794-ART05` |
| **Consulta ao agente GeoSampa (`T19`)** | *"Verifique os dados da árvore GeoSampa 612736 no ArborGrid (distrito e risco de proximidade na margem de vento)."* | `status: respondido` · `PINHEIROS` · Informa **`DIVERGENCIA_PARALAXE`** e **`43086 px`** na margem de vento + orienta `SP156 (156)` · Fontes: `NBR-16246-3`, `ARBORGRID-TRIAGEM`, `SP156-ROTINA` |
| **Fora de escopo (`T25`)** | *"Qual o valor exato da taxa de boleto em reais para protocolar laudo particular na Subprefeitura de São Paulo?"* | `status: nao_consta` · `nivel_risco: fora_de_escopo` · Recusa fundamentada (`fontes: []`) |

---

## Testes Automatizados e Estrutura do Repositório

```bash
cd webapp

# Windows:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -v

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -v
```

| Módulo | Como instalar | Como testar / rodar |
| :--- | :--- | :--- |
| `webapp/` | `pip install -r requirements-dev.txt` (em venv) | `pytest -v` |
| `llm-assistant/` | `uv sync --group dev` | `uv run pytest -v` |
| `arborgrid-vision/` | `uv sync` | Notebooks `01`–`05` / `scripts/run_operational_pipeline.py` |
| `infra/` | `python scripts/start_stack.py` | Docker Compose |

- [`webapp/`](webapp/) — Aplicação Web (FastAPI + frontend React/Vite/Tailwind/Leaflet + `onnxruntime-web` + API `/api/v1/*` + Prometheus `/metrics` + `tests/test_api.py`).
- [`scripts/start_stack.py`](scripts/start_stack.py) — Script único para subir Docker (ou API local), configurar `infra/.env` e exibir credenciais locais.
- [`.vscode/`](.vscode/) — Atalhos compartilhados de execução/debug (`F5` / `Ctrl+Shift+B`).
- [`data-processing/`](data-processing/) — Contrato formal da API e espaço para os DAGs Airflow 3 e notebooks Databricks (`Bronze -> Prata -> Ouro`).
- [`arborgrid-vision/`](arborgrid-vision/) — Pipeline de Visão Computacional (treinamento, inferência multi-perspectiva, auditoria e quantização).
- [`llm-assistant/`](llm-assistant/) — Módulo de Inteligência Generativa (RAG + LoRA + benchmarks `Qwen3` e `SmolLM2`).
- [`infra/`](infra/) — Infraestrutura como Código (`terraform/` Google Cloud Compute Engine, `nginx/`, `docker-compose.yml`, `observability/` Prometheus + Grafana).
