# Normas-bot do ArborGrid: Robotic Process Automation (`normas-bot/`)

Automação em Python para verificar e manter rastreável a base de normas e canais oficiais utilizada pelo Assistente do ArborGrid SP.

## O problema

O Assistente do ArborGrid SP responde dúvidas de cidadãos e equipes técnicas com base em 16 trechos de leis municipais, decretos, manuais técnicos e canais de atendimento (Prefeitura de São Paulo, SP156, Enel SP, Corpo de Bombeiros, Defesa Civil, ABNT NBR 16246, NR-10 e especificações internas).

Telefones, prazos e dispositivos legais mudam em portais públicos diferentes. Conferir manualmente todas as fontes e comparar cada trecho leva cerca de uma hora por ciclo. Sem verificação recorrente, uma alteração de canal ou prazo em um portal público passa despercebida até causar encaminhamento incorreto.

## O que o robô faz

O pacote Python fica em `src/rpa_normativo/` e executa um ciclo em lote sobre `fontes.yaml`:

1. Lê a configuração das fontes monitoradas (`fontes.yaml`), a base original (`webapp/data/knowledge_base.jsonl`) e o estado da execução anterior (`estado/ultima_execucao.json`).
2. Percorre cada fonte em lote, sem interromper a rodada quando uma fonte falha.
3. Aplica a regra R1 para fontes `fechada` (ABNT/NR-10) ou `interna` (diretrizes do projeto), pulando o download com justificativa registrada.
4. Baixa páginas HTML e arquivos PDF via HTTPS com `httpx` (timeout de 15 s e `User-Agent` identificando o robô e a FIAP) ou lê cópias locais em `amostras/fontes/` no modo `--offline`.
5. Extrai o texto (HTML com `BeautifulSoup` removendo `script` e `style`; PDF com `pypdf`), normaliza espaços e caixa e calcula o digest `sha256`.
6. Compara o `sha256` com a execução anterior e confere a presença dos fatos críticos configurados por trecho.
7. Grava a base sincronizada, o relatório estruturado, o estado atualizado e o arquivo de auditoria diária, encerrando com código de saída `0` (tudo confirmado ou sem alteração) ou `2` (há revisão humana ou erro).

## Estrutura da pasta

```text
normas-bot/
  README.md
  requirements.txt
  fontes.yaml
  src/rpa_normativo/
  tests/
  amostras/fontes/
  docs/PDD.md
  assets/
  logs/
  saida/
  estado/
```

## Instalação

No diretório `normas-bot/` com Python 3.12:

```bash
python -m venv .venv
```

Ative o ambiente virtual:

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Instale as dependências com versões fixas:

```bash
pip install -r requirements.txt
```

## Como executar

Entre na pasta `normas-bot/src/` e execute um dos comandos abaixo:

```bash
cd normas-bot/src

# Execucao online nas fontes oficiais
python -m rpa_normativo

# Execucao 100% offline usando amostras/fontes/
python -m rpa_normativo --offline

# Simulacao da regra R4 (alteracao de telefone na amostra da Enel)
python -m rpa_normativo --offline --simular-mudanca enel-emergencia
```

Para demonstrar a regra R5 (timeout e retry com falha tratada sem parar o lote) no modo offline:

```bash
python -m rpa_normativo --offline --simular-falha sp156-portal
```

## Regras de negócio

Uma mudança real em uma norma troca parte dos fatos (como um telefone ou um prazo), não todos. Quando nenhum dos fatos monitorados aparece no texto baixado, a página não traz o documento (endereço alterado, bloqueio ou conteúdo montado apenas via JavaScript), sendo classificada como erro de sistema (R7) e não como mudança normativa (R4).

| Regra | Situação | Status | Efeito nos trechos |
| :--- | :--- | :--- | :--- |
| R1 | Fonte `fechada` ou `interna` | `FONTE_FECHADA` ou `FONTE_INTERNA` | Mantém o trecho sem download externo e explica o motivo no relatório |
| R2 | Download concluído e `sha256` igual ao da última execução | `SEM_ALTERACAO` | Mantém o trecho, atualiza `verificado_em` e não reextrai fatos |
| R3 | Hash novo (ou primeira execução) e todos os fatos presentes | `CONFIRMADA` | Mantém o conteúdo original e grava o novo `sha256` no estado |
| R4 | Hash novo e parte dos fatos sumiu | `REVISAO_HUMANA` | Mantém o conteúdo anterior intacto, marca o trecho e lista `fatos_ausentes` |
| R5 | Timeout, erro de conexão ou HTTP 5xx | `ERRO_REDE` após 3 tentativas (espera de 1 s, 2 s e 4 s) | Mantém o trecho como `NAO_VERIFICADO` e continua o lote |
| R6 | HTTP 4xx | `ERRO_HTTP` em 1 tentativa (sem retry) | Mantém o trecho como `NAO_VERIFICADO` e continua o lote |
| R7 | Download ok, mas a página não traz o documento: sem texto extraível (página vazia, PDF escaneado) ou sem nenhum dos fatos monitorados (endereço mudou, bloqueio ou página montada por JavaScript) | `ERRO_CONTEUDO` | Mantém o trecho como `NAO_VERIFICADO` (exceção de sistema) e continua o lote |

## Logs e saídas

- `logs/rpa_normativo_AAAAMMDD.log`: uma linha por evento com timestamp ISO-8601, `exec=`, `fonte=`, `etapa=` (`inicio`, `download`, `extracao`, `decisao`, `gravacao`, `fim`) e `status=`, sem dados pessoais.
- `saida/knowledge_base_synced.jsonl`: os mesmos 16 trechos na mesma ordem, com os cinco campos originais inalterados e o objeto adicional `verificacao` (`status`, `fonte_id`, `url`, `sha256`, `verificado_em`, `fatos_ausentes`).
- `saida/relatorio_execucao.json`: resumo da execução com `execucao_id`, início, fim, duração, modo, contagem por status, lista de exceções separada em `excecoes_negocio` (R4) e `excecoes_sistema` (R5 a R7), fontes e trechos.
- `estado/ultima_execucao.json`: hash `sha256` confirmado e data da última verificação por fonte.

## Integração com o Assistente (MVP)

O serviço do Assistente em `webapp/app/services/assistant_service.py` carrega a base definida pela variável de ambiente `ARBORGRID_KNOWLEDGE_BASE_PATH` (`webapp/app/config.py`) e ignora campos extras como `verificacao` ao indexar os trechos.

Para apontar o web app para a base verificada pelo robô:

```powershell
$env:ARBORGRID_KNOWLEDGE_BASE_PATH = "C:\caminho\para\repo\normas-bot\saida\knowledge_base_synced.jsonl"
```

```bash
export ARBORGRID_KNOWLEDGE_BASE_PATH="/caminho/para/repo/normas-bot/saida/knowledge_base_synced.jsonl"
```

O endpoint `GET /api/v1/assistant/knowledge-status` no web app também lê `normas-bot/saida/relatorio_execucao.json` (ou o caminho definido por `ARBORGRID_KNOWLEDGE_REPORT_PATH`) para expor a data da última sincronização e a contagem por status.

## Testes automatizados

Todos os testes rodam 100% offline com `httpx.MockTransport` e diretórios temporários (`tmp_path`). Na pasta `normas-bot/`:

```bash
pytest -v
```

