"""Gera os 3 notebooks Jupyter reproduzíveis em llm-assistant/notebooks/."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
DIR_NB = ROOT / "notebooks"


def _nb_base_rag() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(
            "# 01 — Base de Conhecimento, Busca Híbrida (RAG) e Varredura de Limiar\n\n"
            "Este notebook documenta:\n"
            "1. A tabela completa dos 16 trechos curados (`base/knowledge_base.jsonl`) com fonte e categoria.\n"
            "2. A avaliação determinística da recuperação semântica + lexical (`Recall@1`, `Recall@3`, `MRR@3`).\n"
            "3. A justificativa empírica do limiar de similaridade (`0.30`) por meio da varredura e gráfico de compensação."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(ROOT))\n\n"
            "from src.common import DEFAULT_RETRIEVAL_THRESHOLD, DIR_ASSETS, DIR_BASE, DIR_EVAL, load_jsonl\n\n"
            "kb = load_jsonl(DIR_BASE / 'knowledge_base.jsonl')\n"
            "df_kb = pd.DataFrame(kb)[['chunk_id', 'categoria', 'titulo', 'fonte']]\n"
            "print(f'Total de trechos curados: {len(df_kb)}')\n"
            "display(df_kb)"
        ),
        nbf.v4.new_code_cell(
            "REEXECUTAR_BUSCA = False  # Altere para True para recalcular embeddings em tempo de execução\n\n"
            "if REEXECUTAR_BUSCA or not (DIR_EVAL / 'retrieval_threshold_sweep.csv').exists():\n"
            "    from scripts.eval_retrieval import main as run_eval\n"
            "    run_eval()\n\n"
            "df_sweep = pd.read_csv(DIR_EVAL / 'retrieval_threshold_sweep.csv')\n"
            "display(df_sweep)\n"
            "display(Image(filename=str(DIR_ASSETS / 'retrieval_threshold_sweep.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "### Decisão: Limiar de Similaridade em `0.30`\n"
            "Pela curva acima, **`0.30`** zera a recusa indevida (`false_refusal_rate = 0.0000`) mantendo `Recall@1 = 0.6818`, `Recall@3 = 0.9545` e `MRR@3 = 0.8030`."
        ),
        nbf.v4.new_code_cell(
            "report = json.loads((DIR_EVAL / 'retrieval_report.json').read_text(encoding='utf-8'))\n"
            "print('Limiar escolhido:', DEFAULT_RETRIEVAL_THRESHOLD)\n"
            "print('Métricas no limiar escolhido:', report['chosen_metrics'])"
        ),
    ]
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    return nb


def _nb_lora() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(
            "# 02 — Especialização via Fine-Tuning LoRA (Qwen3-1.7B)\n\n"
            "Objetivo da especialização (alinhado ao princípio *fine-tuning ensina como responder; RAG e ferramenta fornecem o conhecimento*):\n"
            "- Ensinar disciplina de resposta estrita em JSON de 7 chaves sem texto excedente.\n"
            "- Garantir citação exclusiva de `chunk_id`s presentes no contexto e leitura de `[DADOS_TECNICOS_ARBORGRID]`.\n"
            "- Reforçar recusa fiel (`nao_consta`) quando o contexto for insuficiente ou fora de São Paulo, além de sanitização LGPD."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(ROOT))\n\n"
            "from src.common import DIR_ADAPTERS, DIR_ASSETS, DIR_DATA, load_jsonl\n\n"
            "train_rows = load_jsonl(DIR_DATA / 'sft_train.jsonl')\n"
            "val_rows = load_jsonl(DIR_DATA / 'sft_val.jsonl')\n"
            "print(f'Treino SFT: {len(train_rows)} amostras | Validação SFT: {len(val_rows)} amostras')\n"
            "display(pd.Series([r['familia'] for r in train_rows]).value_counts().rename('qtd_treino'))"
        ),
        nbf.v4.new_code_cell(
            "REEXECUTAR_TREINO = False  # True para retreinar o adaptador LoRA r=16 na GPU (~2 min)\n\n"
            "if REEXECUTAR_TREINO or not (DIR_ADAPTERS / 'lora_arborgrid_r16' / 'adapter_config.json').exists():\n"
            "    from scripts.train_lora import main as run_train\n"
            "    run_train()\n\n"
            "cfg = json.loads((DIR_ADAPTERS / 'lora_arborgrid_r16' / 'adapter_config.json').read_text(encoding='utf-8'))\n"
            "print('Configuração LoRA aplicada:', {k: cfg[k] for k in ['r', 'lora_alpha', 'lora_dropout', 'target_modules', 'base_model_name_or_path']})\n"
            "hist = json.loads((DIR_ASSETS / 'lora_train_history.json').read_text(encoding='utf-8'))\n"
            "eval_losses = [h for h in hist if 'eval_loss' in h]\n"
            "display(pd.DataFrame(eval_losses)[['epoch', 'step', 'eval_loss']])\n"
            "display(Image(filename=str(DIR_ASSETS / 'curva_loss_lora.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "### Convergência Observada\n"
            "Pela curva acima, com 3 épocas (`69` passos efetivos) e monitoramento por época (`eval_strategy='epoch'`), a perda de validação (`eval_loss`) cai continuamente (`0,1330 -> 0,0021 -> 0,00055`) sem sinal de *overfitting*, validando o adaptador `lora_arborgrid_r16`."
        ),
    ]
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    return nb


def _nb_eval() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(
            "# 03 — Placar Comparativo das 3 Configurações, Quantização e Auditoria de Erros\n\n"
            "Compara programaticamente no conjunto de teste independente (`avaliacao/test_set.jsonl`, 30 casos):\n"
            "1. **Base (`Qwen 1.7B`) sem contexto**\n"
            "2. **Base + RAG simples** (`top-3`, limiar `0.30`)\n"
            "3. **Sistema Completo (`Qwen 1.7B` + LoRA `r=16` + RAG + Agente GeoSampa + Validador)**"
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, Markdown, display\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(ROOT))\n\n"
            "from src.common import DIR_ASSETS, DIR_EVAL\n\n"
            "REEXECUTAR_BENCHMARK = False  # True para reavaliar as 3 configurações na GPU\n"
            "if REEXECUTAR_BENCHMARK or not (DIR_EVAL / 'placar_comparativo.csv').exists():\n"
            "    from scripts.run_benchmarks import main as run_bench\n"
            "    run_bench()\n\n"
            "df_placar = pd.read_csv(DIR_EVAL / 'placar_comparativo.csv')\n"
            "display(df_placar)\n"
            "display(Image(filename=str(DIR_ASSETS / 'placar_comparativo.png')))\n"
            "display(Image(filename=str(DIR_ASSETS / 'placar_comparativo_completo.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "### Decisão Arquitetural: Adoção de `LoRA + RAG + Agente`\n\n"
            "Com o salto de **`26,7%`** (Base) e **`33,3%`** (RAG simples) para **`93,3%`** (`28/30` nos 11 quesitos avaliados), adotamos a arquitetura completa como solução lógica do copiloto e passamos a estudar sua quantização e hardware (`GPU` vs. `CPU`) visando à AWS.\n\n"
            "## Estudo Exploratório de Modelos (`SmolLM2-360M` → `Qwen 0.6B` → `Qwen 1.7B`), Quantização e Hardware\n\n"
            "Compara primeiro `SmolLM2-360M` vs. `Qwen 0.6B` vs. `Qwen 1.7B` (justificando o descarte do SmolLM2 e a adoção da família Qwen — ver `EXPLORACAO_MODELOS_E_QUANTIZACAO.md`), e em seguida avalia **`F16` (16 bits)** vs **`Q4_K_M` (4 bits)** via `llama.cpp` (`-c 2048 -np 1`) em **GPU** (`-ngl 99`) e **CPU pura** (`-dev none -ngl 0 -t 2`) "
            "frente às instâncias EC2 (`t3.small` 2 GiB vs `t3.medium` 4 GiB)."
        ),
        nbf.v4.new_code_cell(
            "if not (DIR_EVAL / 'placar_comparativo_modelos.csv').exists():\n"
            "    from scripts.build_size_quant_comparison import main as run_comp\n"
            "    run_comp()\n\n"
            "if (DIR_ASSETS / 'comparativo_smol360m_vs_qwen.png').exists():\n"
            "    display(Image(filename=str(DIR_ASSETS / 'comparativo_smol360m_vs_qwen.png')))\n"
            "df_comp = pd.read_csv(DIR_EVAL / 'placar_comparativo_modelos.csv')\n"
            "display(df_comp[['modelo_base', 'formato_execucao', 'configuracao', 'tamanho_gguf_mib', 'memoria_execucao_mib', 'acuracia_geral_estrita', 'latencia_media_ms', 'latencia_p95_ms', 'tps_medio']])\n"
            "display(Image(filename=str(DIR_ASSETS / 'matriz_2x2_memoria_acuracia.png')))\n"
            "display(Image(filename=str(DIR_ASSETS / 'matriz_2x2_latencia_gpu_cpu.png')))\n"
            "display(Image(filename=str(DIR_ASSETS / 'matriz_2x2_tps_gpu_cpu.png')))\n"
            "display(Image(filename=str(DIR_ASSETS / 'comparativo_17b_vs_06b_gguf.png')))"
        ),
        nbf.v4.new_code_cell(
            "from src.tools import ArborGridInspector\n\n"
            "inspector = ArborGridInspector()\n"
            "for gid in ['629339', '7457', '612162']:\n"
            "    ctx, rec = inspector.maybe_build_tool_context(f'Consulte a árvore GeoSampa {gid}')\n"
            "    print(ctx)"
        ),
        nbf.v4.new_code_cell(
            "audit_path = DIR_EVAL / 'auditoria_erros_reais.md'\n"
            "if not audit_path.exists():\n"
            "    from scripts.analyze_errors import main as run_audit\n"
            "    run_audit()\n"
            "display(Markdown(audit_path.read_text(encoding='utf-8')))"
        ),
    ]
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    return nb


def main() -> None:
    DIR_NB.mkdir(parents=True, exist_ok=True)
    builders = {
        "01_base_e_busca_rag.ipynb": _nb_base_rag,
        "02_treino_lora_qwen3.ipynb": _nb_lora,
        "03_avaliacao_comparativa_e_agente.ipynb": _nb_eval,
    }
    for fname, builder in builders.items():
        nb = builder()
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        out_path = DIR_NB / fname
        with out_path.open("w", encoding="utf-8") as fp:
            nbf.write(nb, fp)
        print(f"Gerado notebook: {out_path}")


if __name__ == "__main__":
    main()
