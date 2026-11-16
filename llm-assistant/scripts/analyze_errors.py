"""Extrai erros reais observados no benchmark comparativo para auditoria."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import DIR_EVAL, load_jsonl


def _describe_failure(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not row["json_valid"]:
        reasons.append("JSON inválido")
    if not row["schema_valid"]:
        reasons.append(f"Esquema inválido ({','.join(row['schema_errors'])})")
    if not row["status_exact"]:
        reasons.append("Status errado (respondido vs nao_consta)")
    if not row["risk_exact"]:
        reasons.append("Nível de risco incorreto")
    if not row["channel_exact"]:
        reasons.append("Órgão/telefone incorreto")
    if not row["prune_exact"]:
        reasons.append("Violação da regra de poda própria")
    if row["hallucinated_source"]:
        reasons.append("Citação de chunk_id inventado ou ausente no contexto")
    elif not row["source_cited"]:
        reasons.append("Não citou a fonte esperada")
    if row["hallucinated_phone"]:
        reasons.append("Telefone não fundamentado no contexto")
    if not row["lgpd_clean"]:
        reasons.append("Eco de dado pessoal sensível (LGPD)")
    return reasons


def main() -> None:
    details = json.loads((DIR_EVAL / "benchmark_detalhes.json").read_text(encoding="utf-8"))
    questions = {item["id"]: item for item in load_jsonl(DIR_EVAL / "test_set.jsonl")}

    lines = ["# Auditoria de Erros Reais no Benchmark\n"]
    for cfg_name, rows in details.items():
        failures = [r for r in rows if not r["overall_exact"]]
        lines.append(f"## {cfg_name} — {len(failures)} falha(s) em {len(rows)} casos\n")
        for r in failures:
            reasons = _describe_failure(r)
            q_text = questions[r["id"]]["pergunta"]
            exp = questions[r["id"]]["esperado"]
            lines.append(f"- **{r['id']}** (`{r['categoria']}`): {'; '.join(reasons)}")
            lines.append(f"  - Pergunta: {q_text}")
            lines.append(f"  - Esperado: `risco={exp['nivel_risco']}`, `canais={exp['canal_tokens']}`, `fontes={exp['fontes']}`")
            lines.append(f"  - Trechos recuperados: `{r.get('retrieved_ids', [])}`")
            parsed = r.get("parsed")
            if parsed:
                lines.append(
                    f"  - Obtido: `status={parsed.get('status')}`, `risco={parsed.get('nivel_risco')}`, "
                    f"`tel={parsed.get('telefone_ou_canal')}`, `fontes={parsed.get('fontes')}`"
                )
            else:
                lines.append(f"  - Saída bruta: `{str(r.get('raw_output', ''))[:160]}`")
        lines.append("")

    out_path = DIR_EVAL / "auditoria_erros_reais.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Auditoria salva em {out_path}")


if __name__ == "__main__":
    main()
