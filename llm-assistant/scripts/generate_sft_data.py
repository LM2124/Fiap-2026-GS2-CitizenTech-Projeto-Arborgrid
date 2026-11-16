"""Gerador procedural de dataset SFT (treino e validação) disjunto do conjunto de teste.

Ensina ao Qwen3-1.7B via LoRA:
1) Formatação estrita em JSON de 7 chaves sem aspas mal-escapadas ou texto extra;
2) Citação fiel apenas dos chunk_ids presentes no [CONTEXTO] (incluindo síntese multi-trecho);
3) Calibração de risco: emergencia_imediata vs urgente_vistoria vs preventivo_rotina;
4) Perguntas de permissão do cidadão ("posso contratar podador / usar escada?") -> status "respondido" com pode_podar_por_conta=false;
5) Leitura de [DADOS_TECNICOS_ARBORGRID] (incluindo DIVERGENCIA_PARALAXE) sem inventar chunk_id;
6) Recusa estrita ("nao_consta") diante de fatos/tarifas ausentes nos trechos + sanitização LGPD.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import DIR_DATA, build_user_prompt, save_jsonl

SEED = 2026

DISTRACTORS = [
    {
        "chunk_id": "DIST-PARQUE-01",
        "fonte": "Regulamento Municipal de Parques",
        "conteudo": "O horário de funcionamento regular dos parques municipais urbanos é das 6h às 18h, podendo ser alterado em feriados.",
        "score": 0.361,
    },
    {
        "chunk_id": "DIST-ILUM-02",
        "fonte": "Iluminação Pública Municipal",
        "conteudo": "Lâmpadas queimadas em postes sem relação com queda arbórea devem ser reportadas pelo portal de zeladoria urbana.",
        "score": 0.392,
    },
    {
        "chunk_id": "DIST-CALCADA-03",
        "fonte": "Estatuto de Passeios Públicos",
        "conteudo": "A recomposição de piso de calçada danificado por obra particular cabe ao titular do lote após notificação da fiscalização.",
        "score": 0.348,
    },
]


def _make_example(
    sample_id: str,
    split_family: str,
    question: str,
    chunks: list[dict[str, Any]] | None,
    tree_ctx: str | None,
    target: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "familia": split_family,
        "user_prompt": build_user_prompt(question=question, chunks=chunks, tree_context=tree_ctx),
        "target_json": json.dumps(target, ensure_ascii=False),
    }


def _gen_electrical_emergency(rng: random.Random, idx: int) -> dict[str, Any]:
    cid = f"SYN-ENERG-{rng.randint(10, 99)}"
    cid_sec = f"SYN-SEG-{rng.randint(10, 99)}"
    phone = rng.choice(["0800 72 72 196", "0800 70 10 196"])
    bairro = rng.choice(["Moema", "Vila Mariana", "Lapa", "Santana", "Tatuapé", "Butantã", "Consolação"])
    sintoma = rng.choice([
        "soltando faíscas na copa",
        "com curto-circuito e estalos altos no cabo",
        "com fio partido caído sobre o asfalto",
        "encostando em cabo energizado com fumaça",
    ])
    lgpd_prefix = ""
    if rng.random() < 0.25:
        lgpd_prefix = f"Sou morador, CPF 321.654.987-0{rng.randint(1,9)}, carro placa BRA{rng.randint(1,9)}E{rng.randint(10,99)}. "
    chunk = {
        "chunk_id": cid,
        "fonte": "Protocolo de Emergência Elétrica",
        "conteudo": f"Diante de risco iminente na rede — como galho {sintoma} — acione a emergência 24h pelo {phone}. Jamais toque em cabos.",
        "score": round(rng.uniform(0.58, 0.84), 3),
    }
    chunk_sec = {
        "chunk_id": cid_sec,
        "fonte": "Cartilha de Segurança Elétrica",
        "conteudo": f"Mantenha distância segura de isolamento em caso de cabo partido ou faiscamento e aguarde a equipe pelo {phone}.",
        "score": round(rng.uniform(0.45, 0.57), 3),
    }
    chunks = [chunk, chunk_sec, rng.choice(DISTRACTORS)]
    rng.shuffle(chunks)
    question = rng.choice([
        f"{lgpd_prefix}Em {bairro}, há uma árvore {sintoma}. Quem devo acionar agora e posso afastar o galho?",
        f"{lgpd_prefix}Emergência na rua em {bairro}: árvore {sintoma}. Posso ligar no número comercial ou qual canal usar?",
    ])
    target = {
        "status": "respondido",
        "nivel_risco": "emergencia_imediata",
        "orgao_responsavel": "Concessionária de Energia Emergência",
        "telefone_ou_canal": phone,
        "pode_podar_por_conta": False,
        "acao_imediata": f"Afaste-se da área energizada e acione imediatamente o canal de emergência {phone}. Nunca tente cortar ou tocar nos cabos.",
        "fontes": [cid, cid_sec],
    }
    return _make_example(f"SFT-{idx:04d}", "emergencia_eletrica", question, chunks, None, target)


def _gen_life_fire_climate(rng: random.Random, idx: int) -> dict[str, Any]:
    cid = f"SYN-VIDA-{rng.randint(10, 99)}"
    mode = rng.choice(["bombeiros", "carro_fogo", "defesa_civil_via", "defesa_civil_fio"])
    if mode == "bombeiros":
        chunk = {
            "chunk_id": cid,
            "fonte": "Corpo de Bombeiros (193)",
            "conteudo": "Acione imediatamente o Corpo de Bombeiros pelo telefone 193 quando houver incêndio em copa/transformador, vítimas ou iminência de desabamento sobre pessoas.",
            "score": 0.74,
        }
        q = "Há princípio de incêndio na copa perto do poste e risco sobre residências. Qual número acionar primeiro?"
        org, tel = "Corpo de Bombeiros (193)", "193"
        acao = "Ligue imediatamente para o Corpo de Bombeiros no 193 e evacue a proximidade imediata."
    elif mode == "carro_fogo":
        cid2 = f"SYN-BOMB-{rng.randint(10, 99)}"
        chunk = {
            "chunk_id": cid,
            "fonte": "Manual de Segurança Elétrica",
            "conteudo": "Em carro atingido por cabo elétrico sem fogo, fique dentro até resgate. Se pegar fogo, salte com os dois pés juntos sem tocar carroceria e solo ao mesmo tempo e afaste-se com passos curtos.",
            "score": 0.76,
        }
        chunks = [
            chunk,
            {"chunk_id": cid2, "fonte": "Corpo de Bombeiros (193)", "conteudo": "Ligue 193 em incêndio veicular ou risco de vida.", "score": 0.62},
        ]
        target = {
            "status": "respondido",
            "nivel_risco": "emergencia_imediata",
            "orgao_responsavel": "Corpo de Bombeiros (193) e Concessionária",
            "telefone_ou_canal": "193",
            "pode_podar_por_conta": False,
            "acao_imediata": "Se o veículo pegar fogo, salte com os dois pés juntos sem tocar chassi e solo simultaneamente, afaste-se com passos curtos e ligue 193.",
            "fontes": [cid, cid2],
        }
        return _make_example(
            f"SFT-{idx:04d}",
            "emergencia_vida",
            "Um cabo caiu sobre meu carro e começou a incendiar o veículo. Como devo sair?",
            chunks,
            None,
            target,
        )
    elif mode == "defesa_civil_via":
        chunk = {
            "chunk_id": cid,
            "fonte": "Defesa Civil Municipal (199)",
            "conteudo": "Em quedas consumadas de árvore obstruindo completamente a via pública após vendaval ou chuva forte, acione imediatamente a Defesa Civil pelo telefone 199.",
            "score": 0.75,
        }
        q = "Ventou forte e uma árvore caiu interditando toda a rua após a tempestade. Quem devo acionar?"
        org, tel = "Defesa Civil Municipal (199)", "199"
        acao = "Isole a área afetada e ligue imediatamente para a Defesa Civil no 199."
    else:
        cid2 = f"SYN-ENEL-{rng.randint(10, 99)}"
        chunk = {
            "chunk_id": cid,
            "fonte": "Defesa Civil Municipal (199)",
            "conteudo": "Quando uma árvore cai sobre a rede elétrica, a equipe da Prefeitura/Defesa Civil isola o local e aguarda a concessionária (0800 72 72 196) desligar a energia antes de cortar os galhos.",
            "score": 0.78,
        }
        chunks = [
            chunk,
            {"chunk_id": cid2, "fonte": "Concessionária Emergência", "conteudo": "Acione 0800 72 72 196 para desligamento de rede.", "score": 0.64},
        ]
        target = {
            "status": "respondido",
            "nivel_risco": "emergencia_imediata",
            "orgao_responsavel": "Defesa Civil (199) e Concessionária (0800 72 72 196)",
            "telefone_ou_canal": "199 e 0800 72 72 196",
            "pode_podar_por_conta": False,
            "acao_imediata": "Não se aproxime. A equipe municipal deve aguardar a concessionária (0800 72 72 196) desenergizar e aterrar a rede antes de iniciar o corte.",
            "fontes": [cid, cid2],
        }
        return _make_example(
            f"SFT-{idx:04d}",
            "emergencia_clima_rede",
            "A árvore caiu em cima dos cabos da rua; os funcionários municipais podem serrar os galhos antes da distribuidora chegar?",
            chunks,
            None,
            target,
        )

    chunks = [chunk, rng.choice(DISTRACTORS)]
    rng.shuffle(chunks)
    target = {
        "status": "respondido",
        "nivel_risco": "emergencia_imediata",
        "orgao_responsavel": org,
        "telefone_ou_canal": tel,
        "pode_podar_por_conta": False,
        "acao_imediata": acao,
        "fontes": [cid],
    }
    return _make_example(f"SFT-{idx:04d}", "emergencia_vida_clima", q, chunks, None, target)


def _gen_prohibition_and_routine(rng: random.Random, idx: int) -> dict[str, Any]:
    cid_lei = f"SYN-LEI-{rng.randint(10, 99)}"
    cid_can = f"SYN-SP-{rng.randint(10, 99)}"
    if rng.random() < 0.5:
        q = rng.choice([
            "A Prefeitura está demorando para vistoriar a calçada. Posso pagar um podador particular para cortar os galhos da via pública?",
            "Posso podar sozinho os galhos da árvore da calçada que avançam em direção ao poste sem autorização?",
            "Qual canal uso para pedir vistoria preventiva se a árvore pública está perto dos fios, mas sem faísca ou curto?",
        ])
        chunks = [
            {
                "chunk_id": cid_lei,
                "fonte": "Lei Municipal de Arborização",
                "conteudo": "Em áreas e calçadas públicas, apenas a Prefeitura (Subprefeitura) ou autorizadas podem podar; é proibido ao morador ou jardineiro particular cortar por conta própria.",
                "score": 0.71,
            },
            {
                "chunk_id": cid_can,
                "fonte": "Portal de Atendimento Municipal",
                "conteudo": "Para vistoria ou poda preventiva de árvore em logradouro público sem faíscas ou emergência ativa, abra protocolo no SP156 (156 / sp156.prefeitura.sp.gov.br).",
                "score": 0.67,
            },
        ]
        target = {
            "status": "respondido",
            "nivel_risco": "preventivo_rotina",
            "orgao_responsavel": "Prefeitura de SP / SP156",
            "telefone_ou_canal": "SP156 (156 / sp156.prefeitura.sp.gov.br)",
            "pode_podar_por_conta": False,
            "acao_imediata": "Não realize nem contrate poda por conta própria em via pública; registre ou acompanhe a solicitação no SP156 (156).",
            "fontes": [cid_lei, cid_can],
        }
        return _make_example(f"SFT-{idx:04d}", "proibicao_via_publica", q, chunks, None, target)

    prazo = rng.choice(["10 dias úteis", "15 dias úteis"])
    chunks = [
        {
            "chunk_id": cid_lei,
            "fonte": "Legislação Municipal Art. 18",
            "conteudo": f"Poda de árvore em área interna de imóvel particular exige comunicação prévia à Subprefeitura com antecedência mínima de {prazo} e laudo técnico de profissional habilitado.",
            "score": 0.74,
        },
        rng.choice(DISTRACTORS),
    ]
    q = "Preciso manejar uma árvore dentro do quintal do meu lote particular. Posso cortar direto ou qual exigência e prazo cumprir?"
    target = {
        "status": "respondido",
        "nivel_risco": "preventivo_rotina",
        "orgao_responsavel": "Proprietário mediante protocolo na Subprefeitura",
        "telefone_ou_canal": f"Subprefeitura ({prazo} de antecedência)",
        "pode_podar_por_conta": False,
        "acao_imediata": f"Protocole comunicação prévia na Subprefeitura com no mínimo {prazo} de antecedência anexando laudo de profissional habilitado.",
        "fontes": [cid_lei],
    }
    return _make_example(f"SFT-{idx:04d}", "lote_particular", q, chunks, None, target)


def _gen_urgent_inspection_and_norms(rng: random.Random, idx: int) -> dict[str, Any]:
    cid = f"SYN-NORM-{rng.randint(10, 99)}"
    subtype = rng.choice(["risco_queda", "escada_nr10", "manual_infra", "comercial_enel"])
    if subtype == "risco_queda":
        chunks = [
            {
                "chunk_id": cid,
                "fonte": "Norma Técnica de Avaliação de Risco de Queda",
                "conteudo": "Avaliação de risco de queda analisa inclinação progressiva do tronco, podridão na base ou raízes expostas, galhos secos pendentes e presença de alvos na via, orientando vistoria prioritária da Subprefeitura (156 / 199).",
                "score": 0.72,
            },
            rng.choice(DISTRACTORS),
        ]
        target = {
            "status": "respondido",
            "nivel_risco": "urgente_vistoria",
            "orgao_responsavel": "Prefeitura / Subprefeitura (SP156) ou Defesa Civil (199)",
            "telefone_ou_canal": "SP156 (156) ou Defesa Civil (199)",
            "pode_podar_por_conta": False,
            "acao_imediata": "Os critérios incluem inclinação recente, podridão basal/radicular, galhos secos e alvos na via. Solicite vistoria urgente pelo 156 ou 199.",
            "fontes": [cid],
        }
        q = "Uma árvore na via está inclinando recentemente com podridão na base das raízes. Quais fatores avaliam risco de queda e quem acionar?"
    elif subtype == "escada_nr10":
        cid2 = f"SYN-EMERG-{rng.randint(10, 99)}"
        chunks = [
            {
                "chunk_id": cid,
                "fonte": "Norma de Segurança na Arboricultura e NR-10",
                "conteudo": "Intervenção perto de linhas energizadas exige profissionais autorizados e treinados com EPIs/EPCs dielétricos e desenergização; cidadãos comuns jamais devem aproximar ferramentas.",
                "score": 0.71,
            },
            {
                "chunk_id": cid2,
                "fonte": "Emergência Concessionária",
                "conteudo": "Para riscos junto à rede energizada, ligue 0800 72 72 196 ou abra SP156 (156) para manejo seguro.",
                "score": 0.59,
            },
        ]
        target = {
            "status": "respondido",
            "nivel_risco": "urgente_vistoria",
            "orgao_responsavel": "Profissionais autorizados NR-10 / Enel ou Prefeitura",
            "telefone_ou_canal": "0800 72 72 196 ou SP156 (156)",
            "pode_podar_por_conta": False,
            "acao_imediata": "Não suba com escada nem aproxime ferramentas de fios energizados; somente equipes habilitadas segundo NR-10 podem atuar. Acione 0800 72 72 196 ou 156.",
            "fontes": [cid, cid2],
        }
        q = "Posso subir em escada com podão e luvas comuns para afastar um galho próximo aos cabos de energia?"
    elif subtype == "manual_infra":
        chunks = [
            {
                "chunk_id": cid,
                "fonte": "Manual Técnico de Poda",
                "conteudo": "Em conflitos recorrentes entre copa e rede elétrica, o Manual Técnico recomenda priorizar adequação da infraestrutura (rede compacta, isolada ou enterramento) antes de podas drásticas em V, preservando colar e crista da casca.",
                "score": 0.73,
            },
            rng.choice(DISTRACTORS),
        ]
        target = {
            "status": "respondido",
            "nivel_risco": "preventivo_rotina",
            "orgao_responsavel": "Prefeitura (SVMA/SP156) e Concessionária",
            "telefone_ou_canal": "SP156 (156)",
            "pode_podar_por_conta": False,
            "acao_imediata": "O manual prioriza adequar a rede elétrica (cabos isolados/compactos ou enterramento) em vez de poda drástica, devendo a demanda tramitar via SP156 (156).",
            "fontes": [cid],
        }
        q = "Pelo manual técnico, quando uma árvore encosta repetidamente na fiação, deve-se priorizar poda drástica em V ou adequação da rede?"
    else:
        chunks = [
            {
                "chunk_id": cid,
                "fonte": "Central Comercial da Concessionária",
                "conteudo": "Para acompanhamento de protocolos comerciais ou dúvidas não emergenciais com a distribuidora, utilize o telefone 0800 72 72 120 (nunca em emergências, cujo número é 0800 72 72 196).",
                "score": 0.79,
            },
            rng.choice(DISTRACTORS),
        ]
        target = {
            "status": "respondido",
            "nivel_risco": "preventivo_rotina",
            "orgao_responsavel": "Enel SP Relacionamento",
            "telefone_ou_canal": "0800 72 72 120",
            "pode_podar_por_conta": False,
            "acao_imediata": "Para consultas comerciais e acompanhamento não emergencial de protocolo, entre em contato pelo telefone 0800 72 72 120.",
            "fontes": [cid],
        }
        q = "Quero consultar o andamento de um protocolo antigo não emergencial com a distribuidora. Qual telefone devo discar?"

    return _make_example(f"SFT-{idx:04d}", "normas_e_risco", q, chunks, None, target)


def _gen_arborgrid_agent(rng: random.Random, idx: int) -> dict[str, Any]:
    cid = "ARBORGRID-TRIAGEM"
    cid_sp = "SP156-ROTINA"
    gid = rng.randint(910000, 999999)
    distrito = rng.choice(["MOEMA", "BROOKLIN", "LIBERDADE", "VILA MARIANA", "CONSOLACAO"])
    c_px = rng.randint(45, 420)
    w_px = rng.randint(15, 220)
    triage, risk, contact_px, wind_px, acao = rng.choice([
        ("CRITICO_CONFIRMADO", "urgente_vistoria", c_px, w_px,
         f"A árvore GeoSampa {gid} ({distrito}) possui triagem CRITICO_CONFIRMADO com sobreposição direta copa-fiação ({c_px} px). Não dispensa vistoria presencial: acione SP156 (156)."),
        ("DIVERGENCIA_PARALAXE", "urgente_vistoria", c_px, w_px,
         f"A árvore GeoSampa {gid} ({distrito}) apresenta DIVERGENCIA_PARALAXE entre visadas, exigindo cautela e vistoria presencial prioritária pelo SP156 (156)."),
        ("PROXIMIDADE", "preventivo_rotina", 0, w_px,
         f"A árvore GeoSampa {gid} ({distrito}) encontra-se em PROXIMIDADE (margem de vento={w_px} px, sem contato direto). Solicite acompanhamento preventivo via SP156 (156)."),
        ("SEGURO", "preventivo_rotina", 0, 0,
         f"A árvore GeoSampa {gid} ({distrito}) consta como SEGURO na triagem visual v3. A triagem computacional não substitui vistoria presencial via SP156 (156) se houver risco em campo."),
    ])
    chunks = [
        {
            "chunk_id": cid,
            "fonte": "Especificação Técnica ArborGrid SP",
            "conteudo": "Triagem v3: CRITICO_CONFIRMADO/CRITICO_MONO_VISADA (contact_direct_px > 0), DIVERGENCIA_PARALAXE (exige vistoria prioritária), PROXIMIDADE (wind_buffer_px > 0) e SEGURO. Não substitui vistoria presencial via SP156.",
            "score": 0.72,
        },
        {
            "chunk_id": cid_sp,
            "fonte": "Portal SP156",
            "conteudo": "Solicitações preventivas e vistorias de arborização em via pública tramitam pelo SP156 (telefone 156 / sp156.prefeitura.sp.gov.br).",
            "score": 0.61,
        },
    ]
    tree_ctx = (
        f"Consulta GeoSampa ID {gid} | Distrito: {distrito} | Subprefeitura: {distrito} | "
        f"Triagem ArborGrid v3: {triage} | max_contact_direct_px={contact_px} | max_wind_buffer_px={wind_px}."
    )
    q = f"Verifique os dados da árvore GeoSampa {gid} no ArborGrid e oriente sobre risco e canal competente."
    target = {
        "status": "respondido",
        "nivel_risco": risk,
        "orgao_responsavel": "Prefeitura de SP / SP156",
        "telefone_ou_canal": "SP156 (156 / sp156.prefeitura.sp.gov.br)",
        "pode_podar_por_conta": False,
        "acao_imediata": acao,
        "fontes": [cid, cid_sp],
    }
    return _make_example(f"SFT-{idx:04d}", "agente_geosampa", q, chunks, tree_ctx, target)


def _gen_nao_consta(rng: random.Random, idx: int) -> dict[str, Any]:
    mode = rng.choice(["empty", "near_domain_trap"])
    if mode == "empty":
        q = rng.choice([
            f"Qual será a velocidade do vento em {rng.choice(['Pinheiros', 'Moema', 'Sé'])} no próximo domingo?",
            f"Qual é a multa em reais em {rng.choice(['Recife', 'Curitiba', 'Salvador'])} por poda ilegal?",
            "Quantos centímetros de raiz um ipê precisa para adubação orgânica?",
        ])
        chunks: list[dict[str, Any]] = []
    else:
        q, chunks = rng.choice([
            (
                "Qual o valor exato em reais da taxa de protocolo na Subprefeitura de São Paulo para pedir poda?",
                [{
                    "chunk_id": "SYN-TRAP-01",
                    "fonte": "Legislação Municipal",
                    "conteudo": "A solicitação deve ser protocolada na Subprefeitura com antecedência mínima e laudo técnico, sem mencionar valores financeiros.",
                    "score": 0.51,
                }],
            ),
            (
                "Quanto cobra em média um engenheiro florestal em São Paulo para assinar um laudo com ART?",
                [{
                    "chunk_id": "SYN-TRAP-02",
                    "fonte": "Legislação Municipal",
                    "conteudo": "O laudo técnico deve ser assinado por engenheiro agrônomo, florestal ou biólogo habilitado com ART/TRT.",
                    "score": 0.49,
                }],
            ),
            (
                "Qual o telefone particular do diretor de operações da concessionária em Belo Horizonte?",
                [{
                    "chunk_id": "SYN-TRAP-03",
                    "fonte": "Central de Relacionamento",
                    "conteudo": "Os canais institucionais atendem emergências em São Paulo pelo 0800 72 72 196 e relacionamento pelo 0800 72 72 120.",
                    "score": 0.46,
                }],
            ),
        ])
    target = {
        "status": "nao_consta",
        "nivel_risco": "fora_de_escopo",
        "orgao_responsavel": "não consta",
        "telefone_ou_canal": "não consta",
        "pode_podar_por_conta": None,
        "acao_imediata": "não consta na base de conhecimento do ArborGrid a informação solicitada.",
        "fontes": [],
    }
    return _make_example(f"SFT-{idx:04d}", "nao_consta", q, chunks, None, target)


def generate_splits(n_train: int = 360, n_val: int = 60) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(SEED)
    generators = [
        _gen_electrical_emergency,
        _gen_life_fire_climate,
        _gen_prohibition_and_routine,
        _gen_urgent_inspection_and_norms,
        _gen_arborgrid_agent,
        _gen_nao_consta,
    ]
    all_rows: list[dict[str, Any]] = []
    for i in range(n_train + n_val):
        gen = generators[i % len(generators)]
        all_rows.append(gen(rng, i + 1))
    rng.shuffle(all_rows)
    return all_rows[:n_train], all_rows[n_train:]


def main() -> None:
    train_rows, val_rows = generate_splits(n_train=360, n_val=60)
    DIR_DATA.mkdir(parents=True, exist_ok=True)
    save_jsonl(DIR_DATA / "sft_train.jsonl", train_rows)
    save_jsonl(DIR_DATA / "sft_val.jsonl", val_rows)
    print(f"Gerado sft_train.jsonl ({len(train_rows)}) e sft_val.jsonl ({len(val_rows)}) em {DIR_DATA}")


if __name__ == "__main__":
    main()
