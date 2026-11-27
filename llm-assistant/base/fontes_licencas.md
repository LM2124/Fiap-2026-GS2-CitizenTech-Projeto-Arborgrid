# Fontes Curadas e Licenciamento (`base/`)

Todos os trechos da base de conhecimento (`knowledge_base.jsonl`) possuem origem verificada, escopo explícito e rastreabilidade por identificador (`chunk_id`).

| IDs de Trechos | Documento / Órgão Emissor | Formato Original | Licença / Base Legal de Uso | Qtd. Trechos |
| :--- | :--- | :--- | :--- | :---: |
| `LEI-17794-ART05`, `LEI-17794-ART18`, `DEC-61859-PRZ` | Prefeitura de São Paulo — Lei Municipal nº 17.794/2022 e Decreto Municipal nº 61.859/2022 | HTML / Legislação Oficial | Domínio público legal (atos normativos oficiais, art. 8º, IV da Lei 9.610/1998) | 3 |
| `SP156-ROTINA` | Prefeitura de São Paulo — Portal SP156 | HTML / Portal Público | Dados públicos institucionais PMSP (finalidade informativa e acadêmica) | 1 |
| `SP156-PODA-PRAZO`, `SP156-INDENIZACAO`, `SP156-AREA-INTERNA`, `SP156-DENUNCIA-PODA`, `SP156-PLANTIO` | Prefeitura de São Paulo — páginas de serviço do Portal SP156 (avaliação de poda, reparação de danos), lista de serviços das Subprefeituras e notícia sobre plantio em calçada (12/12/2023) | HTML / Portal Público | Dados públicos institucionais PMSP (finalidade informativa e acadêmica) | 5 |
| `LEI-9605-ART49` | Presidência da República — Lei Federal nº 9.605/1998, art. 49 (com o § 2º da Lei nº 15.299/2025) | HTML / Legislação Oficial (planalto.gov.br) | Domínio público legal (atos normativos oficiais, art. 8º, IV da Lei 9.610/1998) | 1 |
| `SVMA-MANUAL-PODA` | Secretaria Municipal do Verde e do Meio Ambiente (PMSP/SVMA) — Manual Técnico de Poda de Árvores | PDF institucional | Material técnico público governamental (citação educativa sem fins comerciais) | 1 |
| `ENEL-EMERG-01`, `ENEL-RELAC-02`, `ENEL-PODA-REDE` | Enel Distribuição São Paulo — Canais Oficiais de Emergência e Relacionamento e página sobre podas em São Paulo | HTML / Portal Oficial | Informação pública de utilidade e segurança elétrica (uso informativo) | 3 |
| `ANEEL-167` | ANEEL — Registrar reclamação sobre problema não resolvido pela distribuidora (gov.br) | HTML / Portal gov.br | Informação pública de serviço federal (uso informativo) | 1 |
| `BOMBEIROS-193`, `DEFESA-CIVIL-199`, `SEGURANCA-FIACAO`, `DEFESA-CIVIL-ALERTAS` | Corpo de Bombeiros PMESP (193), Defesa Civil SP (199 e alertas por SMS 40199) e Cartilhas de Segurança com Rede Elétrica | HTML / Manuais de Defesa Civil e gov.br | Informação pública de proteção civil e prevenção de acidentes | 4 |
| `NBR-16246-1`, `NBR-16246-2`, `NBR-16246-3` | ABNT NBR 16246 (Partes 1, 2 e 3 — Arborização Urbana) e NR-10 (MTE) | Resumo escrito pela equipe | A NBR 16246 é uma norma paga da ABNT: a base não reproduz o texto, só resumos da equipe que citam a norma. A NR-10 é pública (MTE) | 3 |
| `ARBORGRID-TRIAGEM`, `ARBORGRID-LGPD`, `CHANGELOG-2026-ARBOR` | Projeto ArborGrid SP (Visão Computacional + API GeoSampa/SISGAU + Lei nº 13.709/2018) e a Nota Operacional Integrada v1.2 | Texto próprio da equipe | Autoria própria (MIT — repositório acadêmico FIAP CitizenTech) | 3 |

**Total:** 25 trechos curados (`base/knowledge_base.jsonl`).

O `CHANGELOG-2026-ARBOR` não é documento da Prefeitura: é uma regra operacional escrita pela equipe, que junta os canais oficiais dos outros trechos para desempatar os casos (SP156 para rotina, Enel `0800 72 72 196` para risco elétrico, `193` para fogo ou feridos).
