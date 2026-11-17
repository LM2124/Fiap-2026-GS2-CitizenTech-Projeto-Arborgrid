# Declaração e Governança de Fontes de Dados — Projeto ArborGrid

- Instituição: FIAP — Engenharia de Software (4º Ano / 4ES)
- Desafio: Global Solution 2026.2 — *CitizenTech*

Este documento registra as fontes de dados públicas, APIs externas e datasets utilizados (ou avaliados durante a etapa exploratória) no ecossistema ArborGrid, garantindo rastreabilidade acadêmica e transparência metodológica entre os módulos (*Computer Vision, Generative AI, Big Data e Web/Cloud*).

---

## 1. Fontes Primárias em Uso Ativo

### 1.1. Prefeitura do Município de São Paulo — GeoSampa / SISGAU (Arborização Viária)
- **Órgão Responsável:** Secretaria Municipal do Verde e do Meio Ambiente (SVMA) / GEOINFO - PMSP.
- **Portal Oficial:** [https://geosampa.prefeitura.sp.gov.br/](https://geosampa.prefeitura.sp.gov.br/)
- **Serviço / Endpoint (OGC WFS):**
  - URL: `http://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs`
  - Camada (`typeName`): `geoportal:arvore` (*Indivíduos Arbóreos — Arborização Viária / SISGAU*)
- **Sistema Geodésico:** Originalmente projetado em `SIRGAS 2000 / UTM Zone 23S (EPSG:31983)`, convertido via `srsName=EPSG:4326` (`WGS84`) na ingestão automatizada.
- **Base Legal:** Dados Abertos Governamentais — Decreto Municipal nº 57.576/2017 e Lei de Acesso à Informação (Lei nº 12.527/2011). Uso livre com citação da fonte oficial.
- **Papel no Projeto:**
  - *Visão Computacional:* Fornece coordenadas `(lat, lon)` e identificador municipal (`cd_identificador`) para as `7.912` árvores cadastrais baixadas nos recortes de `geosampa_cadastral/*.geojson` (`3.356` em Pinheiros, `1.801` no Alto da Mooca e `2.755` na Vila Mariana), servindo de base para amostragem de campo e visualização no mapa interativo.
  - *Big Data e Priorização:* Evidencia o déficit de vistoria manual no cadastro público (campos `dt_inspecao` e `dc_tipo_estado_geral` ausentes em `88,2%` das `271` árvores auditadas em campo), cruzado com os scores de risco da IA.
- **Citação:**
  > SÃO PAULO (Município). Prefeitura do Município de São Paulo. *GeoSampa: Camada de Arborização Viária (`geoportal:arvore` / SISGAU)*. São Paulo: SVMA, 2026. Disponível em: https://geosampa.prefeitura.sp.gov.br/. Acesso em: nov. 2026.

---

### 1.2. Google Maps Platform — Street View Metadata API & Street View Static API
- **Provedor:** Google LLC (Google Maps Platform).
- **Documentação Oficial:** [https://developers.google.com/maps/documentation/streetview/overview](https://developers.google.com/maps/documentation/streetview/overview)
- **Endpoints Utilizados:**
  - Metadados (custo zero): `https://maps.googleapis.com/maps/api/streetview/metadata`
  - Imagem estática direcional: `https://maps.googleapis.com/maps/api/streetview`
- **Termos de Uso e Escopo:**
  - Atribuição obrigatória em exibições visuais e relatórios: `Imagens © Google`.
  - Uso estritamente acadêmico sem fins comerciais (Prova de Conceito FIAP Global Solution), com registro de proveniência (`pano_id`, `capture_date`, `heading`, `pitch`, `fov`) em `manifest.json`.
  - As imagens brutas coletadas ficam excluídas do controle de versão via `.gitignore`, permanecendo versionados apenas exemplos ilustrativos com overlays em `assets/` e 4 amostras do MVP em `webapp/data/samples/`.
- **Citação:**
  > GOOGLE. *Google Maps Platform: Street View Static API & Metadata API*. Mountain View: Google LLC, 2026. Imagens de nível de rua do município de São Paulo (© Google). Disponível em: https://developers.google.com/maps/documentation/streetview. Acesso em: nov. 2026.

---

### 1.3. Dataset Autoral Anotado — *ArborGrid SP Canopy-Wire Segmentation*
- **Autoria:** Equipe CitizenTech 4ES — FIAP Global Solution 2026.2.
- **Descrição:** Conjunto multi-região de `143` fotografias anotadas em polígonos (`pacote_1/`, `pacote_2/` e `pacote_3_hard_negatives/`) + `20` de teste cego na Av. Paulista, dentro da amostra operacional de `271` árvores (`536` vistas) cobrindo Pinheiros, Alto da Mooca, Vila Mariana e Av. Paulista para segmentação multi-label de copas (`copa`), fiação aérea (`fiacao`) e postes (`poste`).

---

### 1.4. Esri / ArcGIS Online — Mapas-Base Cartográficos (Frontend Web)
- **Provedor:** Esri — *ArcGIS REST MapServer Services* (`World_Topo_Map`, `World_Street_Map` e `World_Imagery`) renderizados via `Leaflet.js` (`BSD-2-Clause`).
- **Atribuição Exibida no Mapa:** `Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, HERE, Garmin, OpenStreetMap contributors, and the GIS User Community`.
- **Papel no Projeto:** Camadas cartográficas e de satélite de fundo para inspeção geoespacial no mapa de vistorias.

---

## 2. Fontes Complementares Avaliadas (Etapa Exploratória)

Registram-se abaixo datasets públicos testados na fase exploratória e os motivos técnicos pelos quais não foram adotados como fonte principal:

### 2.1. Mapillary Platform (`CC BY-SA 4.0`)
- **URL:** [https://www.mapillary.com/](https://www.mapillary.com/)
- **Avaliação:** Boa cobertura em avenidas principais por câmeras frontais veiculares (*dashcam*), porém cobertura esparsa de panorâmicas `360°` em vias secundárias residenciais de São Paulo, limitando o apontamento lateral e vertical necessário para enquadrar copas e fiação elétrica.

### 2.2. *PL-YOLOv8 / Power Line Vegetation Dataset*
- **Origem:** Dataset acadêmico associado a artigo em *IEEE Transactions on Power Delivery* (`17.176` imagens de linhas de transmissão e `1.390` de vegetação).
- **Motivo do Descarte:** Perspectiva majoritariamente aérea/drone sobre torres de alta tensão rurais, incompatível com fotos ao nível da calçada em ambiente urbano.

### 2.3. Roboflow Universe — *Road-Side Pole-Tree Detection (v3)* (`CC BY 4.0`)
- **URL:** [https://universe.roboflow.com/object-detection-coa6q/road-side-pole-tree-detection/dataset/3](https://universe.roboflow.com/object-detection-coa6q/road-side-pole-tree-detection/dataset/3)
- **Avaliação:** Útil apenas para validar o ambiente local de treinamento em GPU (`RTX 3060 Ti`, `mAP@50 = 65,5%` em 5 épocas); entretanto, a classe `Road-side Tree` marcava somente troncos verticais sem delimitar a copa foliar. Substituído pela base própria georreferenciada via GeoSampa.

