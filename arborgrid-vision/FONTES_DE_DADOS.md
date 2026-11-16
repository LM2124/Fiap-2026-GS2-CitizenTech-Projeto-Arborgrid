# Declaração e Governança de Fontes de Dados — Projeto ArborGrid

**Instituição:** FIAP — Engenharia de Software (4º Ano / 4ES)  
**Desafio:** Global Solution 2026.2 — *CitizenTech*  
**Última atualização:** 15/11/2026  

Este documento registra todas as fontes de dados públicas, APIs externas e datasets utilizados (ou avaliados durante a fase exploratória) no ecossistema **ArborGrid**, garantindo rastreabilidade acadêmica, conformidade de licenciamento e transparência metodológica em todas as disciplinas integradas (*Computer Vision, Generative AI, Big Data, Business Intelligence, RPA e Cloud Computing*).

---

## 1. Fontes Primárias em Uso Ativo

### 1.1. Prefeitura do Município de São Paulo — GeoSampa / SISGAU (Arborização Viária)
* **Órgão Responsável:** Secretaria Municipal do Verde e do Meio Ambiente (SVMA) / Coordenadoria de Produção e Análise de Informação (GEOINFO - PMSP).
* **Portal Oficial:** [https://geosampa.prefeitura.sp.gov.br/](https://geosampa.prefeitura.sp.gov.br/)
* **Serviço / Endpoint Técnico (OGC WFS):**
  * URL: `http://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs`
  * Camada (`typeName`): `geoportal:arvore` (*Indivíduos Arbóreos — Arborização Viária / SISGAU*)
* **Sistema de Referência Geodésico:** Originalmente projetado em `SIRGAS 2000 / UTM Zone 23S (EPSG:31983)`, convertido via parâmetro `srsName=EPSG:4326` (`WGS84`) na ingestão automatizada.
* **Base Legal e Licença:** Dados Abertos Governamentais — **Decreto Municipal nº 57.576/2017** (Política Municipal de Dados Abertos e Transparência Ativa da Prefeitura de São Paulo) e Lei de Acesso à Informação (Lei Federal nº 12.527/2011). Uso livre com citação da fonte oficial.
* **Papel no Projeto:**
  * **Computer Vision:** Fornece as coordenadas exatas `(lat, lon)` e o identificador único municipal (`cd_identificador`) de cada árvore viária mapeada em múltiplos bairros de São Paulo (ex.: **13.100 árvores** em Pinheiros, **8.719** em Mooca/Água Rasa, **13.201** no Campo Belo, **11.376** em Moema), incluindo a camada integral de pontos governamentais (`geosampa_cadastral/*.geojson`) exibida no mapa interativo.
  * **Big Data, RPA e Business Intelligence:** Base tabular/geoespacial oficial demonstrando a lacuna operacional de vistoria manual (campos do SISGAU como `dt_inspecao` e `dc_tipo_estado_geral` majoritariamente nulos — ex.: `94,4%` sem vistoria no recorte de Pinheiros e `100,0%` sem vistoria nos recortes do Alto da Mooca e da Vila Mariana), cruzada com os índices de risco gerados pela IA.
* **Como citar no trabalho/relatório:**
  > SÃO PAULO (Município). Prefeitura do Município de São Paulo. *GeoSampa: Camada de Arborização Viária (`geoportal:arvore` / SISGAU)*. São Paulo: Secretaria Municipal de Urbanismo e Licenciamento / SVMA, 2026. Disponível em: https://geosampa.prefeitura.sp.gov.br/. Acesso em: nov. 2026.

---

### 1.2. Google Maps Platform — Street View Metadata API & Street View Static API
* **Provedor:** Google LLC (Google Cloud / Google Maps Platform).
* **Documentação Oficial:** [https://developers.google.com/maps/documentation/streetview/overview](https://developers.google.com/maps/documentation/streetview/overview)
* **Endpoints Utilizados:**
  * Metadados (custo zero): `https://maps.googleapis.com/maps/api/streetview/metadata`
  * Imagem estática direcional: `https://maps.googleapis.com/maps/api/streetview`
* **Termos de Uso e Escopo:**
  * Atribuição obrigatória em exibições visuais e relatórios: **`Imagens © Google`**.
  * Utilizado estritamente em **escopo acadêmico/universitário sem fins comerciais** (Prova de Conceito FIAP Global Solution) dentro da cota gratuita *Essentials* (até 10.000 requisições/mês), com registro de proveniência (`pano_id`, `capture_date`, `heading`, `pitch`, `fov`) em manifesto local (`manifest.json`).
  * *Nota de Arquitetura para Produção:* Para eventual publicação aberta de pesos comerciais ou redistribuição pública irrestrita de imagens, prevê-se substituição/complementação por captura cidadã direta via aplicativo mobile do ArborGrid e/ou sequências panorâmicas sob licença aberta do **Mapillary API v4 (`CC BY-SA 4.0`)**.
* **Como citar no trabalho/relatório:**
  > GOOGLE. *Google Maps Platform: Street View Static API & Metadata API*. Mountain View: Google LLC, 2026. Imagens de nível de rua do município de São Paulo (© Google). Disponível em: https://developers.google.com/maps/documentation/streetview. Acesso em: nov. 2026.

---

### 1.3. Dataset Autoral Anotado — *ArborGrid SP Canopy-Wire Segmentation*
* **Autoria:** Equipe 4ES — FIAP Global Solution 2026.2.
* **Descrição:** Conjunto curado multi-região de imagens ao nível da calçada em São Paulo (`143` fotografias anotadas nos Pacotes 1, 2 e 3 + `20` de teste cego na Av. Paulista, dentro de uma amostra operacional de `271` árvores / `536` vistas cobrindo **Pinheiros**, **Alto da Mooca**, **Vila Mariana** e o corredor de controle subterrâneo da **Avenida Paulista**) rotulado no X-AnyLabeling com polígonos para segmentação semântica multi-label de **copas arbóreas (`copa`)**, **feixes de fiação aérea (`fiacao`)** e **postes de distribuição (`poste`)**.

---

### 1.4. Esri / ArcGIS Online — Mapas-Base Cartográficos e Ortofotos de Satélite (Frontend Web)
* **Provedor:** Environmental Systems Research Institute (Esri) — *ArcGIS REST MapServer Services* (`World_Topo_Map`, `World_Street_Map` e `World_Imagery`) renderizados via biblioteca open-source **Leaflet.js (`BSD-2-Clause`)**.
* **Atribuição Exibida no Mapa:** `Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, HERE, Garmin, OpenStreetMap contributors, and the GIS User Community`.
* **Papel no Projeto:** Camadas visuais de fundo para inspeção geoespacial e comparação direta entre a cobertura arbórea real vista por satélite e o cadastro oficial do GeoSampa.

---

## 2. Fontes Complementares e Referências Avaliadas (Histórico Metodológico)

Para fins de rigor científico no relatório técnico de *Computer Vision*, registram-se abaixo os datasets públicos inspecionados durante a etapa exploratória e os motivos técnicos de sua substituição ou uso parcial:

### 2.1. Mapillary Platform (`CC BY-SA 4.0`) — *[Referência Comparativa Open-Data]*
* **URL:** [https://www.mapillary.com/](https://www.mapillary.com/)
* **Avaliação Técnica:** Plataforma colaborativa aberta excelente para auditoria de pavimento/calçadas (câmeras frontais veiculares *dashcam*), porém com cobertura esparsa de câmeras esféricas `360°` em vias secundárias residenciais de São Paulo, limitando o apontamento lateral e inclinação superior (`pitch +15°`) necessários para enquadrar copas altas e rede elétrica de distribuição.

### 2.2. *PL-YOLOv8 / Power Line Vegetation Dataset* (`image640.zip` & `images_vegmetric.zip`) — *[Descartado na Etapa 1]*
* **Origem:** Dataset acadêmico associado a artigo em *IEEE Transactions on Power Delivery* (17.176 imagens de linhas de transmissão + 1.390 imagens de métricas de vegetação).
* **Motivo do Descarte:** Inspeção visual (`mosaico_amostras_poc.jpg`) revelou perspectiva majoritariamente aérea/drone sobre torres de alta tensão rurais, incompatível com o domínio urbano de fotos tiradas da calçada em São Paulo.

### 2.3. Roboflow Universe — *Road-Side Pole-Tree Detection (v3)* — *[Usado Apenas em Teste de Infraestrutura GPU]*
* **URL:** [https://universe.roboflow.com/object-detection-coa6q/road-side-pole-tree-detection/dataset/3](https://universe.roboflow.com/object-detection-coa6q/road-side-pole-tree-detection/dataset/3)
* **Licença:** Creative Commons Attribution 4.0 (`CC BY 4.0`) — Workspace `object-detection-coa6q` (2.243 imagens).
* **Avaliação Técnica:** Validou com sucesso o pipeline de treinamento local em GPU (`RTX 3060 Ti`, `mAP@50 = 65,5%` em 5 épocas), porém a classe `Road-side Tree` delimitava apenas os **troncos** verticais das árvores, não permitindo medir invasão da **copa foliar** nos cabos aéreos. Substituído pelo pipeline georreferenciado GeoSampa + segmentação de copas e feixes.

