"""Configuração fixa do ArborGrid - alinhada às 4 regiões da API de Visão Computacional."""

# Regiões auditadas pela API ArborGrid (mesmo region_key da API).
# lat/lon = centro da região, usado para pedir o clima à Open-Meteo e posicionar as denúncias simuladas.
# Pinheiros e Alto da Mooca: média das coordenadas das árvores reais da API.
# Vila Mariana e Av. Paulista: centro aproximado da região.
REGIOES = [
    {"region_key": "pinheiros",        "regiao": "Pinheiros (Zona Oeste)",       "lat": -23.5630, "lon": -46.6857},
    {"region_key": "alto_da_mooca",    "regiao": "Alto da Mooca (Zona Leste)",   "lat": -23.5616, "lon": -46.5928},
    {"region_key": "vila_mariana",     "regiao": "Vila Mariana (Zona Sul)",      "lat": -23.5890, "lon": -46.6350},
    {"region_key": "avenida_paulista", "regiao": "Avenida Paulista (controle)",  "lat": -23.5614, "lon": -46.6558},
]

# Identificação do projeto nas chamadas às APIs
USER_AGENT = "ArborGrid-FIAP-GS2026/1.0 (projeto academico)"

# Endereços das APIs públicas
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
VARIAVEIS_CLIMA = ["wind_gusts_10m", "wind_speed_10m", "precipitation"]

# API ArborGrid (Visão Computacional): a URL e o token ficam em Variables do Airflow.
# De dentro do container do Airflow, a API que roda no seu PC é "host.docker.internal".
ARVORES_ENDPOINT = "/api/v1/inspections"
ARVORES_LIMITE_PAGINA = 1000
