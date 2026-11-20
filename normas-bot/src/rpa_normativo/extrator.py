"""Extracao de texto de HTML e PDF, normalizacao e calculo de hash SHA-256."""

from __future__ import annotations

import hashlib
import io
import re

from bs4 import BeautifulSoup
from pypdf import PdfReader


class ErroExtracaoConteudo(Exception):
    """Erro disparado quando nao ha texto extraivel ou o arquivo esta invalido."""


def normalizar_texto(texto: str) -> str:
    """Colapsa espacos em branco e converte para minusculas."""
    return re.sub(r"\s+", " ", texto).strip().lower()


def calcular_sha256(texto_normalizado: str) -> str:
    """Calcula digest SHA-256 em hexadecimal do texto normalizado."""
    return hashlib.sha256(texto_normalizado.encode("utf-8")).hexdigest()


def extrair_texto_html(conteudo: bytes) -> str:
    """Extrai texto visivel de HTML ignorando tags script e style."""
    if not conteudo or not conteudo.strip():
        raise ErroExtracaoConteudo("Pagina HTML vazia.")

    try:
        html_str = conteudo.decode("utf-8")
    except UnicodeDecodeError:
        html_str = conteudo.decode("latin-1", errors="replace")

    sopa = BeautifulSoup(html_str, "html.parser")
    for tag in sopa(["script", "style"]):
        tag.decompose()

    texto_norm = normalizar_texto(sopa.get_text(" "))
    if not texto_norm:
        raise ErroExtracaoConteudo("Pagina HTML sem texto visivel extraivel.")
    return texto_norm


def extrair_texto_pdf(conteudo: bytes) -> str:
    """Extrai texto de todas as paginas de um PDF usando pypdf."""
    if not conteudo:
        raise ErroExtracaoConteudo("Arquivo PDF vazio.")

    try:
        leitor = PdfReader(io.BytesIO(conteudo))
        partes = [(pagina.extract_text() or "") for pagina in leitor.pages]
    except Exception as exc:
        raise ErroExtracaoConteudo(f"Falha ao ler PDF: {type(exc).__name__}") from exc

    texto_norm = normalizar_texto(" ".join(partes))
    if not texto_norm:
        raise ErroExtracaoConteudo("PDF sem texto extraivel (possivelmente escaneado).")
    return texto_norm


def extrair_e_hashear(conteudo: bytes, tipo: str) -> tuple[str, str]:
    """Extrai texto conforme o tipo da fonte (html ou pdf) e retorna (texto_normalizado, sha256)."""
    if tipo == "html":
        texto = extrair_texto_html(conteudo)
    elif tipo == "pdf":
        texto = extrair_texto_pdf(conteudo)
    else:
        raise ErroExtracaoConteudo(f"Tipo de fonte nao suportado para extracao: {tipo}")
    return texto, calcular_sha256(texto)
