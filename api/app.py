"""API do AgroConecta IA.

Rodar (na raiz do projeto):  python -m uvicorn api.app:app --reload
Documentação interativa:     http://127.0.0.1:8000/docs
"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agroconecta import rag  # noqa: E402

# LGPD: o log registra só decisões e tipos de dado pessoal encontrados, nunca o texto da pergunta
log = logging.getLogger("agroconecta")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

estado: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("carregando base e modelo de embeddings (%s)...", rag.EMB_MODEL)
    estado["buscador"] = rag.Buscador()
    log.info("pronto: %d trechos, limiar %.3f, LLM %s",
             len(estado["buscador"].base), rag.LIMIAR_PADRAO, rag.LLM_MODEL)
    yield


app = FastAPI(
    title="AgroConecta IA",
    description="Assistente para pequenos produtores rurais sobre energia e escoamento da produção. "
                "Responde só com base em fontes oficiais e diz 'não consta' quando a base não cobre a pergunta.",
    version="2.0",
    lifespan=lifespan,
)


class Pergunta(BaseModel):
    pergunta: str = Field(..., min_length=3, max_length=500,
                          examples=["Existe desconto na conta de luz para quem irriga a lavoura?"])


class Explicacao(BaseModel):
    score_similaridade: float | None
    limiar: float
    decisao: str
    trecho_usado: str | None


class Resposta(BaseModel):
    resposta: str
    fonte: str | None
    url: str | None
    id_trecho: int | None
    explicacao: Explicacao


@app.get("/health")
def health():
    return {"api": "ok", "ollama": rag.ollama_disponivel(), "modelo": rag.LLM_MODEL,
            "trechos": len(estado["buscador"].base) if "buscador" in estado else 0}


@app.post("/chat", response_model=Resposta)
def chat(q: Pergunta):
    _, tipos_pii = rag.mascarar_pii(q.pergunta)
    if tipos_pii:
        log.info("dados pessoais removidos da pergunta: %s", tipos_pii)
    try:
        r = rag.config_agente(q.pergunta, estado["buscador"])
    except requests.RequestException:
        raise HTTPException(status_code=503,
                            detail="LLM local indisponível. Rode 'ollama serve' e 'ollama pull llama3.2:1b'.")
    log.info("decisão: %s | trecho=%s | chamadas LLM=%s", "recusa" if r.id_trecho is None else "resposta",
             r.id_trecho, r.tentativas)
    trecho = next((d["texto"] for d in estado["buscador"].base if d["id"] == r.id_trecho), None)
    return Resposta(
        resposta=r.resposta, fonte=r.fonte, url=r.url, id_trecho=r.id_trecho,
        explicacao=Explicacao(score_similaridade=r.score, limiar=rag.LIMIAR_PADRAO,
                              decisao=r.motivo, trecho_usado=trecho),
    )
