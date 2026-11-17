"""Pipeline de IA do AgroConecta, compartilhado pelo notebook e pela API.

Três configurações são avaliadas:
  A) LLM sem contexto  -> o modelo responde só com o que "sabe"
  B) RAG simples       -> top-1 da busca vai no prompt, sem limiar, sem JSON
  C) Agente RAG        -> limiar de recusa + top-3 + JSON com id do trecho
                          + validação da citação + nova tentativa
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

RAIZ = Path(__file__).resolve().parent.parent
BASE_PATH = RAIZ / "base" / "conhecimento.json"

EMB_MODEL = os.getenv("AGRO_EMB_MODEL", "intfloat/multilingual-e5-small")
LLM_MODEL = os.getenv("AGRO_LLM_MODEL", "llama3.2:1b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# limiar escolhido no conjunto de calibração (notebook, seção 3 / avaliacao/limiar.csv)
LIMIAR_PADRAO = float(os.getenv("AGRO_LIMIAR", "0.825"))
NAO_CONSTA = "não consta"


# ---------------------------------------------------------------- LGPD
_PII = [
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("TELEFONE", re.compile(r"(?:\(?\b\d{2}\)?\s?)?\b9?\d{4}-?\d{4}\b")),
    ("CEP", re.compile(r"\b\d{5}-\d{3}\b")),
]


def mascarar_pii(texto: str) -> tuple[str, list[str]]:
    """Troca dados pessoais (CPF, e-mail, telefone, CEP) por marcadores
    antes de qualquer processamento. Retorna o texto mascarado e os tipos
    encontrados (sem os valores)."""
    achados = []
    for nome, padrao in _PII:
        if padrao.search(texto):
            achados.append(nome)
            texto = padrao.sub(f"[{nome}]", texto)
    return texto, achados


def minimizar(pergunta: str) -> tuple[str, list[str]]:
    """Minimização (LGPD art. 6º, III): mascara os dados pessoais e descarta as
    frases que continham esses dados, mandando à busca e à LLM só a dúvida.
    Os marcadores [CPF]/[TELEFONE] confundiam o modelo 1B (README, histórico)."""
    mascarado, achados = mascarar_pii(pergunta)
    if not achados:
        return pergunta, []
    frases = re.split(r"(?<=[.!?])\s+", mascarado)
    limpas = [f for f in frases if not re.search(r"\[(?:CPF|EMAIL|TELEFONE|CEP)\]", f)]
    return (" ".join(limpas).strip() or mascarado), achados


# ---------------------------------------------------------------- busca
def carregar_base(caminho: Path = BASE_PATH) -> list[dict]:
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


class Buscador:
    """Busca vetorial por similaridade do cosseno (embeddings normalizados)."""

    def __init__(self, base: list[dict] | None = None, modelo: str = EMB_MODEL):
        from sentence_transformers import SentenceTransformer

        self.base = base if base is not None else carregar_base()
        self.nome_modelo = modelo
        self.modelo = SentenceTransformer(modelo)
        # modelos da família E5 foram treinados com os prefixos "query:" e "passage:"
        self._pq, self._pp = ("query: ", "passage: ") if "e5" in modelo.lower() else ("", "")
        textos = [f"{self._pp}{d['titulo']}. {d['texto']}" for d in self.base]
        self.emb = self.modelo.encode(textos, normalize_embeddings=True)

    def buscar(self, pergunta: str, k: int = 3) -> list[dict]:
        q = self.modelo.encode([self._pq + pergunta], normalize_embeddings=True)[0]
        sims = self.emb @ q
        ordem = np.argsort(-sims)[:k]
        return [{**self.base[i], "score": float(sims[i])} for i in ordem]




# ---------------------------------------------------------------- LLM
def gerar(prompt: str, temperatura: float = 0.0, timeout: int = 120) -> str:
    corpo = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        # num_predict limita o tamanho da saída (no modo JSON o 1B chegou a gerar
        # espaços em branco sem parar até estourar o timeout - README, erro 3)
        "options": {"temperature": temperatura, "seed": 42, "num_predict": 200},
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=corpo, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"].strip()


def ollama_disponivel() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/version", timeout=3).ok
    except requests.RequestException:
        return False


# ---------------------------------------------------------------- configurações
@dataclass
class Resultado:
    resposta: str
    fonte: str | None = None
    url: str | None = None
    id_trecho: int | None = None
    score: float | None = None
    json_valido: bool | None = None
    tentativas: int = 1
    motivo: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def to_json(self) -> str:
        """Contrato de saída do agente (o que a API devolve)."""
        return json.dumps({"resposta": self.resposta, "fonte": self.fonte, "url": self.url,
                           "id_trecho": self.id_trecho}, ensure_ascii=False)


PROMPT_SEM_CONTEXTO = """Você é um assistente para pequenos produtores rurais do Brasil.
Responda à pergunta em português, de forma curta e direta.

Pergunta: {pergunta}
Resposta:"""


def config_sem_contexto(pergunta: str) -> Resultado:
    """A) Mesmo modelo, sem nenhum trecho da base (baseline puro).

    Obs.: com a instrução extra "se não souber, diga não consta" o llama3.2:1b
    recusou 38 de 38 perguntas, então essa versão não serve como baseline
    (README, erro 1)."""
    return Resultado(resposta=gerar(PROMPT_SEM_CONTEXTO.format(pergunta=pergunta)))


PROMPT_RAG_SIMPLES = """Você é um assistente para pequenos produtores rurais do Brasil.
Use o texto abaixo para responder à pergunta em português, de forma curta e direta.
Se a resposta não estiver no texto, responda exatamente: não consta

Texto: {texto}

Pergunta: {pergunta}
Resposta:"""


def config_rag_simples(pergunta: str, buscador: Buscador) -> Resultado:
    """B) RAG simples: sempre injeta o top-1, sem limiar, sem verificação e sem citação."""
    doc = buscador.buscar(pergunta, k=1)[0]
    resposta = gerar(PROMPT_RAG_SIMPLES.format(texto=doc["texto"], pergunta=pergunta))
    return Resultado(resposta=resposta, id_trecho=doc["id"], score=doc["score"])


PROMPT_AGENTE = """Você é o AgroConecta, assistente para pequenos produtores rurais do Brasil.
Leia o trecho oficial abaixo e responda à pergunta usando apenas informações do trecho.

Trecho: {texto}

Pergunta: {pergunta}

Responda em português, com uma ou duas frases completas que expliquem a resposta usando as palavras do trecho.
Copie números e valores exatamente como aparecem no trecho.
Resposta:"""


# ---- verificador de fundamentação (anti-alucinação) ------------------------
_STOP = set("""a o as os um uma uns umas de do da dos das no na nos nas em por para com sem
que se e ou ao aos à às é são ser foi como mais menos muito pode podem deve devem
isso esse essa este esta seu sua seus suas pelo pela pelos pelas também já não sim
quando onde qual quais quem até entre sobre após depois antes cada mesmo outro outra""".split())
_POR_EXTENSO = {"dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6,
                "sete": 7, "oito": 8, "nove": 9, "dez": 10}


def _sem_acento(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _numeros(s: str) -> set[float]:
    """Valores numéricos do texto: '30.000', '30 mil', '8,5' e 'três' viram números."""
    s = _sem_acento(s)
    vals = set()
    for m in re.finditer(r"(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)(\s*mil\b)?", s):
        n = float(m.group(1).replace(".", "").replace(",", "."))
        vals.add(n * 1000 if m.group(2) else n)
    vals.update(v for w, v in _POR_EXTENSO.items() if re.search(rf"\b{w}\b", s))
    return vals


def _radicais(s: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]+", _sem_acento(s)) if len(w) > 3 and w not in _STOP}


def verificar_fundamentacao(resposta: str, trecho: str, pergunta: str,
                            min_cobertura: float = 0.6) -> tuple[bool, str]:
    """Aprova a resposta só se (1) todo número dela aparece no trecho ou na
    pergunta e (2) pelo menos 60% das palavras de conteúdo vêm do trecho/pergunta."""
    if re.search(r"[{}\[\]]", resposta):
        return False, "resposta contém código/JSON em vez de texto"
    fonte = trecho + " " + pergunta
    inventados = _numeros(resposta) - _numeros(fonte)
    if inventados:
        return False, f"números fora do trecho: {sorted(inventados)}"
    palavras = _radicais(resposta)
    if not palavras:
        return False, "resposta sem conteúdo (ex.: só 'sim' ou 'não')"
    cobertura = len(palavras & _radicais(fonte)) / len(palavras)
    if cobertura < min_cobertura:
        return False, f"cobertura {cobertura:.0%} < {min_cobertura:.0%}"
    return True, f"cobertura {cobertura:.0%}"


def config_agente(pergunta: str, buscador: Buscador, limiar: float = LIMIAR_PADRAO,
                  k: int = 3, tentativas_por_trecho: int = 2) -> Resultado:
    """C) Agente RAG: buscar -> gerar -> verificar -> corrigir / próximo trecho.

    1. Mascara dados pessoais (LGPD).
    2. Porteiro: se o melhor trecho tem score < limiar, recusa sem chamar a LLM.
    3. Para cada trecho candidato (score >= limiar, até k): gera a resposta.
    4. Verificador de fundamentação: números e palavras precisam vir do trecho.
       Reprovou -> pede de novo, dizendo o motivo; reprovou de novo -> próximo trecho.
    5. Nenhum trecho aprovado -> "não consta".
    6. A saída JSON é montada pelo código: a fonte e o link vêm do trecho aprovado,
       nunca do texto do modelo.
    """
    pergunta, _ = minimizar(pergunta)
    docs = buscador.buscar(pergunta, k=k)

    if docs[0]["score"] < limiar:
        return Resultado(resposta=NAO_CONSTA, score=docs[0]["score"], json_valido=True,
                         tentativas=0, motivo=f"porteiro: score {docs[0]['score']:.3f} < limiar {limiar}")

    chamadas, log = 0, []
    for d in [d for d in docs if d["score"] >= limiar]:
        prompt = PROMPT_AGENTE.format(texto=d["texto"], pergunta=pergunta)
        for _ in range(tentativas_por_trecho):
            chamadas += 1
            resp = gerar(prompt)
            if eh_recusa(resp):
                log.append(f"trecho {d['id']}: modelo disse que o trecho não responde")
                break
            ok, msg = verificar_fundamentacao(resp, d["texto"], pergunta)
            if ok:
                r = Resultado(resposta=resp, fonte=d["fonte"], url=d["url"], id_trecho=d["id"],
                              score=d["score"], tentativas=chamadas,
                              motivo="; ".join(log + [f"trecho {d['id']}: aprovado ({msg})"]))
                r.json_valido = _json_ok(r.to_json())
                return r
            log.append(f"trecho {d['id']}: reprovado ({msg})")
            # Só vale pedir de novo quando a resposta veio vazia ("Sim."/"Não.").
            # Se o problema foi número inventado ou fuga do trecho, insistir faz o
            # modelo apenas copiar o trecho e "passar" no verificador (README, erro 5).
            if not msg.startswith("resposta sem conteúdo"):
                break
            prompt += ("\n\nSua resposta anterior foi curta demais. Escreva uma frase completa "
                       "que explique a resposta usando as palavras do trecho.\nResposta:")

    return Resultado(resposta=NAO_CONSTA, score=docs[0]["score"], json_valido=True,
                     tentativas=chamadas, motivo="verificador: " + "; ".join(log))


_RECUSA = re.compile(r"nao consta|nao encontrei|nao ha (?:informac|mencion)|nao tenho (?:informac|acesso)|"
                     r"nao posso (?:fornecer|ajudar)|nao (?:e|esta) (?:mencionad|informad)|"
                     r"o trecho nao (?:menciona|informa|fala|traz)|no trecho (?:fornecido|apresentado)")


def eh_recusa(resposta: str) -> bool:
    """O modelo às vezes recusa com outras palavras ('Não encontrei informações...')."""
    return _RECUSA.search(_sem_acento(resposta)) is not None


def _json_ok(s: str) -> bool:
    try:
        obj = json.loads(s)
        return isinstance(obj.get("resposta"), str) and obj.get("fonte") is not None
    except (json.JSONDecodeError, AttributeError):
        return False
