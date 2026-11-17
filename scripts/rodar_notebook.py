"""Executa notebooks/pipeline_rag.ipynb do início ao fim e salva as saídas no próprio arquivo.

Uso (na raiz do projeto, com o Ollama rodando):  python scripts/rodar_notebook.py
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path(__file__).resolve().parent.parent / "notebooks" / "pipeline_rag.ipynb"

nb = nbformat.read(NB, as_version=4)
cliente = NotebookClient(nb, timeout=1800, kernel_name="python3",
                         resources={"metadata": {"path": str(NB.parent)}})
try:
    cliente.execute()
    print(f"OK: {NB} executado e salvo com as saídas.")
finally:
    nbformat.write(nb, NB)
