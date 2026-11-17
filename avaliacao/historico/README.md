# Histórico das rodadas de avaliação

Cada rodada usou o mesmo conjunto de teste (38 perguntas) e o mesmo `llama3.2:1b`. Os números são acurácia geral / taxa de alucinação / recusa indevida.

| Rodada | O que mudou | A. sem contexto | B. RAG simples | C. Agente |
|---|---|---|---|---|
| 1 | Primeira versão: A com instrução "se não souber, diga não consta"; C com top-3 numerados, JSON `{"resposta", "id_trecho"}` e opção de recusa dada à LLM | 26% / 0% / **100%** | 58% / 13% / 39% | 39% / 18% / 57% |
| 2 | A vira baseline puro; `num_predict=200` contra o loop de espaços no modo JSON | 13% / 87% / 0% | 58% / 13% / 39% | 47% / 13% / 54% |
| 3 | C: um trecho por vez + JSON schema + verificador de fundamentação (teste rápido, só C) | - | - | JSON aninhado e campos inventados continuaram |
| 4 | C: geração em texto livre, JSON montado pelo código, detector de recusa, nova tentativa só para resposta vazia | 13% / 87% / 0% | 58% / 13% / 39% | 87% / 11% / 4% |
| final | Correção também reconhece recusas educadas da config A; minimização de dados pessoais (LGPD) | 24% / 71% / 7% | 58% / 13% / 39% | 87% / 11% / 4% |

Arquivos: `rodada2_agente_json_*.csv` (rodada 2) e `rodada4_agente_texto_*.csv` (rodada 4). As respostas da rodada 1 não foram guardadas; os números acima saíram da saída do notebook naquela execução.
