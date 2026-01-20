\# Integração de Análise de Sentimento com o Modelo Black-Litterman



\## 1. Contexto e Motivação



Mercados de commodities são fortemente influenciados por \*\*informações qualitativas\*\*, como:



\- clima  

\- política agrícola  

\- relatórios governamentais  

\- logística  

\- conflitos geopolíticos  

\- expectativas macroeconômicas  



Essas informações chegam ao mercado principalmente via \*\*notícias\*\*, cujo impacto não é diretamente quantificável.



Modelos clássicos de otimização de portfólio, como \*\*Markowitz\*\*, assumem que:



\- retornos passados capturam toda a informação relevante;  

\- expectativas futuras podem ser estimadas apenas por médias históricas.  



Essa hipótese é frágil em mercados informacionalmente dinâmicos.



Usar sentimento diretamente como \*\*sinal de trading\*\* leva a:

\- instabilidade,

\- excesso de ruído.



\*\*Proposta:\*\* formalizar o sentimento dentro de um \*\*framework bayesiano disciplinado\*\*, utilizando o \*\*modelo Black-Litterman\*\* para integrar:

\- informação quantitativa (mercado),

\- informação qualitativa (notícias).



---



\## 2. Análise de Sentimento Financeiro (FSA)



\### 2.1 Papel do Sentimento no Modelo



O sentimento \*\*não é um modelo de retorno\*\*. Ele é tratado como:



\- uma \*\*opinião (view)\*\*,

\- sujeita a erro,

\- com grau de confiança variável.



Formalmente:



> sentimento ≠ previsão direta  

> sentimento → expectativa condicional → view probabilística



---



\### 2.2 Abordagens Avaliadas



\#### 2.2.1 Modelos Especializados (BERT-like)



\*\*FinBERT\*\*

\- Prós: léxico financeiro, baixo custo, fácil implementação.

\- Contras:

&nbsp; - baixa sensibilidade a valores numéricos,

&nbsp; - dificuldade com ironia, causalidade e contexto curto.



\*\*BloombergGPT\*\*

\- Prós: estado da arte.

\- Contras: custo e acesso inviáveis.



\*\*Conclusão:\*\* bons como baseline, insuficientes como núcleo do sistema.



---



\#### 2.2.2 LLMs Generalistas com Engenharia de Prompt



Resultados empíricos indicam que:

\- LLMs bem instruídos superam FinBERT em \*F1-score\*,

\- vantagem maior em notícias curtas e ambíguas.



Técnicas-chave:

\- role-playing (“aja como analista de commodities”),

\- prompts estruturados,

\- cadeias de raciocínio implícitas.



Limitação estrutural:

\- fraqueza em raciocínio numérico se não guiados.



---



\#### 2.2.3 Frameworks Cognitivos



\*\*FAP (Financial Attribute Prompting)\*\*  

Força análise explícita de:

\- semântica,

\- temporalidade,

\- causalidade,

\- risco,

\- comparação implícita.



\*\*HAD (Multi-agentes)\*\*  

Debate interno entre agentes especializados reduz erros sutis.



Trade-off:

\- ↑ qualidade  

\- ↑ custo computacional



---



\#### 2.2.4 Recuperação de Contexto (RAG)



Problemas comuns em notícias financeiras:

\- omissão de contexto histórico,

\- linguagem ambígua.



RAG resolve ao:

\- recuperar eventos passados,

\- incorporar relatórios oficiais (USDA, CONAB),

\- adicionar contexto setorial.



Resultado: redução significativa de erros de classificação.



---



\### 2.3 Escolha Metodológica



Para aplicações acadêmicas e práticas:



\*\*LLM via API + engenharia de prompt + agregação temporal\*\*  

é o melhor trade-off entre:

\- desempenho,

\- custo,

\- simplicidade,

\- interpretabilidade.



---



\## 3. Modelo Black-Litterman



\### 3.1 Fundamentação Conceitual



O Black-Litterman é um modelo \*\*bayesiano\*\* que atualiza expectativas de retorno:



\- \*\*Prior:\*\* crença inicial do mercado  

\- \*\*Likelihood:\*\* views (opiniões)  

\- \*\*Posterior:\*\* retornos ajustados  



Benefícios:

\- evita extremos do Markowitz,

\- reduz instabilidade numérica,

\- disciplina o uso de opiniões subjetivas.



---



\### 3.2 Formulação Matemática



Retorno esperado posterior:



\\\[

\\mu\_{BL} =

\\left\[

(\\tau \\Sigma)^{-1} + P^\\top \\Omega^{-1} P

\\right]^{-1}

\\left\[

(\\tau \\Sigma)^{-1} \\pi + P^\\top \\Omega^{-1} q

\\right]

\\]



\#### Definição dos Termos



| Símbolo | Descrição | Dimensão |

|-------|----------|----------|

| \\( \\pi \\) | Retornos implícitos de equilíbrio (prior) | N×1 |

| \\( \\Sigma \\) | Matriz de covariância | N×N |

| \\( \\tau \\) | Incerteza do prior (0,01–0,05) | 1×1 |

| \\( P \\) | Matriz de alocação das views | K×N |

| \\( q \\) | Retornos esperados das views | K×1 |

| \\( \\Omega \\) | Incerteza das views | K×K |

| \\( \\mu\_{BL} \\) | Retornos posteriores | N×1 |



---



\### 3.3 Interpretação Econômica



\- O mercado domina quando as views são fracas.

\- Views dominam apenas quando:

&nbsp; - são fortes,

&nbsp; - possuem alta confiança.

\- Nenhuma opinião isolada “quebra” o portfólio.



---



\## 4. Construção das Views a partir do Sentimento



\### 4.1 Conversão Sentimento → Retorno



O sentimento é normalizado em \\( \[-1, 1] \\) e mapeado linearmente:



\\\[

q\_i = \\pi\_i + \\alpha \\cdot s\_i

\\]



Onde:

\- \\( s\_i \\): score de sentimento normalizado,

\- \\( \\alpha \\): amplitude máxima do tilt (ex.: 2–5% a.a.),

\- \\( \\pi\_i \\): retorno de equilíbrio.



Vantagens:

\- evita não linearidades arbitrárias,

\- mantém interpretabilidade econômica.



---



\### 4.2 Definição da Incerteza das Views



A confiança da view depende do módulo do sentimento:



\- |sentimento| alto → menor incerteza,

\- |sentimento| baixo → maior incerteza.



Implementações:

\- método de \*\*Idzorek\*\*,

\- função proporcional a \\( |s| \\),

\- \\( \\Omega \\) escalar (simplificação industrial).



---



\## 5. Arquitetura do Sistema



\### 5.1 Pipeline



1\. Coleta de notícias por ativo  

2\. Limpeza e deduplicação  

3\. Inferência de sentimento via LLM  

4\. Agregação temporal (semanal/mensal)  

5\. Geração das views  

6\. Atualização Black-Litterman  

7\. Otimização média–variância  

8\. Rebalanceamento  



---



\### 5.2 Decisões de Engenharia



\- Uma view por ativo por período  

\- Views apenas para sentimentos extremos  

\- Rebalanceamento de curto prazo  

\- Covariância recalibrada periodicamente  



Objetivo: reduzir ruído, overfitting e instabilidade.



---



\## 6. Avaliação e Validação



\### 6.1 Métricas Financeiras



\- Retorno acumulado  

\- Volatilidade  

\- Sharpe Ratio  

\- Maximum Drawdown  



\### 6.2 Métricas Informacionais



\- Correlação sentimento × retorno futuro  

\- Estabilidade das views  

\- Sensibilidade a \\( \\alpha, \\tau, \\Omega \\)  



---



\## 7. Limitações



\- Dependência da qualidade das notícias  

\- Latência informacional  

\- Sentimento captura narrativa, não fundamento  

\- Suposição de normalidade aproximada  



Mitigadas, não eliminadas.



---



\## 8. Extensões Possíveis



\- Views relativas entre commodities  

\- Pesos por fonte de notícia  

\- Ensemble de modelos de sentimento  

\- Backtests multi-regime  

\- Integração com fatores macro  



---



\## 9. Conclusão



A integração de \*\*LLMs para análise de sentimento\*\* com o \*\*modelo Black-Litterman\*\* resulta em um sistema:



\- teoricamente consistente,  

\- estatisticamente robusto,  

\- economicamente interpretável,  

\- computacionalmente viável.  



O sentimento deixa de ser um sinal frágil e passa a atuar como \*\*opinião probabilística disciplinada pelo mercado\*\*, tornando o framework adequado para uso acadêmico e aplicado.



