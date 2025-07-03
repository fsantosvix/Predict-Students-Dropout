## Predict-Students-Dropout

## 1. Contexto e Informações Gerais

### 1.1 Descrição do Problema

A evasão estudantil é um dos principais desafios enfrentados por universidades e faculdades, afetando a eficiência do ensino e até mesmo a reputação dessas instituições. Esse fenômeno, frequentemente denominado _student churn_, refere-se à saída de estudantes antes da conclusão do curso. Manter os estudantes engajados e garantir sua permanência até a conclusão do curso envolve fatores acadêmicos, sociais, econômicos e individuais que nem sempre são fáceis de identificar.

O objetivo deste estudo é antecipar o risco de evasão, utilizando dados históricos e características conhecidas dos estudantes ao longo de sua trajetória acadêmica. A partir dessa análise, espera-se contribuir para que universidades e faculdades possam identificar perfis de risco com maior precisão e, futuramente, fundamentar ações proativas voltadas à retenção estudantil.

### 1.2 Hipóteses do Problema

As hipóteses que tratarei nesse estudo que envolve apenas a etapa de Pré-Processamento dos dados são as seguintes:

1. A idade dos estudantes interfere na nota de admissão?

2. A nota obtida em qualificação anterior é indicativo da nota de admissão no curso estudado posteriormente?

3. Estudantes com necessidades educacionais especiais apresentam maior taxa de evasão?

### 1.3 Tipo de Problema

Dentro de um escopo mais abrangente, trata-se de um problema de **classificação supervisionada**. Dado um conjunto de características detalhadas no decorrer do estudo, busca-se prever se o estudante tende a abandonar o curso ou permanecer matriculado (ou concluir o curso) decorrida a duração normal da qualificação.

### 1.4 Seleção de Dados

Os dados utilizados nesse estudo foram selecionados a partir de bases disponíveis no [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/), uma plataforma amplamente conhecida que disponibiliza datasets para a pessoas interessadas em aprimorar seus conhecimentos em machine learning.
O banco de dados em questão foi construído a partir de datasets distintos, relacionados a estudantes matriculados em diversos cursos de graduação.
As informações contidas no conjunto de dados abrangem tanto informações coletadas no momento da matrícula dos estudantes quanto o desempenho acadêmico deles ao final do primeiro e segundo semestres.

Dentre as limitações deste dataset, há que se destacar a descrição insuficiente de algumas variáveis. Enquanto algumas são auto-explicativas, outras não fornecem um detalhamento adequado para uma análise e interpretação mais precisa.

### 1.5 Atributos do Dataset

O dataset contém 4424 instâncias, com 36 atributos e um target. Abaixo encontra-se um quadro resumo dos atributos.

| Nome da Variável                        | Papel   | Tipo        | Descrição                                    | Valores Possíveis                                                                               | Valores Ausentes |
| :-------------------------------------- | :------ | :---------- | :------------------------------------------- | :---------------------------------------------------------------------------------------------- | :--------------- |
| Marital status                          | Feature | Categorical | Estado civil do estudante                    | 1 – solteiro, 2 – casado, 3 – viúvo, 4 – divorciado, 5 – união estável, 6 – legalmente separado | Não              |
| Application mode                        | Feature | Categorical | Forma de ingresso na instituição             | (códigos de aplicação)                                                                          | Não              |
| Application order                       | Feature | Integer     | Ordem de preferência da candidatura          | 1–9                                                                                             | Não              |
| Course                                  | Feature | Categorical | Curso de graduação                           | (códigos de curso)                                                                              | Não              |
| Daytime/evening attendance              | Feature | Binary      | Turno do curso                               | 1=daytime, 0=evening                                                                            | Não              |
| Previous qualification                  | Feature | Categorical | Tipo de qualificação anterior                | (códigos de qualificação)                                                                       | Não              |
| Previous qualification (grade)          | Feature | Continuous  | Nota da qualificação anterior                | 0–200                                                                                           | Não              |
| Nacionality                             | Feature | Categorical | Nacionalidade do estudante                   | (códigos de país)                                                                               | Não              |
| Mother's qualification                  | Feature | Categorical | Escolaridade da mãe                          | (códigos de qualificação)                                                                       | Não              |
| Father's qualification                  | Feature | Categorical | Escolaridade do pai                          | (códigos de qualificação)                                                                       | Não              |
| Mother's occupation                     | Feature | Categorical | Profissão da mãe                             | (códigos de ocupação)                                                                           | Não              |
| Father's occupation                     | Feature | Categorical | Profissão do pai                             | (códigos de ocupação)                                                                           | Não              |
| Admission grade                         | Feature | Continuous  | Nota de admissão no curso                    | 0–200                                                                                           | Não              |
| Displaced                               | Feature | Binary      | Se o estudante reside fora da área habitual  | 1=sim, 0=não                                                                                    | Não              |
| Educational special needs               | Feature | Binary      | Se o estudante tem necessidades especiais    | 1=sim, 0=não                                                                                    | Não              |
| Debtor                                  | Feature | Binary      | Se o estudante tem dívidas com a instituição | 1=sim, 0=não                                                                                    | Não              |
| Tuition fees up to date                 | Feature | Binary      | Se as mensalidades estão em dia              | 1=sim, 0=não                                                                                    | Não              |
| Gender                                  | Feature | Binary      | Gênero do estudante                          | 1=masculino, 0=feminino                                                                         | Não              |
| Scholarship holder                      | Feature | Binary      | Se o estudante é bolsista                    | 1=sim, 0=não                                                                                    | Não              |
| Age at enrollment                       | Feature | Integer     | Idade no momento da matrícula                | 17–70                                                                                           | Não              |
| International                           | Feature | Binary      | Se o estudante é estrangeiro                 | 1=sim, 0=não                                                                                    | Não              |
| Curricular units 1st sem (credited)     | Feature | Integer     | Disciplinas creditadas no 1º semestre        | 0–20                                                                                            | Não              |
| Curricular units 1st sem (enrolled)     | Feature | Integer     | Disciplinas matriculadas no 1º semestre      | 0–26                                                                                            | Não              |
| Curricular units 1st sem (evaluations)  | Feature | Integer     | Avaliações realizadas no 1º semestre         | 0–45                                                                                            | Não              |
| Curricular units 1st sem (approved)     | Feature | Integer     | Disciplinas aprovadas no 1º semestre         | 0–26                                                                                            | Não              |
| Curricular units 1st sem (grade)        | Feature | Continuous  | Nota média no 1º semestre                    | 0–20                                                                                            | Não              |
| Curricular units 1st sem (without eval) | Feature | Integer     | Disciplinas sem avaliação no 1º semestre     | 0–12                                                                                            | Não              |
| Curricular units 2nd sem (credited)     | Feature | Integer     | Disciplinas creditadas no 2º semestre        | 0–20                                                                                            | Não              |
| Curricular units 2nd sem (enrolled)     | Feature | Integer     | Disciplinas matriculadas no 2º semestre      | 0–26                                                                                            | Não              |
| Curricular units 2nd sem (evaluations)  | Feature | Integer     | Avaliações realizadas no 2º semestre         | 0–45                                                                                            | Não              |
| Curricular units 2nd sem (approved)     | Feature | Integer     | Disciplinas aprovadas no 2º semestre         | 0–26                                                                                            | Não              |
| Curricular units 2nd sem (grade)        | Feature | Continuous  | Nota média no 2º semestre                    | 0–20                                                                                            | Não              |
| Curricular units 2nd sem (without eval) | Feature | Integer     | Disciplinas sem avaliação no 2º semestre     | 0–12                                                                                            | Não              |
| Unemployment rate                       | Feature | Continuous  | Taxa de desemprego no país                   | 0–100                                                                                           | Não              |
| Inflation rate                          | Feature | Continuous  | Taxa de inflação no país                     | 0–100                                                                                           | Não              |
| GDP                                     | Feature | Continuous  | Produto Interno Bruto                        | (-5)–5                                                                                          | Não              |
| Target                                  | Target  | Categorical | Situação final do estudante                  | Dropout, Enrolled, Graduate                                                                     | Não              |

## 2. Conclusão

Apesar de não apresentar dados faltantes, trata-se de um dataset complexo, que apresenta grande variedade de atributos de diferentes tipos (categóricos, numéricos discretos, contínuos). Isso demanda especial atenção por parte do analista e exige a aplicação de técnicas de pré-processamento separadas para não comprometer a integridade dos dados e garantir que o modelo interpretará corretamente cada variável.

Esse estudo buscou fazer comparações pontuais entre algumas variávies para explorar diferentes visualizações possíveis. No entanto, não era o objetivo esgotar todos os cruzamentos e hipóteses possíveis. Considerando as 36 features iniciais, há uma gama quase inesgotável de comparações e verificações que podem ser feitas em um estudo mais completo.

Quanto às hipóteses levantadas inicialmente:

_1. A idade dos estudantes interfere na nota de admissão?_

**RESPOSTA**: Não. A correlação é praticamente inexistente entre essas variáveis.
<br>

_2. A nota obtida em qualificação anterior é indicativo da nota de admissão no curso estudado posteriormente??_

**RESPOSTA**: Inconclusivo. A correlação entre as variáveis foi considerada moderada, portanto, seria leviano alegar que o comprotamento de uma é determinante para a outra.
<br>

_3. Estudantes com necessidades educacionais especiais apresentam maior taxa de evasão?_

**RESPOSTA**: Não. Os índices de evasão para estudantes com e sem necessidades educacionais especiais foi muito próximo, o que indica um comportamento quase idêntico em cada uma das duas populações.
