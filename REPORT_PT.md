# Relatório

Relatório sobre o desenvolvimento da aplicação

## Processamento

Todos os comentários passam por um processamento antes de serem procossados

- **Remove nome próprios:** Alguns comentário possuem nomes de atores e/ou diretores e devem ser desconsiderados do comentário
- **Remove Stop Words:** Remove dos comentários palavras que não possuem relevância para a classificação como por exemplo artigos(the, an, a), pronomes(he, she, it)
- **Stemming:** Remove o final das palavras para retirar plurais e derivações
- **Remove Número e Simbolos:** Remove dos comentários números e simbolos
- **Renomear films para movies:** Os termos film e movie possuem o significado, portanto um deles foi substituído para ficar pelo outro

## Análise de Sentimento

Foram utilizados dois métodos para efetuar a análise de sentimento do comentário

- **Vader(Valence Aware Dictionary and sEntiment Reasoner):** Uma ferramenta para análise de sentimento feita especificamente para mídias socias
- **Senti Word Net:** Uma fonte de palavras com classificações de sentimento 

## Bag of Words

Depois do preprocessamento foi extraido um conjunto das palavras mais comuns entre os comentários para gerar uma lista com essas palavras.

A extração foi realizada separando os comentários positivos e negativos. Cada conjunto foi as 100 palavras mais frequentes, utilizando n-gramas={1, 2, 3}. Então o conjunto de palavras dos comentários positivos e negativos foram agrupados removendo as duplicações


## Classificadores
 
- Multinomial Naive Bayes
	- Padrão
- Gaussian Naive Bayes
	- Padrão
- Bernoulli Naive Bayes
	- Padrão
- KKN (K-Nearest Neighbor)
    - **n_neighors**: 5, 10
- SVM
	- **kernel**: brf, linear
- Voting Classifier
	- **Abordagem 1**: 
		- **Classificadores**: Multinomial Naive Bayes, Gaussian Naive Bayes e Bernoulli Naive Bayes
		- **Pesos**: 2, 2, 1
	- **Abordagem 2:**
	    - **Classificadores**: Multinomial Naive Bayes, Bernoulli Naive Bayes, SVM - Linear
	    - **Pesos:** 1, 2, 2
	


## Resultados

A seguir os resultados obtidos por cada classificador:

| Classificador           | Accuracy | Precision | Recall  |
|-------------------------|----------|-----------|---------|
| Multinomial Naive Bayes | 0.73552  | 0.72105421| 0.76824 |
| Gaussian Naive Bayes    | 0.70828  | 0.69187117| 0.75104 |
| Bernoulli Naive Bayes   | 0.75712  | 0.74422492| 0.8208  |
| Voting - Abordagem 1    | 0.73708  | 0.71898322| 0.7784  |
| Voting - Abordagem 2    | 0.75576  | 0.74300699| 0.782   |
| KNN                     | 0.667    | 0.66373833| 0.67686 |
| SVM - RBF               | 0.702    | 0.66302944| 0.82152 |
| SMV - Linear            | 0.75448  | 0.07453717| 0.77304 |


Devido ao Bernoulli Naive Bayes ter obtido a maior acuracia, este método foi adotado para realizar a classificação dos comentários
