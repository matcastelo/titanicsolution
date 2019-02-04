#!/usr/bin/env python
# coding: utf-8

# # 1. Definição do problema

# Analisar os dados dos passageiros e prever as caracteristicas para sobrevivência no desastre do Titanic. Geralmente são trabalhados dois conjuntos de dados, um de treino e outro para o teste.
# 
# - train.csv = tem todos os dados, inclusive de quem sobreviveu ou não.
# - test.csv = nos dados teste o valor de quem sobreviveu precisa se deduzido (no survival data)
# 

# ![tittle](Downloads/news94391.jpg)

# ## 1.1 Dados do problema

# - PassengerId: Número de identificação do passageiro;
# - Survived: Indica se o passageiro sobreviveu ao desastre. É atribuído o valor de 0 para aqueles que não sobreviveram, e 1 para quem sobreviveu;
# - Pclass: Classe na qual o passageiro viajou. É informado 1 para primeira classe; 2 para segunda; e 3 para terceira;
# - Name: Nome do passageiro;
# - Sex: Sexo do passageiro;
# - Age: Idade do passageiro em anos;
# - SibSp: Quantidade de irmãos e cônjuges a bordo;
# - Parch: Quantidade de pais e filhos a bordo;
# - Ticket: Número da passagem;
# - Fare: Preço da passagem;
# - Cabin: Número da cabine do passageiro;
# - Embarked: Indica o porto no qual o passageiro embarcou. Há apenas três valores possíveis: Cherbourg, Queenstown e Southampton, indicados pelas letras C, Q e S, respectivamente.

# # 2. Importação dos dados

# ## 2.1 Importação das bibliotecas

# In[2]:


#Importar numpy para trabalhar com vetores, matrizes e ferramentas de Algebra Linear
import numpy as np 

# #Importar a biblioteca pandas para processamento e manipulação dos dados do problema
import pandas as pd 

#Gráficos mostrados diretamente no Notebook
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import style


# ## 2.2 Importação dos arquivos

# In[3]:


#Importar os arquivos e associar as váriaveis test e train
test = pd.read_csv('Downloads/test.csv')
train = pd.read_csv('Downloads/train.csv')


# In[4]:


#Dados das primeiras 5 linhas do arquivo treino
train.head(10)


# In[5]:


#Dados das primeiras 15 linhas do arquivo teste
test.head(10)


# Os dados do dataset terão que ser convertidos em numéricos ao longo do problema, pois só assim poderemos utilizar os algoritmos de Machine Learning. Os recursos também possuem muitos intervalos diferentes que precisam ser convertidos na mesma escala
# 
# Há muito valores ausentes (NaN) que precisam ser tratados.

# In[6]:


#Formato da matriz dos dados treino, 891 linhas e 12 colunas
print(train.shape)

#Formato da matriz dos dados teste, 418 linhas e 11 colunas (sem a coluna survived)
print(test.shape)


# In[7]:


train['Name'][train['PassengerId'] == 743]


# In[8]:


train['Name'][train['Sex'] == 'male'][train['Age'] == 40] 


# # 3. Análise dos dados

# ## 3.1 Estudando os dados

# In[9]:


#Analisar a natureza dos dados contidos do df train
train.info()


# In[10]:


#Realizar uma análise estatistica geral dos dados numéricos
train.describe()


# Analisando a tabela podemos conluir que:
# - Há 891 dados (linhas) e algumas colunas possuem dados faltantes, como por exemplo em Age (177)
# - As idades variam de 0.4 anos a 80 anos

# In[11]:


#média da idade
train['Age'].mean()


# In[12]:


#Quantos morreram
train[train['Survived'] == 1]['Survived'].count()


# In[13]:


train[train['Survived']==1].value_counts()


# In[14]:


train[train["Survived"] == 1].count()


# In[15]:


train[train['Survived'] == 0].count()


# ## 3.2 Retirando dados irrelevantes

# Alguns dados são irrelevantes nesse problema, assim poderemos retira-los do nosso dataset
# - Name: Já que temos o sexo, não importa o nome das pessoas para a análise (seria útil caso tivessemos dados faltantes no 'sex' poderíamos descobrir o sexo do passageiro pelo seu nome
# - O número do ticket também é irrelevante já que temos as classes
# - Cabine: Já temos a classe 'pclasses'então não precisaremos

# In[16]:


#utilizar o drop() para retirar as colunas do train e test
#axis é a direção, sendo 0 para linhas (index) e 1 para colunas
#inplace = True serve para ja substituir no dataframe


# In[17]:


train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace = True)


# In[18]:


train.head(25)


# In[19]:


test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace = True)


# In[20]:


test.head()


# ## 3.3 Análise dos dados por categoria

# A função pivot_table() do Pandas cria uma tabela que agraga os dados mencionados, passando qual deve ser a variável das linhas e colunas, a função que deve agregar os valores contagem,soma,etc e os valores aos quais esta função será aplicada, dividindo entre os valores já definidos nas linhas e colunas.
# 
# É parecido com a tabela dinâmica do Excel

# ## Sex

# In[21]:


sex_pivot = train.pivot_table(values = 'PassengerId', index = 'Sex',columns = "Survived", aggfunc = 'count')
sex_pivot


# In[22]:


sex_pivot.plot(kind = 'bar')
plt.title('Survived x Sex')
plt.show()


# Podemos concluir que o sexo é importante variavel na decisão, onde grande parte das mulheres sobreviveu e alta quantidades de homens morreram

# ## Pclass

# In[23]:


pclass_pivot = train.pivot_table(values = 'PassengerId', index = 'Pclass',columns = "Survived", aggfunc = 'count')
pclass_pivot


# In[24]:


pclass_pivot.plot(kind = 'bar')
plt.title('Survived x Pclass')
plt.show()


# Podemos concluir que as classes afetam o resultado final e caso você fosse da terceira classe sua chance de morrer é bem alta

# ## Parch (pais e filhos abordo)

# In[25]:


parch_pivot = train.pivot_table(values = 'PassengerId', index = 'Parch',columns = "Survived", aggfunc = 'count')
parch_pivot


# In[26]:


parch_pivot.plot(kind = 'bar')
plt.title('Parch x Pclass')
plt.show()


# Podemos concluir que se você estivesse sem pai e sem filho, sua probabilidade de morrer é alta. Pessoas com familia sobreviveram mais.

# ## Age

# In[27]:


age_pivot = train.pivot_table(values = 'PassengerId', index = 'Age',columns = "Survived", aggfunc = 'count')
age_pivot.head(15)


# In[28]:


age_pivot.plot(kind = 'bar' )
plt.title('Parch x Pclass')
plt.show()


# In[29]:


age_pivot.plot(kind ='bar', figsize = (12,8))


# In[30]:


#atribuir o objeto survived para survived = 1 e died para survived = 0 do dataset train
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]

#plotar os gráficos histogramas survived e died 
#alpha é a transparência e bins é o detalhamento das entradas
survived["Age"].plot(kind= 'hist', alpha=0.8,color='orange',bins=50)
died["Age"].plot(kind = 'hist', alpha=0.6,color='blue',bins=50)

#Imprimindo os gráficos
plt.legend(['Survived','Died'])
plt.show()


# Podemos concluir que se você fosse criança a sua probabilidade de sobreviver era maior

# # 4. Engenharia de Dados

# Para utilizar os modelos do machine learning deve possuir todos os dados em valores numéricos. Neste caso utilizarei o One-hot encoding, utilizando get_dummies(). O modelo transforma variáveis categóricas (dados não numéricos) em dados numéricos. 

# In[32]:


#Criar novo dataframe a partir do one hot coding
new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)


# In[31]:


train.head()


# OBS: get_dummies() aceita uma série de argumentos, como informar colunas específicas que desejamos codificar, se nada for passado nos parênteses, ele irá codificar todas as colunas não-numéricas de nosso dataset. Por sorte, os conjuntos de dados do Titanic possuem apenas duas colunas com variáveis não-numéricas: 
# - o sexo do passageiro
# - porto de embarque.

# In[33]:


new_data_train.head(15)


# In[50]:


new_data_test.head()


# # 5. Tratar valores nulos (NaN)

# In[34]:


# qtdd de valores nulos no train
new_data_train.isnull().sum().sort_values(ascending = False).head(10)

# isnull() retorna os valores nulos encontrados no determinado dataframe
# sum() é para somar os valores encontrados 
# sort_values(ascending = False) ordenando do maior pro menor


# Não podemos deixar os valores nulos, neste caso farei uma aproximação, colocarei dentro dos NaN o valor da idade média do datset

# In[35]:


# substituindo os valores
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace = True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace = True)

# primeiro selecionei o objeto: a coluna Age do dataframe
# fillna() substitui os valores NaN pelo argumento selecionado
# no caso, o argumento é o valor médio da coluna Age
# Inplace True serve para já substituir o resultado no próprio dataframe


# In[36]:


train['Age'].mean()


# In[47]:


new_data_train.head(15)


# In[48]:


new_data_train.isnull().sum().sort_values(ascending = False).head(10)


# In[49]:


new_data_test.isnull().sum().sort_values(ascending = False).head(10)


# In[51]:


new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace = True)
new_data_test.isnull().sum().sort_values(ascending = False).head(10)


# # 6. Aplicar o Modelo de Machine Learning

# ## 6.1 Decision Tree

# Esse é um problema de classificação em aprendizagem supervisionada: 
# - Classificação é o processo de tomar algum tipo de entrada e atribuir um rótulo a ela. Sistemas de classificação são usados geralmente quando as previsões são de natureza distinta, ou seja, um simples “sim ou não”. Neste caso, a decisão binária é de sobreviveu ou não.
# 
# O algoritmo selecionado será o Decision Tree (Árvore de Decisão)

# In[52]:


from sklearn.tree import DecisionTreeClassifier


# In[72]:


# A váriavel x armazenará todos os dados do passageiro (todo o dataset menos o survived)
x = new_data_train.drop('Survived', axis = 1)

# A variável y armazenará o que queremos prever, ou seja, se sobreviveu ou não. 
y = new_data_train['Survived']

# O modelo
tree = DecisionTreeClassifier(max_depth = 3, random_state = 0)
# max_depth é a profundidade da arvore, ou seja a quantidade de perguntas a serem respondidas

# Utilizar o método fit()
tree.fit(x,y)


# In[55]:


tree.score(x,y)


# Score no Kaggle = 0.77990 (Posição 5094)

# ## 6.1.1 Submissão no Kaggle

# In[57]:


# Criar um dataframe com duas colunas, o passengerId e a coluna survived (utilizar arquivo treino)
kaggle = pd.DataFrame()

# Associar a coluna passengerId com os dados do arquivo test
kaggle['PassengerId'] = new_data_test['PassengerId']

# Associar o survived com o modelo de previsão criado
kaggle['Survived'] = tree.predict(new_data_test)


# In[59]:


#Criando o arquivo CSV
kaggle.to_csv('Downloads/titanic_challenge', index = False)


# ## 6.2 Linear Regression

# In[64]:


# Importando a classe regressão logistica
from sklearn.linear_model import LogisticRegression


# In[65]:


# Criando o objeto da regressão
lr = LogisticRegression()


# Nos dois argumentos do método fit(x,y), x deve ser um dataframe bidemensional e y unidemensional (alvo ou target)

# In[71]:


a = new_data_train.drop('Survived', axis = 1)
b = new_data_train['Survived']

#A função de regressão linear aceita dois argumentos lr.fit(x,y)
lr.fit(a, b)


# Que erro é esse?? ^

# In[73]:


lr.score(a,b)


# In[74]:


# Criar um dataframe com duas colunas, o passengerId e a coluna survived (utilizar arquivo treino)
kaggle2 = pd.DataFrame()

# Associar a coluna passengerId com os dados do arquivo test
kaggle2['PassengerId'] = new_data_test['PassengerId']

# Associar o survived com o modelo de previsão criado
kaggle2['Survived'] = lr.predict(new_data_test)


# In[75]:


#Criando o arquivo CSV
kaggle2.to_csv('Downloads/titanic_challenge2', index = False)


# Score no Kaggle = 0.74641 (Decision Tree = 0.77990)

# # 7. Tentando melhorar o modelo

# In[78]:


x = new_data_train.drop('Survived', axis = 1)
y = new_data_train['Survived']

# utilizar max_depth
tree2 = DecisionTreeClassifier(max_depth = 10, random_state = 0)
# max_depth é a profundidade da arvore, ou seja a quantidade de perguntas a serem respondidas

tree2.fit(x,y)


# In[79]:


tree2.score(x,y)


# O score aumentou, tem chance de causar overfiting?

# In[82]:


kaggle3 = pd.DataFrame()
kaggle3['PassengerId'] = new_data_test['PassengerId']
kaggle3['Survived'] = tree2.predict(new_data_test)


# In[83]:


kaggle3.to_csv('Downloads/titanic_challenge3', index = False)


# Deu ruim, score kaggle = 0.68421

# In[ ]:




