---
layout: home
---

Bem-vindo a uma jornada prática pelo mundo da inteligência artificial aplicada aos negócios! Se você já se pegou pensando em como aqueles modelos de *machine learning* podem, de fato, resolver problemas reais do dia a dia empresarial, este livro é para você.

Antes de começarmos, um aviso amigável: este livro não é um curso de *machine learning* nem um manual de **Python**. Vamos assumir que você já sabe o que é um algoritmo, já escreveu algumas linhas de código em **Python** e entende os conceitos básicos de aprendizado de máquina. Se você sabe que *overfitting* não é um novo treino na academia e que `pandas` não são apenas aqueles ursos chinenes que comem bambu, estamos no caminho certo!

Nosso foco aqui é a aplicação prática. Vamos pegar um algoritmo originalmente desenvolvido para prever doenças cardíacas e mostrar como adaptá-lo para resolver seis problemas comuns no mundo dos negócios. Sim, seis! Porque acreditamos que a melhor maneira de aprender é colocando a mão na massa, ou melhor, no teclado.

Ao longo dos capítulos, vamos explorar situações como prever a inadimplência de um cliente, detectar fraudes em transações financeiras e até mesmo antecipar se aquele cliente fiel está prestes a abandonar o barco. Tudo isso utilizando técnicas avançadas de otimização de modelos de *ensemble* com **Otimização Bayesiana** e adicionando uma pitada de interpretabilidade com valores SHAP.

E não se preocupe, apesar dos temas complexos, vamos manter a conversa leve e descontraída. Afinal, quem disse que não podemos aprender e nos divertir ao mesmo tempo? Como diria Marcus du Sautoy, a matemática (e, por extensão, a ciência de dados) é uma aventura repleta de descobertas e surpresas.

Prepare-se para decifrar mistérios, solucionar enigmas de negócios e, quem sabe, até impressionar aquele colega que sempre fala em jargões complicados. Vamos descomplicar juntos o mundo do *machine learning* aplicado!

Então, ajeite a cadeira, prepare o ambiente de desenvolvimento e vamos começar essa jornada rumo à aplicação prática e eficiente de algoritmos de aprendizado de máquina nos negócios.

## O Artigo que Inspirou Nossa Jornada
Tudo começou com o artigo [Optimized Ensemble Learning Approach with Explainable AI for Improved Heart Disease Prediction](https://www.mdpi.com/2078-2489/15/7/394)[1], publicado em junho de 2024. Não se assuste com o título complicado! Embora não seja um assunto simples e trivial, com um pouco de esforço e dedicação, você verá que é perfeitamente compreensível e, mais importante, aplicável a problemas reais de negócios. Este trabalho explora o uso da **Otimização Bayesiana** para ajustar hiperparâmetros em modelos de *ensemble*, como **Random Forest**, **AdaBoost** e **XGBoost**, além de empregar valores **SHAP** para interpretar as decisões do modelo.

Em termos simples, eles encontraram uma forma de deixar modelos já poderosos ainda melhores e, de quebra, mais transparentes. E aqui entre nós, quem não quer um modelo que além de eficiente, consegue explicar suas próprias decisões? É quase como ter um GPS que, além de te mostrar o caminho, ainda te conta as histórias dos lugares por onde você passa.

## Ferramentas que Vamos Utilizar
Para acompanhar os exemplos e colocar em prática o que vamos aprender, usaremos o **Jupyter Notebook**. Se você ainda não é amigo dessa ferramenta, prepare-se para conhecê-la melhor — ela será sua companheira fiel nesta jornada.

Agora, se você prefere a nuvem e não quer se preocupar com instalações, pode replicar todos os exercícios no **Google Colab**. Basta ter uma conta **Google**, e pronto: todos os recursos estarão ao seu alcance, sem precisar instalar nada.

Mas se você é do tipo que gosta de ter tudo rodando na sua própria máquina, recomendamos a instalação do **Anaconda**. Ele já vem com o **Jupyter Notebook** e todas as bibliotecas necessárias, como `numpy`, `pandas`, `scikit-learn`, `xgboost`, `shap` e muito mais. É como um pacote completo, tipo aquelas cestas de café da manhã que vêm com tudo o que você precisa para começar o dia feliz.

## O Que Esperar Desta Jornada
Ao longo dos capítulos, vamos:

- **Explorar o algoritmo original**: Entender como o artigo inspirador abordou a previsão de doenças cardíacas usando modelos de *ensemble* otimizados e interpretáveis.

- **Adaptar e aplicar o algoritmo a diferentes problemas de negócios**: Mostrar como podemos pegar essa base sólida e moldá-la para resolver desafios em diversas áreas.

- **Utilizar técnicas avançadas de otimização de hiperparâmetros com Otimização Bayesiana**: Porque ficar testando combinação por combinação é tão empolgante quanto assistir tinta secar.

- **Tornar nossos modelos interpretáveis usando valores SHAP**: Afinal, queremos saber não só o resultado, mas também o "porquê" por trás dele.

- **Escrever muito código em Python**: E quando dizemos muito, queremos dizer o suficiente para você se sentir um verdadeiro ninja dos notebooks.

Tudo isso de forma prática, objetiva e, esperamos, divertida. Afinal, aprender não precisa ser chato. **Python** é uma linguagem, e queremos que você se torne fluente nela — ou pelo menos saiba pedir um café e perguntar onde fica o banheiro.

## Preparando o Ambiente
Antes de começarmos, certifique-se de que tem tudo o que precisa:

- **Python 3.x** instalado (se estiver usando **Anaconda**, já está incluso).

- **Jupyter Notebook** ou acesso ao **Google Colab**.

- As bibliotecas necessárias, que iremos instalar e importar ao longo dos capítulos.

Se optar por usar o **Google Colab**, basta acessar `colab.research.google.com` e fazer *login* com sua conta **Google**. Todos os notebooks podem ser executados diretamente na nuvem, sem necessidade de instalação.

Caso prefira trabalhar localmente, recomendamos a instalação do **Anaconda**, que pode ser baixado em `www.anaconda.com`. Ele já vem com o **Jupyter Notebook** e um monte de bibliotecas úteis. É como aquele canivete suíço que todo aventureiro precisa ter.

## Vamos Começar?
Se você está pronto para aplicar algoritmos poderosos a problemas reais e entender cada passo do caminho, então vamos em frente. Ajuste sua cadeira, prepare o ambiente de desenvolvimento, e vamos iniciar essa jornada rumo à aplicação prática e eficiente de algoritmos de aprendizado de máquina nos negócios.

Ah, e não esqueça de trazer seu senso de humor. Ele pode não ser obrigatório, mas definitivamente torna a viagem mais agradável!

**Nota**: Os *notebooks* completos com os códigos utilizados ao longo do livro estão disponíveis no nosso **GitHub**. Recomendamos que você explore e reproduza todos os passos das análises por si mesmo!

[1] Mienye, Ibomoiye Domor, and Nobert Jere, "Optimized Ensemble Learning Approach with Explainable AI for Improved Heart Disease Prediction," Information, vol. 15, no. 7, p. 394, 2024. Disponível em: https://www.mdpi.com/2078-2489/15/7/394
