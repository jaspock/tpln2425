# Architectures for written-text processing

!!! danger

    These materials are temporary and incomplete. If you choose to read them, you do so at your own risk.

In this module, we will study some neural models used to process texts. The professor of this module is Juan Antonio Pérez Ortiz. The module begins with a review of the functioning of logistic regression, which will help us establish the necessary knowledge to understand subsequent models. Next, we study in some detail *skip-grams*, one of the algorithms for obtaining out-of-context word *embeddings*. Then, we review the functioning of *feedforward* neural architectures and study their application to language models. The ultimate goal is to address the study of the most important architecture in current text processing systems: the transformer. Once we have studied these architectures, we will conclude with an analysis of the functioning of pretrained models (foundational models) in general, and language models in particular.

Class materials complement the reading of some chapters from a textbook ("Speech and Language Processing" by Dan Jurafsky and James H. Martin, third edition draft, available online) with annotations made by the professor.

## First session of this module (December 20, 2023)

### Contents to prepare before the session on 12/20/2023

The activities to complete before this class are:

- Reading and studying the contents of [this page](https://dlsi.ua.es/~japerez/materials/transformers/regresor/) on logistic regression. As you will see, the page indicates which contents you should read from the book. After a first reading, read the professor's annotations, whose purpose is to help you understand the key concepts of the chapter. Then, perform a second reading of the chapter from the book. In total, this part should take you about 3 hours 🕒️ of work.
- Watching and studying the video tutorials in this [official PyTorch playlist](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN). Study at least the first 4 videos (“Introduction to PyTorch”, “Introduction to PyTorch Tensors”, “The Fundamentals of Autograd”, and “Building Models with PyTorch”). In total, this part should take you about 2 hours 🕒️ of work.
- After completing the two previous parts, take this [assessment test](https://forms.gle/V3U9MTHo7c9DNhkc6) on these contents. There are few questions, and it will take you a few minutes.

### Contents for the in-person session on 12/20/2023

In the in-person class (2.5 hours 🕒️ in duration), we will see how to implement a logistic regressor in PyTorch by following the implementations of a binary logistic regressor <a href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/logistic.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> and a multinomial one <a href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/softmax.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> discussed in [this section](https://dlsi.ua.es/~japerez/materials/transformers/implementacion/#codigo-para-un-regresor-logistico-y-uno-multinomial).

The idea is for you to study and slightly modify the notebooks we are working on. In a later class, a more advanced assignment involving modifying the transformer's code will be presented.

## Second session (January 10, 2024)

### Contents to prepare before the session on 01/10/2024

The activities to complete before this class are:

- Reading and studying the contents of [this page](https://dlsi.ua.es/~japerez/materials/transformers/embeddings/) on embeddings. As you will see, the page indicates which contents you should read from the book. After a first reading, read the professor's annotations to help you understand the key concepts of the chapter. Then, perform a second reading of the chapter from the book. In total, this part should take you about 4 hours 🕒️ of work.
- Reading and studying the contents of [this page](https://dlsi.ua.es/~japerez/materials/transformers/ffw/) on feedforward neural networks and their use as very basic language models. Perform at least two readings complemented with the professor's notes as in the previous point. In total, this part should take you about 2 hours 🕒️ of work.
- Reading and studying the contents of [this page](https://dlsi.ua.es/~japerez/materials/transformers/attention/) as an introduction to transformers. As always, perform at least two readings complemented with the professor's notes. In total, this part should take you about 4 hours 🕒️ of work.
- After completing the previous parts, take this [assessment test](https://forms.gle/7KDwRtXcrpxsKjHp7) on these contents. There are few questions, and it will take you a few minutes.
- If you have time left, take the opportunity to review the contents of the first session.

### Contents for the in-person session on 01/10/2024

In the in-person class (5 hours 🕒️ in duration), we will see how to implement in PyTorch the skip-gram algorithm <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/skipgram.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>, a language model based on a feedforward neural network <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/ffnn.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>, and a transformer <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/transformer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> following the implementations discussed in [this section](https://www.dlsi.ua.es/~japerez/materials/transformers/implementacion/#codigo-para-skip-grams) and the next two.

The idea is for you to study and slightly modify the notebooks we are working on. In a later class, a more advanced assignment involving modifying the transformer's code will be presented.

## Third session (January 17, 2024)

### Contents to prepare before the session on 01/17/2024

The activities to complete before this class are:

- Reading and studying the contents of [this page](https://dlsi.ua.es/~japerez/materials/transformers/attention2/) on the complete transformer model (with encoder and decoder) and the possible uses of an architecture that only includes the encoder. As you will see, the page indicates which contents you should read from the book. In particular, you will need to read some sections of the chapter on machine translation and others from the chapter on pretrained models, in addition to standalone sections on *beam search* and subword tokenization. After a first reading, read the professor's annotations to help you understand the key concepts of each section. Then, perform a second reading of the book's contents. In total, this part should take you about 4 hours 🕒️ of work.
- Watching and studying Jesse Mu's lecture titled “[Prompting, Reinforcement Learning from Human Feedback](https://youtu.be/SXpJ9EmG3s4?si=j4B1U2Z-JCyYJwlc)” from Stanford's CS224N course in 2023 about language models based on the transformer's decoder. This should take you about 2 hours 🕒️ of work, as you'll need to take notes so you don't have to rewatch the video when reviewing. Downloading the [slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture11-prompting-rlhf.pdf) and annotating them may be helpful. You can focus on the basic ideas discussed between minutes 39 and 46, as the reinforcement learning equations are not a priority topic for this course and will be covered in other courses. It's important to review what you've already studied about [transformers](https://dlsi.ua.es/~japerez/materials/transformers/attention/) as a language model based on the decoder before watching the video. Don't be confused by encoder-based models also sometimes being called language models. This video discusses the properties of decoder-based models trained to predict the next token in a sequence.
- Study the description of [multilingual models](https://dlsi.ua.es/~japerez/materials/transformers/attention2/#multilingual-models) in this section of one of the pages on transformers. It's a brief section that will take you about 🕒️ 15 minutes.
- After completing the previous parts, take this [assessment test](https://forms.gle/GRK5SLc3STkup8at9) on these contents. There are few questions, and it will take you a few minutes.
- If you have time left, take the opportunity to review all the contents from previous sessions.

### Contents for the in-person session on 01/17/2024

In the in-person class (5 hours 🕒️ in duration), we will see how to implement on top of our transformer architecture code both a language model based on a decoder <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/lmgpt.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> and a named entity recognition model <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/nerbert.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> based on an encoder.

We will take the opportunity to review some aspects of the code from previous sessions and relate theoretical aspects with practical ones. We will also present the assignment you need to submit for this module of the course.

## Fourth session (January 19, 2024)

This fourth session is actually the first and only session on the topic of speech. See the page on [speech](speech.md) to view the contents prior to this session.


# Architectures for written-text processing

En este bloque se aborda el estudio de algunos modelos neuronales utilizados para procesar textos. El profesor de este bloque es Juan Antonio Pérez Ortiz. El bloque comienza con un repaso del funcionamiento del regresor logístico, que nos servirá para asentar los conocimientos necesarios para entender posteriores modelos. A continuación se estudia con cierto nivel de detalle *skip-grams*, uno de los algoritmos para la obtención de *embeddings* incontextuales de palabras. Después se repasa el funcionamiento de las arquitecturas neuronales *feedforward* y se estudia su aplicación a modelos de lengua. El objetivo último es abordar el estudio de la arquitectura más importante de los sistemas actuales de procesamiento de textos: el transformer. Una vez estudiadas estas arquitecturas, finalizaremos con un análisis del funcionamiento de los modelos preentrenados (modelos fundacionales), en general, y de los modelos de lengua, en particular.

Los materiales de clase complementan la lectura de algunos capítulos de un libro de texto ("Speech and Language Processing" de Dan Jurafsky y James H. Martin, borrador de la tercera edición, disponible online) con anotaciones realizadas por el profesor.

## Primera sesión de este bloque (20 de diciembre de 2023)

### Contenidos a preparar antes de la sesión del 20/12/2023

Las actividades a realizar antes de esta clase son:

- Lectura y estudio de los contenidos de [esta página](https://dlsi.ua.es/~japerez/materials/transformers/regresor/) sobre regresión logística. Como verás, la página te indica qué contenidos has de leer del libro. Tras una primera lectura, lee las anotaciones del profesor, cuyo propósito es ayudarte a entender los conceptos clave del capítulo. Después, realiza una segunda lectura del capítulo del libro. En total, esta parte debería llevarte unas 3 horas 🕒️ de trabajo.
- Visionado y estudio de los tutoriales en vídeo de esta [playlist oficial de PyTorch](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN).  Estudia al menos los 4 primeros vídeos (“Introduction to PyTorch”, “Introduction to PyTorch Tensors”, “The Fundamentals of Autograd” y “Building Models with PyTorch”). En total, esta parte debería llevarte unas 2 horas 🕒️ de trabajo.
- Tras acabar con las dos partes anteriores, realiza este [test de evaluación](https://forms.gle/V3U9MTHo7c9DNhkc6) de estos contenidos. Son pocas preguntas y te llevará unos minutos.

### Contenidos para la sesión presencial del 20/12/2023

En la clase presencial (2,5 horas 🕒️ de duración), veremos cómo se implementa un regresor logístico en PyTorch siguiendo las implementaciones de un regresor logístico binario <a href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/logistic.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab\"></a> y de uno multinomial <a href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/softmax.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab\"></a> que se comentan en [este apartado](https://dlsi.ua.es/~japerez/materials/transformers/implementacion/#codigo-para-un-regresor-logistico-y-uno-multinomial).

La idea es que vayas estudiando y modificando ligeramente los notebooks que vayamos estudiando. En una clase posterior se presentará una práctica más avanzada que implicará modificar el código del transformer.

## Segunda sesión (10 de enero de 2024)

### Contenidos a preparar antes de la sesión del 10/01/2024

Las actividades a realizar antes de esta clase son:

- Lectura y estudio de los contenidos de [esta página](https://dlsi.ua.es/~japerez/materials/transformers/embeddings/) sobre los embeddings. Como verás, la página te indica qué contenidos has de leer del libro. Tras una primera lectura, lee las anotaciones del profesor, cuyo propósito es ayudarte a entender los conceptos clave del capítulo. Después, realiza una segunda lectura del capítulo del libro. En total, esta parte debería llevarte unas 4 horas 🕒️ de trabajo.
- Lectura y estudio de los contenidos de [esta página](https://dlsi.ua.es/~japerez/materials/transformers/ffw/) sobre las redes neuronales hacia delante y su uso como modelos de lengua muy básicos. Realiza al menos dos lecturas complementadas con las notas del profesor como en el punto anterior. En total, esta parte debería llevarte unas 2 horas 🕒️ de trabajo.
- Lectura y estudio de los contenidos de [esta página](https://dlsi.ua.es/~japerez/materials/transformers/attention/) de introducción a los transformers. Realiza, como siempre, al menos dos lecturas complementadas con las notas del profesor. En total, esta parte debería llevarte unas 4 horas 🕒️ de trabajo.
- Tras acabar con las partes anteriores, realiza este [test de evaluación](https://forms.gle/7KDwRtXcrpxsKjHp7) de estos contenidos. Son pocas preguntas y te llevará unos minutos.
- Aprovecha, si te queda tiempo, para repasar los contenidos de la primera sesión.

### Contenidos para la sesión presencial del 10/01/2024

En la clase presencial (5 horas 🕒️ de duración), veremos cómo se implementa en PyTorch el algoritmo de skip-grams <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/skipgram.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>, un modelo de lengua basado en red neuronal hacia delante <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/ffnn.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> y un transformer <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/transformer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> siguiendo las implementaciones que se comentan en [este apartado](https://www.dlsi.ua.es/~japerez/materials/transformers/implementacion/#codigo-para-skip-grams) y los dos siguientes.

La idea es que vayas estudiando y modificando ligeramente los notebooks que vayamos estudiando. En una clase posterior se presentará una práctica más avanzada que implicará modificar el código del transformer.

## Tercera sesión (17 de enero de 2024)

### Contenidos a preparar antes de la sesión del 17/01/2024

Las actividades a realizar antes de esta clase son:

- Lectura y estudio de los contenidos de [esta página](https://dlsi.ua.es/~japerez/materials/transformers/attention2/) sobre, por un lado, el modelo transformer completo (con codificador y descodificador) y, por otro, los posibles usos de una arquitectura que solo incluye el codificador. Como verás, la página te indica qué contenidos has de leer del libro. En particular, tendrás que leer algunas secciones del capítulo sobre traducción automática y otras del capítulo sobre modelos preentrenados, además de alguna sección suelta sobre *beam search* y tokenización en subpalabras. Tras una primera lectura, lee las anotaciones del profesor, cuyo propósito es ayudarte a entender los conceptos clave de cada apartado. Después, realiza una segunda lectura de los contenidos del libro. En total, esta parte debería llevarte unas 4 horas 🕒️ de trabajo.
- Visonado y estudio de la clase de Jesse Mu titulada "[Prompting, Reinforcement Learning from Human Feedback](https://youtu.be/SXpJ9EmG3s4?si=j4B1U2Z-JCyYJwlc)" del curso CS224N de Stanford de 2023 sobre modelos de lengua basados en el descodificador del transformer. En total, esta parte debería llevarte unas 2 horas 🕒️ de trabajo, porque tendrás que tomar notas del vídeo para no tener que verlo cada vez que quieras repasar algo; para tomar notas, te puede venir bien descargar las [diapositivas](https://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture11-prompting-rlhf.pdf) y escribir sobre ellas. Puedes quedarte solo con las ideas básicas de lo que se comenta entre los minutos 39 y 46, porque las ecuaciones del aprendizaje por refuerzo son un tema no prioritario para esta asignatura que verás en otras asignaturas. Es importante que antes de ver el vídeo repases lo que ya estudiaste sobre los [transformers](https://dlsi.ua.es/~japerez/materials/transformers/attention/) como modelo de lengua basado en el descodificador. Que no te confunda que a los modelos basados en codificador también se les conozca a veces con el nombre de modelos de lengua. En este vídeo se habla de las propiedades de modelos basados en descodificador que han sido entrenados para predecir el siguiente token de una secuencia.
- Estudia la descripción sobre [modelos multilingües](https://dlsi.ua.es/~japerez/materials/transformers/attention2/#multilingual-models) que se hace en este apartado de una de las páginas sobre transformers. Es un apartado breve que te llevará unos 🕒️ 15 minutos.
- Tras acabar con las partes anteriores, realiza este [test de evaluación](https://forms.gle/GRK5SLc3STkup8at9) de estos contenidos. Son pocas preguntas y te llevará unos minutos.
- Aprovecha, si te queda tiempo, para repasar todos los contenidos de las sesiones anteriores.

### Contenidos para la sesión presencial del 17/01/2024

En la clase presencial (5 horas 🕒️ de duración), veremos cómo se implementa sobre nuestro código de la arquitectura transformer tanto un modelo de lengua basado en descodificador <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/lmgpt.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> como un modelo de reconocimiento de entidades nombradas <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/nerbert.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> basado en codificador. 
  
Aprovecharemos para repasar algunos aspectos del código de sesiones anteriores y relacionar los aspectos teóricos con los prácticos. Presentaremos también la práctica que tienes que entregar para este bloque de la asignatura.

## Cuarta sesión (19 de enero de 2024)

Esta cuarta sesión es realmente la primera y única sesión del tema de voz. Mira la página sobre [voz](speech.md) para ver los contenidos previos a esta sesión.
