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
