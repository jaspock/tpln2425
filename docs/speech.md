# Architectures for speech

En este bloque se aborda brevemente el estudio de algunos modelos neuronales utilizados para procesar voz. El profesor de este bloque es Juan Antonio Pérez Ortiz. 

Los materiales de clase complementan la lectura de algunos capítulos de un libro de texto ("Speech and Language Processing" de Dan Jurafsky y James H. Martin, borrador de la tercera edición, disponible online) con anotaciones realizadas por el profesor.

## Primera sesión de este bloque (19 de enero de 2024)

### Contenidos a preparar antes de la sesión del 19/01/2024

Las actividades a realizar antes de esta clase son:

- Lectura y estudio de los contenidos de [esta página](https://dlsi.ua.es/~japerez/materials/transformers/speech/) sobre reconocimiento de voz. Como verás, la página te indica qué contenidos has de leer del libro. Tras una primera lectura, lee las anotaciones del profesor, cuyo propósito es ayudarte a entender los conceptos clave del capítulo. Después, realiza una segunda lectura del capítulo del libro. Tras acabar con esta parte, lee la descripción de [arquitecturas modernas](https://dlsi.ua.es/~japerez/materials/transformers/speech/#arquitecturas-modernas-para-el-procesamiento-de-voz) concretas para el reconocimiento de voz. En total, esta parte debería llevarte unas 4 horas 🕒️ de trabajo.
- Después, realiza este [test de evaluación](https://forms.gle/woGk9hkmepMVkrg47) de estos contenidos. Son pocas preguntas (menos que en tests anteriores, de hecho) y te llevará unos minutos.

### Contenidos para la sesión presencial del 19/01/2024

En la clase presencial (2,5 horas 🕒️ de duración), veremos cómo se implementa un sistema de clasificación de voz en PyTorch. Para ello, utilizaremos la librería `torchaudio`, que es parte de PyTorch. En particular, miraremos por encima esta [guía de manipulación de audio con torchaudio](https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html) (únicamente la representación de ondas y la obtención del espectrograma) y la implementación del [clasificador de voz](https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html); ambos documentos tienen un enlace al comienzo a sendos cuadernos de Google Colab. Los dos tutoriales los veremos este curso solo en clase con el propósito de complementar los contenidos teóricos, pero no has de estudiarlos para el examen.
