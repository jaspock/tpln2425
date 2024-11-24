# Architectures for Speech

!!! danger

    These materials are temporary and incomplete. If you choose to read them, you do so at your own risk.


In this module, we briefly study some neural models used to process speech. The professor for this module is Juan Antonio Pérez Ortiz.

Class materials complement the reading of some chapters from a textbook ("Speech and Language Processing" by Dan Jurafsky and James H. Martin, third edition draft, available online) with annotations made by the professor.

## First session of this module (January 19, 2024)

### Contents to prepare before the session on 01/19/2024

The activities to complete before this class are:

- Reading and studying the contents of [this page](https://dlsi.ua.es/~japerez/materials/transformers/speech/) on speech recognition. As you will see, the page indicates which contents you should read from the book. After a first reading, read the professor's annotations to help you understand the key concepts of the chapter. Then, perform a second reading of the chapter from the book. After finishing this part, read the description of [modern architectures](https://dlsi.ua.es/~japerez/materials/transformers/speech/#arquitecturas-modernas-para-el-procesamiento-de-voz) specific to speech recognition. In total, this part should take you about 4 hours 🕒️ of work.
- Then, take this [assessment test](https://forms.gle/woGk9hkmepMVkrg47) on these contents. There are few questions (fewer than in previous tests, in fact), and it will only take a few minutes.

### Contents for the in-person session on 01/19/2024

In the in-person class (2.5 hours 🕒️ in duration), we will see how to implement a speech classification system in PyTorch. To do this, we will use the `torchaudio` library, which is part of PyTorch. Specifically, we will briefly look at this [guide to audio manipulation with torchaudio](https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html) (focusing only on waveform representation and spectrogram extraction) and the implementation of the [speech classifier](https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html). Both documents include links at the beginning to corresponding Google Colab notebooks. 

These two tutorials will only be covered in class to complement the theoretical contents, but you do not need to study them for the exam.


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

