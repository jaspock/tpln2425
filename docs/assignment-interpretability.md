# Assignment on Mechanistic Interpretability of Transformers

!!! danger

    These materials are temporary and incomplete. If you choose to read them, you do so at your own risk.


**Mechanistic interpretability** in the context of artificial intelligence seeks to provide a motivated explanation of how machine learning models function. It is a crucial approach to building trust in systems and inducing certain behaviors in them. Within the field of mechanistic interpretability, there are many techniques that can be applied to transformers. Here, we will focus on [activation patching][patching].

[patching]: https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=qeWBvs-R-taFfcCq-S_hgMqx

Activation patching *intervenes* in a specific model activation by replacing a *corrupted* activation with a *clean* activation. The effect of this change on the model output is then measured. This helps us identify which activations are important for the model's output and locate possible causes of prediction errors.

In this practice, you will write code to run the smallest version of GPT2 (use the string `gpt2` in the code) with two different inputs: two texts that differ by only one token. The idea is that when the corrupted input is fed to the model, we will intervene in the embedding after a certain layer (one at a time) and patch it with the corresponding embedding from the clean run. Then we will measure how much the prediction of the next token changes compared to the clean run. If the change is significant, we can be confident that the altered activation is important for the prediction. This patching process will be performed for each layer of the model and for each token in the input. With all this information, we will generate a heatmap and draw conclusions. For reasons you will soon understand, both texts must have the same number of tokens.

## Analysis Example

Here is an example to better understand the task. Consider the following input text: "Michelle Jones was a top-notch student. Michelle." If we feed it to GPT2 and study the model's emitted probability for the token following the second appearance of Michelle, we obtain the following (only the 20 most probable tokens are shown): 

| Position | Token index | Token  | Probability |
| -------- | ----------- | ------ | ----------- |
| 1        | 373         | was    | 0.1634      |
| 2        | 5437        | Jones  | 0.1396      |
| 3        | 338         | 's     | 0.0806      |
| 4        | 550         | had    | 0.0491      |
| 5        | 318         | is     | 0.0229      |
| 6        | 290         | and    | 0.0227      |
| 7        | 11          | ,      | 0.0222      |
| 8        | 531         | said   | 0.0134      |
| 9        | 468         | has    | 0.0120      |
| 10       | 635         | also   | 0.0117      |
| 11       | 1625        | came   | 0.0091      |
| 12       | 1297        | told   | 0.0084      |
| 13       | 1422        | didn   | 0.0070      |
| 14       | 2993        | knew   | 0.0067      |
| 15       | 1816        | went   | 0.0061      |
| 16       | 561         | would  | 0.0061      |
| 17       | 3111        | worked | 0.0055      |
| 18       | 750         | did    | 0.0054      |
| 19       | 2486        | Obama  | 0.0053      |
| 20       | 2492        | wasn   | 0.0050      |

As expected, the token "Jones" has a notably high probability. Now consider the corrupted input "Michelle Smith was a top-notch student. Michelle." If we provide this input to GPT2, we expect the probability of "Jones" as the next token to be much lower than before, while the probability of "Smith" will be much higher, which (you can verify) indeed happens. However, we want to go further and understand which embeddings most influence this difference. Given that both inputs have 11 tokens (we will explain how to verify this later) and the transformer in the small GPT2 model has 12 layers, if we focus on the embeddings obtained at the output of each layer, we can patch 11×12 = 132 different embeddings. Therefore, we will calculate 132 times the difference between the logit of "Smith" and the logit of "Jones" in the output of the last token of the input ("Michelle") in the corrupted model. Note that we could also calculate the differences after applying the softmax function, but we will not do so here.

A heatmap representation of the result is shown below:

![Logit difference heatmap](images/mechanistic-michelle.png)

In such a graph, due to the attention mask and the arrangement of the layers, information flows from left to right and top to bottom. You can see that intervening in the first column has no effect on the prediction of the next token, which makes sense, as the embeddings patched have exactly the same values in the clean and corrupted models, given the same preceding context. There also seem to be no changes when patching embeddings from the third to the penultimate column. However, note how intervening in the embeddings of many layers of the second token shifts the prediction towards "Jones" (the color darkens as the logit difference between "Smith" and "Jones" becomes negative because "Jones" has a higher logit). Modifying the embeddings of the last layers of the second token has much smaller effects, as the embedding barely influences the sequence's future. At the last position ("Michelle"), the embeddings of the final layers seem to anticipate the token to predict.

Some additional corrupted texts that may be interesting to explore are, for example, "Jessica Jones was a top-notch student. Michelle" or "Michelle Smith was a top-notch student. Jessica."

## Submission

In this assignment, your task is to write the code to generate graphs and probabilities like those above, propose your own clean and corrupted texts (try to be creative and avoid studying very similar texts or phenomena), perform a similar analysis, and write a report in a document of 1500–2000 words (both limits are strict). In this document, you should present and explain the implemented code, along with your approach, results, and relevant conclusions. Original ideas and additional experiments are welcome. The document in PDF format must be submitted via the UACloud tutoring system **before 23:55 on Sunday, February 4, 2024**. The assignment must be done in pairs. Remember to include both authors' names in the document.

## Base Code 

The base code we will use is from the GPT2 implementation found in Andrej Karpathy's [minGPT][mingpt] repository. His code inspired our transformer model code, so it should not be difficult to understand. You can clone the repository on your computer or work in a Google Colab notebook as described below.

Due to changes in external elements, the current code does not work as is. To make it work, you need to change line 200 of the `mingpt/model.py` file from:

```python
assert len(keys) == len(sd)
```

to:

```python
assert len(keys) == len([k for k in sd if not k.endswith(".attn.bias")])
```

[mingpt]: https://github.com/karpathy/minGPT

## Tokenization

The GPT2 model uses a BPE-based tokenizer that segments the input text into words or smaller units depending on their frequency. The minGPT code allows downloading this tokenizer and using it to segment texts. The following code shows how to tokenize a text to obtain its indices and vice versa.

```python
from mingpt.bpe import BPETokenizer

input = "Michelle Jones was a top-notch student. Michelle"
print("Input:", input)
bpe = BPETokenizer()
# bpe() gets a string and returns a 2D batch tensor 
# of indices with shape (1, input_length)
tokens = bpe(input)[0]
print("Tokenized input:", tokens)
input_length = tokens.shape[-1]
print("Number of input tokens:", input_length)
# bpe.decode gets a 1D tensor (list of indices) and returns a string
print("Detokenized input from indices:", bpe.decode(tokens))  
tokens_str = [bpe.decode(torch.tensor([token])) for token in tokens]
print("Detokenized input as strings: " + '/'.join(tokens_str))
```

## Implementation Details

The following are some implementation details that may be useful but are not required to follow.

To write code that allows activation patching, you will need to focus on the files `mingpt/model.py` and `generate.ipynb`. If you are working locally without using a notebook (recommended), copy the code from `generate.ipynb` into a `generate.py` file that you can execute from the command line.

You can also work directly in a Google Colab session. Here is a [project][proyectocolab] (access with your `gcloud.ua.es` account) with instructions on how to use it for development. However, developing locally is much more convenient (among other things, you can work with a better text editor than Colab’s and also debug). Even if you don’t have a GPU, the code runs fine on a CPU and only takes a few seconds longer than on a GPU, as it only works with one text and a not excessively large model.

Add to the transformer’s `forward` function code that allows saving (depending on the value of a boolean *flag* passed as a parameter) the activations of each layer and each position into an instance variable. Remember to make a deep copy of the embeddings rather than only saving a reference that could be overwritten later; for this, check PyTorch’s `.detach().clone()` sequence of calls. Also, add code that allows (again based on a boolean parameter) patching the embedding of a specific layer and position.

Additionally, modify the `forward` function to store the logits of the last token, which contain the information we are interested in regarding the prediction of the next token. You can save this information in an attribute that can later be accessed from outside the class. Note that you only need the vector corresponding to the last token.

Add code to the `generate.py` file to tokenize the clean text, pass it through the model via the `generate` function (asking the model to save the intermediate embeddings), and display the most probable continuations based on the logits of the last token. Keep in mind that if you want to know the probability of a continuation like the token "Jones," for example, you need to find the index of that token in the vocabulary by prefixing it with a space (`index = bpe(' Jones')`). This is because the BPE tokenizer handles tokens at the beginning of a sequence differently from those in the middle. Once you have the token’s index, you can access the corresponding position in the logits vector and obtain the unnormalized probability of it being the continuation.

Then, you can work with the corrupted text. Include a nested loop that iterates over all layers and all positions and calls `generate` each time, passing the layer and position where the intervention should be performed. At each step, compute the appropriate logit difference and store it in a difference matrix.

Finally, use the `matshow` function from `matplotlib` to visualize the difference matrix.

[proyectocolab]: https://colab.research.google.com/drive/1dq2EClvIbEtoEnHWoAXZQTArJDHivQly?usp=sharing

## A More Informal Explanation

The following informal explanation may help you better understand the objective of the assignment.

For simplicity, consider the phrase "a b c" and its corrupted version "d e f." In general, there will be many more tokens in common, but this makes the following discussion clearer. Assume the transformer-based neural model has 5 attention layers. We want to study which embeddings are important for predicting that the token "X" follows these phrases.

First, modify the transformer’s `forward` function (in the `GPT` class) to store (e.g., in a list of lists of tensors) the 3×5=15 embeddings generated at the output of each layer when processing the phrase "a b c." The assignment provides some details because you cannot simply store a reference to the tensors, as they will be modified the next time `forward` is called. Instead, you need to clone the tensors (a "defensive copy"). This will leave you with the 15 tensors (embeddings) for the clean phrase.

Also, save the logits after the last layer. In particular, you only need the logits for the final position (i.e., those corresponding to the token "c"), which provide a measure of the probability of the next token, i.e., the token following "c." Remember that these logits are not actual probabilities (they are values like -11.1, -0.5, 0.78, or 2.32323) because the softmax function has not been applied, but working with them is more convenient due to their broader range. However, the study could equally be conducted using strict probabilities. In reality, you don’t even need to save all the logits, only the scalar corresponding to token "X," as it is the only one you will use later.

Now feed the model the corrupted version "d e f," ensuring that it does not overwrite the stored embeddings from the clean phrase. The corrupted phrase must have the same number of tokens as the clean one for the following discussion to make sense. The idea is to modify only one of the 15 embeddings generated while processing the corrupted phrase. For example, if we focus on the embedding of the first token ("d") after the first layer, the `forward` function should operate "almost" normally. However, when obtaining the output of the first layer and before passing it as input to the second layer, the embedding corresponding to the first word (and only that) should be modified and replaced with the corresponding embedding (from the same layer and position) saved for the clean phrase (in this case, the embedding saved after the first layer for the token "a"). This ensures that the second layer receives as input the embedding generated for "a" instead of "d."

After intervening in the embedding of position 1 after layer 1, the rest of the model operates without any "hiccups." As before, examine the logits for predicting the token following the last token of the corrupted phrase (i.e., "f"). Focus on the value of the logit for the prediction of token "X." The difference between this value and the one saved for the clean phrase provides insight into the relevance of the embedding at layer 1, position 1, for predicting token "X." The assignment shows that some embeddings are much more relevant than others, and you need to conduct a similar study with different phrases.

If you repeat the above operation for the other 14 embeddings (calling the `forward` function 14 more times), you will end up with 15 logit differences (15 scalar values) that can be represented in a 3×5 heatmap as seen above.

Finally, note that this discussion simplifies the task described earlier in the assignment. There, it was proposed to calculate the difference between the logit of "Smith" and the logit of "Jones" in the output of the last token in the corrupted model. This approach provides slightly more information than the difference explained here, which is the difference between the prediction of a single token ("Jones") in the clean and corrupted phrases, rather than two tokens in the corrupted phrase. Either approach is valid for arriving at the conclusions we are interested in: that in the corrupted phrase, the logit of "Jones" becomes much lower except for certain interventions. If you want your heatmap to match the one in the assignment, follow the approach based on the two tokens "Jones" and "Smith."

## Further Knowledge

The above is just one of many analyses proposed within mechanistic interpretability. For this practice, you are not expected to go beyond this. However, if you are interested in learning about a couple more analyses, you can check out [this tutorial][lines50]. Note that although the tutorial uses a library for activation patching, in this practice, you are not allowed to use any library for this and must implement it directly in the minGPT code. A much more detailed review of mechanistic interpretability can be found in [this work][nanda] by Neel Nanda.

[lines50]: https://www.lesswrong.com/posts/hnzHrdqn3nrjveayv/how-to-transformer-mechanistic-interpretability-in-50-lines
[nanda]: https://www.neelnanda.io/mechanistic-interpretability/glossary

# Práctica sobre interpretabilidad mecanicista de transformers

La *interpretabilidad mecanicista* en el contexto de la inteligencia artificial intenta dar una explicación motivada del funcionamiento de los modelos de aprendizaje automático. Es una propuesta muy importante de cara a generar confianza en los sistemas e inducir ciertos comportamientos en ellos. Dentro del campo de la interpretabilidad mecanicista existen un buen número de técnicas que se pueden aplicar a los transformers. Aquí nos centraremos en el [parcheado de activaciones][patching].

[patching]: https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=qeWBvs-R-taFfcCq-S_hgMqx

El parcheado de activaciones *interviene* en una activación específica de un modelo mediante la sustitución de una activación *corrompida* con una activación *limpia*. Se mide entonces cómo afecta el cambio a la salida del modelo. Esto nos permite identificar qué activaciones son importantes para el resultado del modelo y localizar posibles causas de errores en la predicción. 

En nuestro caso particular, vas a escribir código que ejecute la versión más pequeña de GPT2 (usa la cadena `gpt2` en el código) con dos entradas diferentes: dos textos que solo se diferencien en un único token. La idea es que al proporcionar al modelo la entrada corrompida, intervendremos en el embedding tras una cierta capa (uno solo cada vez) y lo parchearemos con el embedding correspondiente de la ejecución limpia. Luego mediremos cuánto cambia la predicción del siguiente token respecto a la ejecución limpia. Si el cambio es significativo, entonces podemos estar seguros de que la activación que hemos alterado es importante para la predicción. Este proceso de parcheado lo realizaremos para cada capa del modelo y para cada token de la entrada. Con toda esta información, obtendremos una gráfica y sacaremos conclusiones. Por motivos que entenderás en un momento, los dos textos han de tener el mismo número de tokens.

## Ejemplo de análisis

Daremos un ejemplo para que se entienda mejor. Considera el siguiente texto de entrada: "Michelle Jones was a top-notch student. Michelle". Si se lo damos a GPT2 y estudiamos la probabilidad emitida por el modelo para el token que sigue a la segunda aparición de Michelle, obtendremos lo siguiente (solo se muestran los 20 tokens más probables): 

| Position | Token index | Token  | Probability |
| -------- | ----------- | ------ | ----------- |
| 1        | 373         | was    | 0.1634      |
| 2        | 5437        | Jones  | 0.1396      |
| 3        | 338         | 's     | 0.0806      |
| 4        | 550         | had    | 0.0491      |
| 5        | 318         | is     | 0.0229      |
| 6        | 290         | and    | 0.0227      |
| 7        | 11          | ,      | 0.0222      |
| 8        | 531         | said   | 0.0134      |
| 9        | 468         | has    | 0.0120      |
| 10       | 635         | also   | 0.0117      |
| 11       | 1625        | came   | 0.0091      |
| 12       | 1297        | told   | 0.0084      |
| 13       | 1422        | didn   | 0.0070      |
| 14       | 2993        | knew   | 0.0067      |
| 15       | 1816        | went   | 0.0061      |
| 16       | 561         | would  | 0.0061      |
| 17       | 3111        | worked | 0.0055      |
| 18       | 750         | did    | 0.0054      |
| 19       | 2486        | Obama  | 0.0053      |
| 20       | 2492        | wasn   | 0.0050      |

Como era de esperar, el token "Jones" tiene una probabilidad notablemente elevada. Ahora, considera la entrada corrompida "Michelle Smith was a top-notch student. Michelle". Si le damos esta entrada a GPT2, esperamos que la probabilidad de "Jones" como continuación del texto sea mucho menor que antes y que la de "Smith" sea mucho mayor, lo que (puedes comprobarlo) efectivamente ocurre. Pero queremos ir más allá y saber qué embeddings son los que más influyen en esta diferencia. Dado que ambas entradas tienen 11 tokens (más adelante explicaremos cómo averiguarlo) y que el transformer del modelo GPT2 pequeño tiene 12 capas, si nos centramos en los embeddings que se obtienen a la salida de cada capa, podemos parchear 11×12 = 132 embeddings diferentes. Calcularemos, por tanto, 132 veces la diferencia entre el logit de "Smith" y el logit de "Jones" en la salida del último token de la entrada ("Michelle") en el modelo corrompido. Observa que también podríamos calcular las diferencias tras aplicar la función softmax, pero en este caso no lo haremos.

Una representación en forma de mapa de calor del resultado es la siguiente:

![Mapa de calor de diferencias de logits](images/mechanistic-michelle.png)

Recuerda que en un gráfico como el anterior, debido a la máscara de atención y a la disposición de las capas, la información fluye de izquierda a derecha y de arriba a abajo. Puedes ver cómo intervenir en la primera columna no tiene efectos en la predicción del siguiente token, lo que tiene todo el sentido, ya que los embeddings que se parchean tienen exactamente los mismos valores en el modelo limpio y en el corrompido, ya que el contexto anterior es el mismo. Tampoco parece haber cambios al parchear los embeddings de la tercera a la antepenúltima columna. Sin embargo, observa cómo al intervenir los embeddings de muchas capas del segundo token, la predicción se decanta hacia "Jones" (el color se hace oscuro cuando la diferencia entre el logit de "Smith" y el de "Jones" se va haciendo negativa porque "Jones" tiene un logit mayor). Modificar los embeddings de las últimas capas del segundo token tiene efectos mucho menores, ya que el embedding apenas puede influir en el futuro de la secuencia. En la última posición ("Michelle") se observa que los embeddings de las capas finales van anticipándose al token que tienen que predecir.

Algunos textos corrompidos adicionales que puede ser interesante explorar son, por ejemplo, "Jessica Jones was a top-notch student. Michelle" o "Michelle Smith was a top-notch student. Jessica".

## Entrega

En esta práctica se trata de que programes el código que te permite obtener gráficas y probabilidades como las anteriores, propongas tus propios textos limpios y corrompidos (intenta tirar de creatividad y no estudiar textos o fenómenos muy similares), realices un análisis parecido al anterior y escribas un informe dentro de un documento de entre 1500-2000 palabras (ambos límites son estrictos) en el que presentes y comentes el código que has implementado, además de presentar tu enfoque, los resultados y las conclusiones pertinentes. Serán bienvenidas las ideas originales y los experimentos adicionales que se te ocurran. El documento en formato PDF ha de ser enviado por el sistema de tutoría de UACloud **antes de las 23.55 del domingo 4 de febrero de 2024**. La práctica ha de ser realizada en parejas. Recuerda poner el nombre de ambos autores en el documento.

## Código base 

El código base que usaremos es el de la implementación de GPT2 que se encuentra en el repositorio [minGPT][mingpt] de Andrej Karpathy. Su código es el que ha inspirado nuestro código del modelo transformers, por lo que no te resultará difícil entenderlo. Puedes clonar el repositorio en tu ordenador o trabajar en un cuaderno de Google Colab como se indica más abajo.

Debido a cambios en elementos externos, el código actual no funciona tal cual. Para que funcione, tienes que cambiar la línea 200 del fichero `mingpt/model.py` de:

```python
assert len(keys) == len(sd)
```

a:

```python
assert len(keys) == len([k for k in sd if not k.endswith(".attn.bias")])
```

[mingpt]: https://github.com/karpathy/minGPT

## Tokenization

El modelo GPT2 usa un tokenizador basado en BPE que trocea el texto de entrada en palabras o en unidades inferiores dependiendo de su frecuencia. El código de minGPT permite descargar dicho tokenizador y usarlo para segmentar los textos. El siguiente código muestra cómo tokenizar un texto para obtener sus índices y viceversa.

```python
from mingpt.bpe import BPETokenizer

input = "Michelle Jones was a top-notch student. Michelle"
print("Input:", input)
bpe = BPETokenizer()
# bpe() gets a string and returns a 2D batch tensor 
# of indices with shape (1, input_length)
tokens = bpe(input)[0]
print("Tokenized input:", tokens)
input_length = tokens.shape[-1]
print("Number of input tokens:", input_length)
# bpe.decode gets a 1D tensor (list of indices) and returns a string
print("Detokenized input from indices:", bpe.decode(tokens))  
tokens_str = [bpe.decode(torch.tensor([token])) for token in tokens]
print("Detokenized input as strings: " + '/'.join(tokens_str))
```

## Detalles de implementación

Lo siguiente son algunos detalles de implementación que te pueden ser útiles, pero que no es necesario que sigas. 

Para conseguir un código que te permita realizar el parcheado de activaciones te tendrás que centrar en los ficheros `mingpt/model.py` y `generate.ipynb`. Si trabajas en local sin usar un *notebook* (recomendado) copia el código de `generate.ipynb` en un fichero `generate.py` que puedas ejecutar desde la línea de órdenes.

También puedes trabajar directamente en una sesión de Google Colab. Aquí tienes un [proyecto][proyectocolab] (accede con tu cuenta de `gcloud.ua.es`) con instrucciones sobre cómo usarlo para desarrollar. Sin embargo, es mucho más cómodo desarrollar en local (entre otras cosas, puedes trabajar con un mejor editor de texto que el de Colab y también depurar). Incluso si no tienes una GPU, el código funciona sin problemas sobre CPU y solo tarda unos segundos más que sobre GPU al solo trabajar con un texto y con un modelo no excesivamente grande.

Añade a la función `forward` del transformer código que permita salvar (según el valor de cierto *flag* booleano recibido como parámetro) en una variable de instancia las activaciones de cada capa y cada posición. Recuerda hacer una copia profunda de los embeddings y no guardar únicamente una referencia que puede ser sobreescrita posteriormente; para ello, consulta la secuencia de llamadas `.detach().clone()` de PyTorch. Añade también código que permita (de nuevo en base a un parámetro booleano) parchear el embedding de una capa y posición concretas. 

Añade también a la función `forward` código que guarde los logits del último token, que contienen la información que nos interesa sobre la predicción del siguiente token. Puedes guardar esta información en un atributo que luego puedes acceder desde el exterior de la clase. Observa que solo te interesa el vector correspondiente al último token.

Añade código al fichero `generate.py` que divida el texto limpio en tokens, lo pase por el modelo a través de la función `generate` (pidiéndole al modelo que guarde los embeddings intermedios) y muestre las continuaciones más probables a partir de los logits del último token. Ten en cuenta que si quieres saber la probabilidad de una continuación como el token "Jones", por ejemplo, has de buscar el índice de dicho token en el vocabulario anteponiéndole un espacio en blanco (`index = bpe(' Jones')`). Esto es así porque el segmentador de BPE trata de forma diferente los tokens que aparecen al principio de la secuencia y los que aparecen en medio. Una vez tengas el índice del token, puedes acceder a la posición correspondiente del vector de logits y obtener la probabilidad no normalizada de que sea la continuación.

Después, puedes trabajar con el texto corrupto. Incluye un doble bucle que itere sobre todas las capas y todas las posiciones y llame cada vez a `generate` pasándole la capa y la posición en la que realizar la intervención. En cada paso, evalúa la diferencia de logits oportuna y guárdala en una matriz de diferencias.

Usa finalmente la función `matshow` de `matplotlib` para visualizar la matriz de diferencias.

[proyectocolab]: https://colab.research.google.com/drive/1dq2EClvIbEtoEnHWoAXZQTArJDHivQly?usp=sharing

## Una explicación más informal

La siguiente explicación informal puede que te ayude a entender mejor el objetivo de la práctica.

Considera para simplificar la frase "a b c" y la versión corrompida "d e f". En general, habrá muchos más tokens en común, pero así queda todo más claro en la siguiente discusión. Considera que el modelo neuronal basado en el transformer tiene 5 capas de atención. Considera que vamos a estudiar qué embeddings son importantes para la predicción de que tras estas frases vaya el token "X".

Se trata primero de que permitas que en la función forward del transformer (clase `GPT`) se puedan guardar (por ejemplo en una lista de listas de tensores) los 3x5=15 embeddings que se generan a la salida de cada una de las capas cuando se procesa la frase "a b c". En el enunciado se dan algunos detalles porque no puedes guardar simplemente la referencia a los tensores, ya que se modificarán la próxima vez que llames a forward, sino que has de clonar los tensores (lo que se llama "copia defensiva"). Con esto tendrás almacenados los 15 tensores (embeddings) de la frase limpia.

Guárdate también los logits tras la última capa. En particular, solo necesitarás los de la última posición (es decir, los logits correspondientes al token "c"), que te dan una medida de la probabilidad del siguiente token, es decir, del token que irá tras "c". Recuerda que estos logits no son realmente probabilidades (son valores como -11.1, -0.5, 0.78, o 2.32323) porque no se les ha aplicado la función softmax, pero trabajar con ellos es más cómodo que trabajar con las probabilidades porque tenemos valores con un rango más amplio. No obstante, el estudio podría hacerse igualmente con probabilidades estrictas. En realidad, ni siquiera necesitas guardarte todos los logits, sino solo el escalar que corresponde al token "X" porque es lo único que usarás después.

Ahora le das al modelo la versión corrompida "d e f", indicándole que no sobreescriba la copia de los embeddings que obtuvimos con la frase limpia. La frase corrompida ha de tener el mismo número de tokens que la limpia para que la siguiente discusión tenga sentido. La idea es modificar uno solo de los 15 embeddings que se producen mientras se procesa la frase sucia. Si, por ejemplo, nos centramos en el embedding del primer token ("d") tras la primera capa, se trataría de que el código de la función forward opere "casi" de la forma normal, pero cuando se obtenga la salida de la primera capa y antes de pasarla como entrada a la segunda capa, se ha de modificar el embedding correspondiente a la primera palabra (solo ese) y sustituirlo por el embedding correspondiente (de la misma capa y posición) que te guardaste para la frase limpia (es decir, en este caso, sería el embedding que te guardaste tras la primera capa para el token "a"). Con esto, la segunda capa recibirá como entrada el embedding que se generó para "a" en lugar del de "d".

Tras intervenir en el embedding de la posición 1 tras la capa 1, el resto del modelo trabaja sin ningún "contratiempo". De la misma manera que antes, ahora miramos los logits de la predicción del token que va tras el último token de la frase corrompida (es decir, "f"). Y nos centramos en el valor del logit de la predicción del token "X". La diferencia entre este valor y el que nos guardamos para la frase limpia nos da una idea de cómo de relevante es el embedding de la capa 1 y posición 1 en la predicción del token "X". En el enunciado se muestra cómo algunos embeddings son mucho más relevantes que otros. Y tú tienes que hacer un estudio similar con diferentes frases.

Si repites la operación anterior con los otros 14 embeddings (llamando 14 veces más a la función forward), terminarás teniendo 15 diferencias de logits (15 valores escalares) que puedes representar en un mapa de calor de 3x5 como se ve más arriba.

Finalmente, ten en cuenta que la discusión de este apartado tiene una pequeña simplificación respecto a lo que se pide en el enunciado de más arriba. Allí se proponía calcular la diferencia entre el logit de "Smith" y el logit de "Jones" en la salida del último token en el modelo corrompido, lo que da un poco más de información que la diferencia que hemos explicado en este apartado, es decir, la diferencia entre la predicción de un solo token ("Jones") en la frase limpia y la corrompida, en lugar de dos tokens en la frase corrompida. En realidad, cualquiera de las dos opciones es válida para llegar a las conclusiones que nos interesan: que en la frase corrompida, el logit de "Jones" se hace mucho menor excepto para ciertas intervenciones. Si quieres que tu mapa de calor coincida con el de este enunciado, sigue el enfoque basado en los dos tokens "Jones" y "Smith".

## Ampliar conocimientos 

Lo anterior es solo uno de los múltiples análisis que se han propuesto dentro de la interpretabilidad mecanicista. Para esta práctica no se espera que vayas más allá de esto, pero si te interesa conocer un par de análisis más puedes consultar [este tutorial][lines50]. Observa que aunque el tutorial usa una librería para parchear las activaciones, en esta práctica no puedes usar ninguna librería para ello y lo has de hacer directamente sobre el código de minGPT. Una revisión mucho más detallada sobre la interpretabilidad mecanicista se puede encontrar en [este trabajo][nanda] de Neel Nanda.

[lines50]: https://www.lesswrong.com/posts/hnzHrdqn3nrjveayv/how-to-transformer-mechanistic-interpretability-in-50-lines
[nanda]: https://www.neelnanda.io/mechanistic-interpretability/glossary