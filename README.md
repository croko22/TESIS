# Generación de Preguntas Impulsada por Intenciones con T5

Este repositorio documenta el desarrollo de una estrategia para generar preguntas adaptadas a intenciones instruccionales, utilizando transformadores tipo T5.

## Objetivo

Diseñar un generador de preguntas que pueda adaptar su salida según el propósito educativo: evaluación, reflexión o refuerzo. La idea central es condicionar la formulación de preguntas sin necesidad de reentrenar completamente el modelo.


### Informe de Avance: Exploración de Técnicas para Control Intencional

En esta fase se evaluaron distintas variantes para mejorar la alineación entre el tipo de intención y la pregunta generada. Se trabajó con recuperación de contexto, ajustes de generación y plantillas especializadas.

## Técnicas Aplicadas

* **Prompts guiados por intención**
  Se diseñaron estructuras específicas de entrada según el tipo de pregunta esperada:

  * *Recordatorio:* `"¿Qué información esencial recordarías sobre: {contexto}?"`
  * *Reflexión:* `"¿Qué pregunta invitaría a la reflexión sobre: {contexto}?"`
  * *Evaluación:* `"Pregunta de prueba basada en: {contexto}"`

* **RAG simple con Sentence-BERT**
  Para enriquecer el contexto, se integró un esquema de recuperación utilizando `SentenceTransformer` (`paraphrase-MiniLM-L6-v2`).
  Se buscaban los tres fragmentos más cercanos a la entrada usando similitud coseno, y se concatenaban antes de la generación.

* **Modificación del proceso de decodificación**
  Se ajustaron los hiperparámetros de generación:

  * `num_beams = 6`
  * `length_penalty = 0.8`
  * `no_repeat_ngram_size = 2`
  * `max_length = 128`


### Resultados

| Métrica                   | Base (sin ajustes) | Versión con recuperación + plantillas |
| BLEU-4                    | 0.21               | **0.37**                              |
| ROUGE-L                   | 0.38               | **0.49**                              |
| Coincidencia de intención | 61%                | **86%**                               |


### Comparación con variantes conocidas

| Variante                   | BLEU-4 | ROUGE-L | Alineación con intención |
| Generación directa         | 0.21   | 0.38    | 61%                      |
| Versión con RAG            | 0.37   | 0.49    | 86%                      |
| MixQG (referencia externa) | 0.43   | 0.50    | 90%                      |

A pesar de no entrenar sobre datasets masivos o aplicar métodos complejos de entrenamiento, los resultados muestran mejoras sostenidas al introducir señales contextuales adicionales.


### Conclusiones preliminares

La combinación de contexto ampliado + estructura condicional mejora el alineamiento semántico con la intención deseada. No se aplicaron cambios internos al modelo base. Los avances abren la posibilidad de incorporar entrenamientos supervisados por intención o esquemas de control más finos sin alterar el núcleo del generador.


## Estado Actual

* Implementación de fine-tuning con `t5-small` sobre subconjuntos del dataset SQuAD v2.
* Uso de pares “contexto + respuesta” para generar preguntas durante entrenamiento.
* Generación de ejemplos evaluados manualmente y con métricas automáticas.
* Incorporación de recuperación contextual (RAG básico).
* Ajustes en decodificación para mejorar coherencia estructural.
* Evaluación cuantitativa con métricas automáticas.


## Estructura del repositorio

```
├── data/                 
│   └── samples.csv
├── notebooks/
│   ├── 01-intent-examples.ipynb
│   ├── 02-t5-generation.ipynb
│   ├── full-fine-tune-t5-small-squad-qg-v2.ipynb
│   └── evaluate-t5-small-v1.ipynb
├── src/                  
│   ├── generator.py       
│   ├── intent_mapper.py   
│   ├── model.py           
│   ├── train.py           
│   ├── generate.py        
│   ├── evaluate.py        
│   └── utils.py           
├── README.md
└── requirements.txt
```


## Instrucciones de ejecución

1. Crear entorno virtual:

```bash
python -m venv venv && source venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Entrenar:

```bash
python src/train.py
```

4. Generar preguntas y evaluar:

```bash
python src/generate.py
python src/evaluate.py
```

**Autor:** Kevin Joaquin Chambi Tapia
**Curso:** Proyecto Final de Carrera II
