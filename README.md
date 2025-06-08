# Intent-Driven Question Generation with T5

Este repositorio documenta el avance de implementación de un sistema de generación automática de preguntas basado en modelos T5, con un enfoque en escenarios educativos adaptativos guiados por intención.

## Objetivo

Desarrollar un modelo de generación de preguntas controlado por intenciones, utilizando un enfoque basado en transformers del tipo T5. El sistema busca adaptar las preguntas generadas según el propósito instruccional, como evaluación, reflexión o afianzamiento de conceptos.

## Avance actual

- Se implementó un pipeline de fine-tuning sobre el modelo `t5-small` utilizando el dataset SQuAD v2.
- El modelo fue configurado para tareas de generación de preguntas, a partir de pares `"contexto + respuesta"` como entrada.
- Se utilizaron las herramientas provistas por HuggingFace (`Trainer`, `Seq2SeqTrainingArguments`) para gestionar el entrenamiento.
- Se generaron preguntas de prueba con distintos ejemplos de entrada, verificando la calidad semántica de los resultados.

## Estructura del repositorio

```
├── data/                   # Datos procesados y ejemplos
├── notebooks/
│   └── full-fine-tune-t5-small-squad-qg-v2.ipynb
├── src/                    # Módulos futuros para el pipeline final
│   ├── generator.py        # Wrapper del modelo T5 con control por intención
│   └── intent\_mapper.py    # Módulo para mapear contexto a intención
├── README.md
└── requirements.txt
```

## Ejecución

1. Crear un entorno virtual:
   ```bash
   python -m venv venv && source venv/bin/activate
   ```

2. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar el notebook de fine-tuning:

   * `notebooks/full-fine-tune-t5-small-squad-qg-v2.ipynb`

## Próximos pasos

* Implementar un módulo de clasificación de intención a partir de entradas textuales.
* Adaptar los prompts de entrada al modelo según el tipo de intención detectado.
* Evaluar los resultados generados utilizando métricas automáticas (BLEU, ROUGE) y validación manual básica.

---

**Autor:** Kevin Joaquin Chambi Tapia
**Curso:** Proyecto Final de Carrera II