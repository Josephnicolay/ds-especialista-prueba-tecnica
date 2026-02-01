# Clasificador de Sentimientos - Transformer

## Descripción
Implementación de un clasificador de sentimientos utilizando modelos transformer preentrenados (BERT, RoBERTa, etc.).

## Estructura del Proyecto

```
03_clasificador_sentimientos_transformer/
├── data/
│   ├── raw/          # Datos de texto originales
│   ├── processed/    # Datos procesados para transformers
│   └── external/     # Datasets adicionales
├── notebooks/        # Jupyter notebooks para experimentación
├── src/              # Código fuente del proyecto
├── models/           # Modelos fine-tuned
└── docs/             # Documentación del proyecto
```

## Objetivo
Desarrollar un clasificador de sentimientos usando modelos transformer preentrenados, aprovechando transfer learning para mejorar el rendimiento sobre el clasificador tradicional.

## Tareas Principales
1. Selección del modelo transformer preentrenado apropiado
2. Preparación de datos para el formato requerido por transformers
3. Fine-tuning del modelo preentrenado
4. Evaluación del modelo
5. Comparación con el clasificador tradicional
6. Análisis de ventajas y limitaciones del enfoque transformer
