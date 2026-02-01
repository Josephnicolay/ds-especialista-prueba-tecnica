# ds-especialista-prueba-tecnica

Resultados obtenidos para la propuesta de solución de la prueba técnica para el cargo de especialista en ciencia de datos en la empresa Protección SA.

## Descripción del Proyecto

Este repositorio contiene las soluciones para la prueba técnica que incluye tres componentes principales:

1. **Clasificador de Inmuebles de Alto Valor**: Modelo clasificador para identificar grupos de inmuebles de alto valor
2. **Clasificador de Sentimientos (Texto)**: Clasificador de texto para predecir sentimientos usando técnicas tradicionales de NLP
3. **Clasificador de Sentimientos (Transformer)**: Modelo transformer preentrenado para clasificación de sentimientos

## Estructura del Repositorio

```
ds-especialista-prueba-tecnica/
│
├── 01_clasificador_inmuebles_alto_valor/
│   ├── data/
│   │   ├── raw/              # Datos originales sin procesar
│   │   ├── processed/        # Datos procesados y limpios
│   │   └── external/         # Datos de fuentes externas
│   ├── notebooks/            # Notebooks de Jupyter para EDA
│   ├── src/                  # Código fuente del proyecto
│   ├── models/               # Modelos entrenados
│   ├── docs/                 # Documentación
│   └── README.md             # Documentación específica del proyecto
│
├── 02_clasificador_sentimientos_texto/
│   ├── data/
│   │   ├── raw/              # Datos de texto originales
│   │   ├── processed/        # Datos procesados y vectorizados
│   │   └── external/         # Datasets adicionales
│   ├── notebooks/            # Notebooks de experimentación
│   ├── src/                  # Código fuente del proyecto
│   ├── models/               # Modelos y vectorizadores
│   ├── docs/                 # Documentación
│   └── README.md             # Documentación específica del proyecto
│
├── 03_clasificador_sentimientos_transformer/
│   ├── data/
│   │   ├── raw/              # Datos de texto originales
│   │   ├── processed/        # Datos procesados para transformers
│   │   └── external/         # Datasets adicionales
│   ├── notebooks/            # Notebooks de experimentación
│   ├── src/                  # Código fuente del proyecto
│   ├── models/               # Modelos fine-tuned
│   ├── docs/                 # Documentación
│   └── README.md             # Documentación específica del proyecto
│
├── shared/
│   ├── utils/                # Utilidades compartidas
│   ├── config/               # Configuraciones compartidas
│   └── README.md             # Documentación de recursos compartidos
│
├── .gitignore
├── LICENSE
└── README.md                 # Este archivo
```

## Guía de Uso

Cada proyecto tiene su propia carpeta independiente con:
- **data/**: Datos organizados por nivel de procesamiento
- **notebooks/**: Jupyter notebooks para análisis exploratorio y experimentación
- **src/**: Código fuente modular y reutilizable
- **models/**: Modelos entrenados serializados
- **docs/**: Documentación adicional del proyecto

### Requisitos Previos

Los requisitos específicos de cada proyecto se documentan en su respectivo README.

### Estructura de Datos

- `data/raw/`: Almacena los datos originales sin modificar
- `data/processed/`: Almacena los datos procesados listos para modelado
- `data/external/`: Almacena datos de fuentes externas o datasets de referencia

## Navegación

Para más detalles sobre cada componente, consulta los README específicos:

- [Clasificador de Inmuebles de Alto Valor](./01_clasificador_inmuebles_alto_valor/README.md)
- [Clasificador de Sentimientos - Texto](./02_clasificador_sentimientos_texto/README.md)
- [Clasificador de Sentimientos - Transformer](./03_clasificador_sentimientos_transformer/README.md)
- [Recursos Compartidos](./shared/README.md)

## Licencia

Ver archivo [LICENSE](./LICENSE) para más detalles.
