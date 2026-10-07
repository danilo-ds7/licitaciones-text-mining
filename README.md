# Text mining + Transformers sobre licitaciones de Mercado Público

Material del **Módulo V · Arquitecturas de Atención y Transformers para la Gestión de Complejidad** (IA Generativa para la Gestión Industrial).
Docente: Danilo Gómez Correa.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danilo-ds7/licitaciones-text-mining/blob/main/TextMining_Transformers_Licitaciones.ipynb)

## Contenido

| Archivo | Qué es |
|---|---|
| `TextMining_Transformers_Licitaciones.ipynb` | Notebook de clase (Colab): exploración clásica y visualizaciones con Transformers |
| `app.py` | App Streamlit interactiva con los mismos análisis |
| `data/licitaciones.csv` | 5.021 licitaciones reales (1 al 6 de octubre de 2026); 213 con detalle completo |
| `data/embeddings.npy` | Embeddings de los nombres (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensiones) |
| `data/mapa_embeddings.csv`, `data/topicos_bertopic.csv` | Coordenadas UMAP y temas BERTopic precalculados |
| `scripts/` | Scripts que generaron los archivos de `data/` |

## Flujo

1. **Exploración clásica** (qué dice el texto): nube de palabras, frecuencias y TF-IDF, n-gramas, red de co-ocurrencia, temas con LDA.
2. **Visualizaciones con Transformers** (qué significa): mapa de embeddings (UMAP), clusters semánticos con BERTopic, mapa de atención, extracción estructurada y dashboard.

## Ejecutar la app localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fuente de datos

API pública de Mercado Público (ChileCompra), `api.mercadopublico.cl`, consultada el 7 de octubre de 2026.
Para descargar otros días se necesita un ticket propio (gratuito); ver el Anexo A del notebook.
Los datos no incluyen nombres de funcionarios ni datos de contacto.
