# Scientific RAG API

Pipeline de Retrieval-Augmented Generation (RAG) sobre un pequeño corpus de 
PDFs científicos (teledetección hiperespectral/índices de vegetación), 
expuesto como un servicio FastAPI con respuestas citadas.

Algunas de las principales características son:

- **Chunking de Markdown en dos fases** con atribución exacta de
  páginas por chunk.
- **Recuperación híbrida**: embeddings densos (`BAAI/bge-small-en-v1.5` +
  ChromaDB) fusionados con BM25 vía Reciprocal Rank Fusion
- **LLM multi-proveedor** (OpenAI, Anthropic, DeepSeek, Ollama) gestionado por
  una factoría basada en `.env`.
- **Citación estricta**: los marcadores `[n]` del LLM se mapean 1:1 a los chunks
  recuperados (con documento, páginas y sección).

## Estructura del proyecto

```
├── app/                    # Capa FastAPI (solo preocupaciones HTTP)
│   ├── config.py           # pydantic-settings, lee .env
│   ├── schemas.py          # modelos de request/response (fuente de la OpenAPI)
│   ├── dependencies.py     # wiring: singletons + re-detección perezosa del LLM
│   ├── main.py             # factory de la app, lifespan, mapeo excepción -> HTTP
│   └── routers/            # health.py, retrieve.py, query.py
├── rag/                    # librería core (sin dependencia de FastAPI)
│   ├── ingestion/          # loader.py (pymupdf4llm), chunker.py (2 fases), indexer.py
│   ├── retrieval/          # embeddings.py, store.py (Chroma), hybrid.py (BM25 + RRF)
│   ├── llm/                # providers.py (REST httpx), factory.py, prompts.py
│   ├── pipeline.py         # retrieve -> prompt -> generate -> cite
│   └── exceptions.py       # errores de dominio mapeados a estados HTTP
├── scripts/ingest.py       # CLI de ingesta offline (reconstruye el índice)
├── tests/                  # unitarios (chunker, RRF) + integración (API vía TestClient)
├── data/raw_pdfs/          # los 5 artículos fuente
└── data/chroma/            # vector store generado (gitignored)
```

La ingesta se separa deliberadamente de la API: es lenta (carga del
modelo, parseo, embeddings) y nunca debe bloquear las peticiones.

## Preguntas técnicas

### 1. Estrategia de chunking y embeddings

**Chunking estructural y recursivo.** `pymupdf4llm` convierte cada PDF
página a página a Markdown, preservando cabeceras y tablas. La fase 1
(`MarkdownHeaderTextSplitter`) recorre esa estructura y nunca corta por
tamaño: cada fragmento es una sección lógica completa (Abstract, Materials
and Methods, Results...) con la jerarquía de cabeceras inyectada en sus
metadatos. La fase 2 (`RecursiveCharacterTextSplitter`, `chunk_size=750`,
`chunk_overlap=140`) redimensiona esas secciones para que los embeddings se
mantengan enfocados en un párrafo denso o tabla, en lugar de
una sección demasiado grande.

Esta elección se debe a que el corpus está compuesto por literatura de agronomía 
y teledetección, con párrafos densos llenos de fórmulas y estadísticas. Un
corte ciego, solo por tamaño, rompería las relaciones causales entre
variables experimentales y partiría fórmulas por la mitad. Como los
fragmentos se localizan dentro del texto concatenado por páginas, un chunk
puede extenderse a lo largo de páginas consecutivas (`pages: [4, 5]`) y lleva
asociado un id determinista (`{doc}_p{pages}_c{index}`) para poder referenciarlo.

**Embeddings.** Como modelo de embeddings, se seleccionó BAAI/bge-small-en-v1.5, 
ejecutado localmente mediante sentence-transformers. Se trata de un modelo ligero 
que supera claramente a all-MiniLM-L6-v2 para retrieval en inglés (51.7 vs 41.7 
NDCG@10 en BEIR). Para compensar las limitaciones inherentes a la búsqueda densa 
pura, los resultados semánticos de ChromaDB se combinan con un índice léxico BM25 
(rank_bm25) mediante Reciprocal Rank Fusion (RRF). Así, el modelo denso captura 
similitudes semánticas abstractas o parafraseadas, mientras que BM25 asegura la 
recuperación exacta de entidades léxicas específicas, (tales como acrónimos 
de sensores, fórmulas o nombres de cultivares) que a menudo sufren atenuación
o dispersión en el espacio de embeddings densos. Como variante ligera de contextual 
retrieval, el encabezado de la sección (p. ej., Metodología) se prefija al pasaje 
antes de generar el embedding denso.

### 2. Si el corpus fuera 100 veces más grande

Se consideran las siguientes opciones para escalar la infraestructura:

- **Prefiltrado por metadatos** (año, cultivo, tipo de
  sensor, sección del artículo) antes de cualquier búsqueda vectorial,
  reduciendo así el espacio de búsqueda.
- **Recuperación en dos etapas.** La fusión densa + BM25/RRF actual pasa a ser
  el generador de candidatos (top 20). Un reranker tipo cross-encoder (p.
  ej. `bge-reranker-base`) reordena entonces los candidatos antes de formular
  el prompt. Las colisiones semánticas y los falsos positivos crecen
  con el corpus, y esto evita inyectar ruido en el contexto del LLM.
- **Sustituir ChromaDB embebido por una base de datos vectorial servida y
  distribuida** (Qdrant o Milvus) con índices HNSW y cuantización escalar/por
  producto (SQ/PQ) para mantener RAM y latencia de query bajo control a
  escala.
- **Query decomposition/HyDE** para preguntas complejas o comparativas
  entre varios documentos: descomponer la query en sub-queries atómicas
  antes de recuperar, y hacer que la ingesta se realice en paralelo.


### 3. Cómo medir si el sistema funciona bien

Para evaluar cuantitativamente el sistema, se recomienda emplear un framework 
automatizado de evaluación de RAG (p. ej., RAGAS o TruLens) sobre un conjunto 
de prueba curado con preguntas y fragmentos de referencia (ground truth):

- Calidad de la generación (LLM):
  * Fidelidad (Faithfulness): Porcentaje de afirmaciones de la respuesta que pueden 
  inferirse estrictamente del contexto recuperado. Funciona como detector directo 
  de alucinaciones en datos numéricos y fórmulas científicas.
  * Relevancia (Answer Relevancy): Grado de adecuación de la respuesta generada respecto 
  a la pregunta original del usuario, penalizando redundancias o divagaciones.

- Calidad de la recuperación (Retriever):
  * Exhaustividad del contexto (Context Recall): Proporción de fragmentos relevantes 
  recuperados respecto a todos los necesarios para formular una respuesta completa.
  * Precisión del ranking (Context Precision@k): Ubicación de los fragmentos pertinentes 
  en las primeras posiciones del ranking para optimizar el uso de la ventana de contexto 
  del modelo.

- Rendimiento operativo (Infraestructura y API):
  * Latencia del endpoint /query: Desglose del tiempo de ejecución entre la búsqueda 
  vectorial/léxica y la inferencia del modelo generativo.
  * Métricas de servicio y coste: Tiempo hasta el primer token (Time-to-First-Token o TTFT) 
  y balance de tokens consumidos (entrada vs. salida) para auditar la eficiencia del 
  empaquetado del prompt y el coste por consulta.

## Instalación y ejecución local

Requisitos: Python >= 3.12 (probado en CPython 3.12.14 y 3.14.7, Linux x86_64) y aproximadamente
2 GB de disco para el entorno virtual (torch CPU-only).

1. **Instalar dependencias** (solo dependencias directas, fijadas con versión
   exacta; el índice extra proporciona la wheel de torch CPU-only):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configurar un proveedor de LLM** (opcional, solo necesario para
   `POST /query`):

   ```bash
   cp .env.example .env    # rellena UNA API key, o lanza un servidor Ollama local
   ```

   `.env.example` documenta todas las variables disponibles con sus valores
   por defecto; la tabla de configuración de abajo las resume. Las variables
   dejadas vacías se tratan como no definidas (se aplica el valor por defecto).

3. **Construir el índice vectorial** (one-off, ~70 s en CPU; descarga el
   modelo de embeddings de ~130 MB de Hugging Face en la primera ejecución):

   ```bash
   python scripts/ingest.py
   ```

4. **Arrancar la API** (docs interactivos en http://127.0.0.1:8000/docs):

   ```bash
   uvicorn app.main:app --reload
   curl -s localhost:8000/health
   ```

`POST /retrieve` funciona sin configuración; `POST /query` devuelve un 503
claro hasta que haya un proveedor disponible (también re-sondea perezosamente,
así que lanzar Ollama después funciona sin reiniciar la API).

### Configuración (`.env` / variables de entorno)

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `auto` | Primero disponible de openai → anthropic → deepseek → ollama; un valor explícito falla rápido al arrancar si no es usable |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | – / `gpt-5.6-luna` | Proveedor OpenAI |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | – / `claude-haiku-4-5-20251001` | Proveedor Anthropic |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL` | – / `deepseek-flash` | DeepSeek (REST OpenAI-compatible) |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `llama3.2` | Proveedor local, sin key |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Encoder de sentence-transformers de 384 dims |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `750` / `140` | Split recursivo de fase 2 |
| `EMBED_SECTION_CONTEXT` | `true` | Prefijar el título de sección al pasaje *embebido* (el contenido almacenado permanece limpio) |
| `BGE_QUERY_INSTRUCTION` | `true` | Instruction prefix de BGE para las queries |
| `PDF_DIR` / `CHROMA_DIR` / `COLLECTION_NAME` | `data/raw_pdfs` / `data/chroma` / `scientific_docs` | Layout del almacenamiento |
| `LLM_TIMEOUT_SECONDS` | `60` | Presupuesto de llamada para proveedores cloud (Ollama tiene 300 s) |

## API

### `GET /health`

```json
{"status": "healthy", "vector_store": "connected", "documents_indexed": 5,
 "total_chunks": 555, "active_llm_provider": "none"}
```

`vector_store` es `empty` hasta que se ejecute la ingesta;
`active_llm_provider` es `none` cuando no hay proveedor configurado.

### `POST /retrieve` — auditar retrieval sin tocar ningún LLM

```bash
curl -s -X POST localhost:8000/retrieve -H 'Content-Type: application/json' \
  -d '{"query": "Which vegetation index is best for wheat water status under mildew?", "top_k": 4}'
```

```json
{
  "query": "Which vegetation index is best for wheat water status under mildew?",
  "chunks_retrieved": 4,
  "results": [
    {
      "chunk_id": "fpls-08-01219.pdf_p5_c49",
      "content": "= 0.60–0.64), but less so at flowering and filling stages (R2 = 0.41–0.53 ...",
      "source": "fpls-08-01219.pdf",
      "title": "Canopy Vegetation Indices from In situ Hyperspectral Data to Assess Plant Water Status of Winter Wheat under Powdery Mildew Stress",
      "pages": [5],
      "section": "Results",
      "similarity_score": 0.8381
    }
  ]
}
```

Body: `query` (obligatorio), `top_k` (1–10, default 4), `filter_source`
(opcional, restringe a un nombre de PDF). Los chunks que cruzan un límite de
página reportan ambas páginas (p. ej. `fpls-08-01219.pdf_p2-3_c21`,
`pages: [2, 3]`) — 70 de los 555 chunks indexados lo hacen.

### `POST /query` — ciclo RAG completo con citas

```bash
curl -s -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "Which vegetation index is best for wheat water status under mildew?"}'
```

```json
{
  "question": "Which vegetation index is best for wheat water status under mildew?",
  "answer": "Under powdery mildew stress, the Photochemical Reflectance Index (PRI) is the most reliable indicator, correlating with chlorophyll content and PSII activity [1]. Conventional water indices such as WBI and FWBI show weaker relationships with plant water content across developmental stages [2].",
  "sources": [
    {"citation_id": 1, "chunk_id": "fpls-08-01219.pdf_p9_c70", "document": "fpls-08-01219.pdf",
     "title": "Canopy Vegetation Indices ...", "pages": [9], "section": "Discussion"},
    {"citation_id": 2, "chunk_id": "fpls-08-01219.pdf_p5_c49", "document": "fpls-08-01219.pdf",
     "title": "Canopy Vegetation Indices ...", "pages": [5], "section": "Results"}
  ],
  "llm_used": "gpt-4o-mini",
  "latency_seconds": 1.842
}
```

Body: `question` (obligatorio), `top_k` (1–10, default 4), `filter_source`,
`temperature` (0.0–1.0, default 0.0). Las peticiones son stateless — sin
memoria de chat ni reescritura de queries — y los marcadores de citas que
apuntan fuera de las fuentes proporcionadas se descartan en lugar de generar
referencias colgantes.

Nota: el proveedor OpenAI ignora `temperature` — los modelos razonadores
recientes solo aceptan el valor por defecto y rechazan cualquier otro con un
400.

### Códigos de estado

| Código | Cuándo | Body |
|---|---|---|
| 200 | Health, retrieval o respuesta exitosa | Response JSON |
| 400 | `query` o `question` vacío / solo espacios | `{"detail": "The query field cannot be empty."}` |
| 404 | `filter_source` nombra un archivo nunca indexado | `{"detail": "Unknown source file 'x.pdf'. Available sources: ..."}` |
| 422 | Fallo de validación de Pydantic (`top_k: "many"`, `temperature: 1.5`, ...) | Detalle automático de FastAPI |
| 500 | Fallo del vector store (disco, corrupción) | `{"detail": "Internal error processing vector store."}` |
| 502 | Proveedor LLM configurado falla durante la petición | `{"detail": "LLM provider error: ..."}` |
| 503 | `/query` sin proveedor usable, o queries antes de la ingesta | `{"detail": "No LLM provider available. Set an API key or use the /retrieve endpoint."}` |

## Pruebas

```bash
pip install -r requirements.txt
pip install pytest ruff    # herramientas de desarrollo (no son dependencias de runtime)
pytest                     # 25 tests
ruff check .               # lint (limpio)
```

- `tests/test_chunker.py` — tests unitarios con markdown sintético: metadatos
  de sección, chunks multipágina, límites de tamaño, ids únicos (no requieren
  modelo ni store).
- `tests/test_rrf.py` — matemática de fusión y tokenización.
- `tests/test_settings.py` — parsing de `.env`: valores vacíos caen al default,
  valores explícitos se respetan, defaults coinciden con la documentación.
- `tests/test_api.py` — el stack real vía `TestClient` (modelo de embeddings +
  Chroma + recuperación híbrida): health, ranking, filtrado por fuente, paths
  400/404/422, `/query` end-to-end con un LLM falso (determinista, offline), y
  el path 503 sin LLM. Se saltan automáticamente si la ingesta no se ha
  ejecutado.

## Limitaciones conocidas

- BM25 vive en memoria y se reconstruye cuando cambia el tamaño de la
  colección.
- Sin presupuesto de tokens: el contexto del LLM es simplemente `top_k` chunks
  (<= 10 x 750 caracteres), lo cual manejan cómodamente todos los proveedores
  soportados.
- Un worker de uvicorn. El modelo de embeddings y el cliente de Chroma no se 
  comparten entre procesos.

## Uso de Asistentes de IA

A continuación se desglosa el alcance de las decisiones de diseño e implementación 
propias frente a las tareas aceleradas mediante asistentes de IA:

### 1. Razonamiento, Criterio y Decisiones Propias (Sin Asistencia)
- **Arquitectura del sistema y filosofía Local-First:** Decisión de aislar el pipeline 
  en disco mediante ChromaDB embebido y prescindir de servicios cloud o contenedores 
  pesados para asegurar un arranque inmediato, reproducible e independiente de claves de pago.
- **Estrategia de chunking híbrida:** Concepción del particionado en dos etapas (estructural 
  por cabeceras científicas con `pymupdf4llm` y recursivo acotado a 750 caracteres) 
  para respetar la causalidad de los párrafos técnicos y tablas de espectrometría sin diluir 
  el espacio latente.
- **Diseño del Contextual Retrieval ligero (`embedding_text`):** Identificación de la 
  limitación de invisibilidad de metadatos en el encoder denso y diseño de la técnica 
  de prefijado de sección para habilitar búsquedas temáticas conceptuales sin alterar 
  el texto citado.
- **Trazabilidad y citas indexadas:** Definición de los esquemas deterministas de 
  identificación (`chunk_id`), soporte para fragmentos multi-página (`pages: [X, Y]`) y 
  diseño del mecanismo de citación estricta en el LLM mediante índices correlativos (`[1]`, `[2]`).
- **Selección del modelo de embeddings:** Elección fundamentada de `BAAI/bge-small-en-v1.5` 
  frente a `all-MiniLM-L6-v2` basada en su entrenamiento contrastivo con negativos duros 
  (*hard negatives*) y su rendimiento superior documentado en el benchmark MTEB/BEIR para textos técnicos.
- **Diseño de la API:** Desacoplamiento de los endpoints (`POST /retrieve` para auditoría 
  pura sin LLM vs. `POST /query` para RAG) y lógica de degradación a `503 Service Unavailable`
  en caso de ausencia de modelos generativos configurados.

### 2. Partes Desarrolladas con Asistencia de IA
- **Contratos de datos:** Generación del código base para los modelos de validación Pydantic 
  (`BaseModel`) y tipado estricto en FastAPI.
- **Implementación compacta de algoritmos:** Asistencia en la sintaxis matemática condensada
  para el algoritmo de *Reciprocal Rank Fusion* (RRF) combinando listas ordenadas de BM25 
  y búsqueda densa.
- **Normalización de texto y mapeo de offsets:** Refactorización de las expresiones regulares 
  y funciones auxiliares en Python para calcular la correspondencia exacta de páginas tras 
  el colapsado de espacios en blanco de Markdown.
- **Plantillas de tests unitarios:** Estructuración de los casos de prueba sintéticos con `TestClient` 
  de FastAPI en `pytest`.
- **Estructuración y formato Markdown:** Asistencia en la maquetación visual del README 
  (tablas de endpoints, bloques de código y llamadas de atención) para facilitar la navegación 
  y lectura rápida del documento.
