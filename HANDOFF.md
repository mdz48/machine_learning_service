# HANDOFF — Microservicio ML (NLP de bitácoras + evaluación LLM local)

> **Para el siguiente agente (posiblemente Gemini):** este documento resume TODO lo hecho en la
> sesión previa. El proyecto es un microservicio FastAPI de riesgo de preeclampsia (modelo K-Means/KNN)
> con un módulo NLP que extrae síntomas del texto libre de la paciente ("bitácora"). Léelo completo
> antes de tocar nada. Fecha de cierre: **2026-07-17**.

---

## 0. Estado en una frase

Se implementó y probó una feature de NLP (zonas del cuerpo), se **arregló un bug de producción**
(negación de signos de alarma), y se **evaluó a fondo un LLM local (qwen2.5:3b vía Ollama) contra el
pipeline NER curado**. El LLM gana claramente. **Nada de esto está commiteado ni desplegado todavía.**

---

## 1. Arquitectura del sistema

- **App móvil → API principal (puerto 8000) → este microservicio ML (puerto 8001).** La app NUNCA
  llama al ML directo.
- Este microservicio (`main.py`, FastAPI):
  - `POST /predict` — modelo de riesgo K-Means(K=4)+KNN sobre datos estructurados. Devuelve clúster +
    XAI + (para clúster 1) recomendaciones **SOMANZ** (`recommendations.py`).
  - `POST /nlp/extract-symptoms` — extrae síntomas de texto libre. Pipeline curado en `nlp.py`.
- **VPS de producción:** `mdz@saludprenatal.sytes.net` puerto SSH **5024** (hay alias SSH
  `salud-prenatal`, sin password). Repo en `~/proyectos/machine_learning_service`, docker-compose en
  `~/proyectos/`. 8 contenedores en producción (gateway, auth, pagos, usuarios, ML, etc.).
  Specs: **7.8 GB RAM, 4 cores, sin GPU, 132 GB disco.** Se añadió **8 GB de swap** (persistente).
- **Deploy:** push a `main`/`develop`/`dev` → GitHub Action SSH → `git pull` + `docker compose up -d --build`.

## 2. El pipeline NLP curado (`nlp.py`) — cómo funciona

Texto → **NER** (ONNX `bsc-bio-ehr-es-symptemist`) → filtro stopwords → **negación NegEx** (spaCy) →
**normalización** (embeddings MiniLM ONNX, coseno vs anclas del catálogo de 14 síntomas en
`nlp_catalog.py`) → síntomas. Rama aparte: **zonas del cuerpo** (gazetteer difuso spaCy). Los modelos
ONNX (~255 MB int8) requieren exportarse una vez (`scripts/export_nlp_models.py`, gitignored) y
subirse; sin ellos el endpoint da 503 (degradación segura). Corre CPU-only, sin torch en runtime.

## 3. Lo que se HIZO esta sesión (todo probado, NADA commiteado)

### 3a. Feature: zonas del cuerpo + fix síntomas fantasma
- `nlp_catalog.py`: `BODY_ZONE_CATALOG` (14 zonas) + `EXTRA_STOPWORDS`/`KEEP_WORDS` configurables.
- `nlp.py`: matcher difuso de zonas (tolera typos), vinculación zona↔síntoma por contención
  (`_span_gap`), `_is_content_span` (filtra spans stopword → mató el "Edema fantasma" que salía del
  token "me"), `extract()` devuelve `{symptoms, body_zones}`.
- `main.py`: modelos Pydantic `ExtractedZone`, `zones`, `body_zones`.
- `test/test_nlp_zones.py`: **45 tests, todos verdes** (offline con `spacy.blank("es")`, sin ONNX).
- Docs (en `docs/`, **gitignored**): `nlp-catalogo-complicaciones.md`, `llm-local-extraccion.md`,
  `deployment-nlp.md`, `vps-instalar-llm.md`, y `docs/superpowers/plans/2026-07-11-*-RESULTS.md`.

### 3b. BUG DE PRODUCCIÓN arreglado (importante)
`_is_negated` marcaba `"no puedo respirar bien"` → **Disnea NEGADA** (¡signo de alarma invertido!).
Causa: el NER corta el span en "respirar", el "no puedo" queda antes, NegEx ve la cue "no". Fix:
`_ABILITY_VERBS` (puedo/puede/logro/consigo…) — cue + verbo de capacidad NO es negación. Verificado
en el pipeline real, 45/45 tests. **Este fix protege producción y sigue SIN desplegar.**

### 3c. Evaluación LLM local vs NER curado (el resultado central)
Se levantó **Ollama en el VPS** (contenedor `ollama`, límite 4 GB RAM, puerto interno 11434,
`qwen2.5:3b` ~1.9 GB). Se creó un dataset de **45 bitácoras etiquetadas** (`evaluation/eval_data.py`)
y se comparó el LLM (VPS) vs el NER (local). Se iteró el prompt del LLM 3 veces:

| Métrica (45 bitácoras) | NER curado | LLM v1 | LLM v2 | **LLM v3 (final)** |
|---|---|---|---|---|
| **Alarmas detectadas** (recall) | 15/24 | 23/24 | 22/24 | **24/24** |
| Alarmas perdidas [PELIGRO] | 9 | 0 | 2 | **0** |
| **Alucinaciones** (síntomas inventados) | 15 | 8 | 4 | **4** |
| Negaciones reales bien | 7/7 | 4/7 | 3/7 | **5/7** |
| Fuera-de-catálogo bien | 2/7 | 3/7 | 5/7 | **5/7** |
| **Bitácoras EXACTAS** | 23/45 | 34/45 | 34/45 | **39/45 (87%)** |
| Latencia (VPS 4-cores) | 25 ms | ~12 s | ~12 s | **mediana ~13 s** |

**Conclusión: el LLM v3 gana claro** — 24/24 alarmas vs 15/24 del NER (el NER se come alarmas por
typos y fraseos que no reconoce). El módulo v3 quedó congelado en **`llm_extractor.py`** (raíz del repo).

**Claves técnicas descubiertas (por qué v3 funciona):**
1. **Salida estructurada por JSON Schema** (no `format:"json"` a secas) → JSON válido 100%.
2. **Anclaje difuso** (`_grounded`): `raw_text` debe estar (≥60% de sus palabras) en el texto de la
   paciente. Mata la alucinación dominante: el 3B **copia pedazos de su propio prompt** como síntomas.
   Difuso (no substring exacto) para tolerar typos del propio modelo en `raw_text`.
3. **Prompt rico** (descripciones de código con ejemplos) para clasificar bien + reglas explícitas
   anti-hipotético ("la doctora dijo si sangro…" → []) y multi-negación.
4. **NO requiere fine-tuning** — todo es prompt + few-shot + schema (in-context).

## 4. Decisiones tomadas / DESCARTADAS con datos

- **Híbrido LLM+NegEx: DESCARTADO.** Hipótesis (NegEx confiable para negación) refutada: NegEx no
  distingue "negar un síntoma" de "síntoma que se nombra negando" ("no puedo respirar" ES el síntoma).
  Con buen prompt, el LLM solo maneja mejor la negación. No lo reimplementes.
- **SNOMED CT / entity-linking: DESCARTADO** (research-grade ~0.61 accuracy ES, el usuario no lo quiere).
- **Resumen para el médico:** usar **plantilla determinista**, NO LLM (el LLM alucinó fechas/omitió
  datos). Regla general: LLM donde no hay alternativa determinista (extracción), plantilla donde sí.
- **Modelo 7B: probablemente NO** (en 4-cores ~2× más lento y RAM justa). El 3B v3 es suficiente.
- **Flujo:** el usuario confirmó **asíncrono** (bitácora diaria, el médico la ve días/semanas después)
  → la latencia de ~13 s **no importa**. Y **NO hay revisión médica** (va directo al médico).

## 5. PENDIENTES (próximos pasos, en orden sugerido)

1. **Validar v3 en un set HELD-OUT nuevo (20-30 bitácoras que el prompt NO haya visto).** El 87% actual
   tiene riesgo de sobreajuste porque el prompt se afinó sobre esas mismas 45. Este es el número honesto
   para el informe/docente. (Correr: ver §6.)
2. **Reducir las 4 alucinaciones restantes** (~9%). Como va directo al médico sin revisión, la salida
   debería marcarse como "auto-extraído, confirmar". Ideas: pasada de verificación, o más few-shot.
3. **Decidir arquitectura:** ¿LLM-only o LLM + pipeline curado como fallback? (`llm_extractor.py` ya
   tiene `use_fallback`.) Si LLM-only, se retiran NER/embeddings/zonas/NegEx (pero el **catálogo**
   `nlp_catalog.py` SOBREVIVE — es el vocabulario controlado + flags `alarm` que el LLM necesita).
4. **Wire `llm_extractor.py` a un endpoint** en `main.py` (ej. `/nlp/extract-symptoms-llm`) y medir e2e.
5. **COMMIT + DEPLOY.** Nada está commiteado. El bug de negación (§3b) sigue activo en producción hasta
   que se despliegue. Ver `git status`. Rama actual: `somanz`. Deploy requiere merge a `main`.

## 6. Cómo reproducir la evaluación (para el siguiente agente)

Todo en `evaluation/` (ya en el repo):
- `eval_data.py` — 45 bitácoras + gold standard (etiquetado a mano = proxy; un clínico debe validarlo).
- `eval_ner.py` — corre el NER local → `ner_results.json`. (Necesita los modelos ONNX en `models/nlp/`.)
- `eval_llm_v3.py` — corre el LLM v3 (escribe `/tmp/llm3.jsonl`). Diseñado para correr EN EL VPS.
- `eval_score.py` — compara NER vs LLM (lee `ner_results.json` + `llm_results.json`).

**Correr el LLM en el VPS** (la red es inestable → usar procesos que auto-escriben + polls cortos):
```bash
scp -P 5024 evaluation/eval_data.py evaluation/eval_llm_v3.py mdz@saludprenatal.sytes.net:/tmp/
ssh salud-prenatal "cd /tmp && nohup python3 eval_llm_v3.py >/dev/null 2>&1 & disown"
# poll: ssh salud-prenatal "wc -l < /tmp/llm3.jsonl; [ -f /tmp/llm3.done ] && echo DONE"
scp salud-prenatal:/tmp/llm3.jsonl evaluation/
```
Para scoring: convertir el `.jsonl` a un `llm_results.json` (array) y correr `eval_score.py`.

**Estado del VPS ahora mismo:** contenedor `ollama` corriendo (`docker ps`), `qwen2.5:3b` descargado
en volumen `ollama_models`, swap de 8 GB activo. El modelo se descarga de RAM tras 5 min ocioso
(primera llamada tras pausa ~31 s). Considerar `OLLAMA_KEEP_ALIVE=-1` si se quiere siempre caliente.

## 7. Advertencias / trampas conocidas

- **La red al VPS se corta en conexiones SSH largas.** Usar procesos `nohup`/self-writing + polls cortos.
- **El gold standard lo etiquetó el agente anterior, no un clínico.** Validación clínica pendiente.
- **`docs/` y `scripts/` están en `.gitignore`** → los docs de esta sesión y el export script NO se
  versionan por git. `evaluation/` y `llm_extractor.py` y `HANDOFF.md` SÍ (están en la raíz/no ignorados).
- **Sin revisión médica + directo al médico:** las alucinaciones del LLM llegan como si fueran reales.
  Recomendación fuerte: etiquetar la salida como sugerencia auto-extraída a confirmar.
- **Ollama en el VPS comparte CPU con auth y pagos en producción.** El límite de 4 GB protege la RAM,
  pero bajo carga la latencia sube. Vigilar `docker stats` / `uptime`.

## 8. Archivos clave

| Archivo | Qué es | ¿En git? |
|---|---|---|
| `llm_extractor.py` | **Módulo LLM v3 congelado** (prompt + anclaje + validación + fallback) | sí |
| `nlp.py` | Pipeline NER curado (con fix de negación) | sí (untracked) |
| `nlp_catalog.py` | Catálogo 14 síntomas + zonas + stopwords config | sí (untracked) |
| `main.py` | FastAPI (endpoints) | sí (modificado) |
| `test/test_nlp_zones.py` | 45 tests | sí (untracked) |
| `evaluation/` | Dataset + scripts + resultados de la comparación | sí |
| `HANDOFF.md` | Este documento | sí |
| `docs/*.md` | Specs, planes, docs de integración, despliegue LLM | **NO (gitignored)** |
