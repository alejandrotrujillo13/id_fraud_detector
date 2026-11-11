# Detector de fraude en cédulas de ciudadanía

Proyecto de grado · Especialización en Inteligencia Artificial · Alejandro Trujillo López

Prototipo que analiza las fotos del frente y el reverso de una cédula colombiana y entrega un
indicador de riesgo de falsificación (0–100), con dos componentes:

1. **Motor de reglas estructurales** sobre el texto del reverso (OCR de Rekognition ya almacenado):
   registrador vs. fecha de producción, coherencia del código, reverso repetido entre personas.
2. **Clasificador de imágenes** por transferencia de aprendizaje (frente y reverso) para lo visual.

El plan completo está en [docs/plan.md](docs/plan.md) y la historia de las decisiones en
[docs/context.md](docs/context.md). Este repositorio es independiente de `mikro_back`: lee la base
de datos en modo solo lectura y descarga imágenes de S3; no escribe nada en producción.

> El código, los identificadores y los comentarios están en inglés; la documentación en español.

## Estructura

```
pyproject.toml              nombre del paquete y dependencias (base + grupos opcionales)
config.yaml                 reglas de etiquetado, tenants por entorno, muestreo de negativos
.env.example                STAGE y ANON_SALT (copiar a .env; nunca se versiona)
docs/
  plan.md                   plan técnico vigente (fuente de verdad)
  context.md                historia de las decisiones
src/id_fraud_detector/      paquete instalable con toda la lógica reutilizable
  common.py                 configuración, anonimización (ANON_SALT), conexión de solo lectura
                            (SSO + Secrets Manager, base sisteya_<STAGE>)
  dataset.py                consultas SQL (solo SELECT), etiquetado y muestreo; lo usan 10 y 11
  crop.py                   localización y rectificación del documento con OpenCV
  rules/
    registrars.py           registradores nacionales por periodo (desde 1999) y zona gris 2023
    ocr_parse.py            extracción de campos del OCR (código del reverso, serie, fechas, número)
    engine.py               reglas con tres estados: PASS / FAIL / NOT_EVALUABLE
scripts/                    puntos de entrada; el primer dígito es la fase
  10_check_database.py      fase 1 · verificaciones previas: conteos, OCR, cruces entre marcas, S3
  11_export_dataset.py      fase 1 · construye manifest.csv, ocr/*.json y descarga imágenes
  12_ocr_coverage.py        fase 1 · legibilidad por campo y tasas de fallo por regla
  13_visual_review.py       fase 1 · app Streamlit para clasificar los positivos por modalidad
  20_crop_test.py           fase 2 · hoja de contacto para juzgar el recorte en 50 fotos por tenant
tests/                      pruebas con datos sintéticos (reglas, etiquetado, conexión)
data/<STAGE>/               (git-ignored) dataset, OCR, imágenes, resultados; private/ tiene datos reales
```

La estructura prevista para las fases siguientes está en [docs/plan.md](docs/plan.md) §6b. Las
carpetas se crean cuando empieza su fase, no antes.

## Reglas de etiquetado (config.yaml)

- **POSITIVE**: `fraud_status = FRAUD` sin ningún crédito `PAID`; o `POSSIBLE FRAUD` dentro de una
  ventana explícita (ESMIO, 2026-09-13 a 2026-10-04: red de la tienda 623) sin crédito `PAID`.
- **NEGATIVE**: sin `fraud_status` y con al menos un crédito `PAID`.
- Fuera del dataset: los demás estados y los clientes sin créditos pagados ni marca.
- Tenants `esmio` y `llevaloya`, registrados desde 2025-01-01, con frente y reverso guardados.

## Puesta en marcha (una sola vez)

```powershell
.venv\Scripts\activate
pip install -e ".[review]"
copy .env.example .env
```

`pip install -e .` instala el paquete `id_fraud_detector` en modo editable: los scripts lo
importan por su nombre y cualquier cambio en `src/` aplica de inmediato, sin reinstalar. El grupo
`review` agrega Streamlit para la revisión visual.

En `.env` van solo dos valores obligatorios; no lleva llaves de AWS ni contraseña de la base:

- `STAGE`: el entorno al que se apunta (`dev`, `test` o `live`). La base es `sisteya_<STAGE>` y
  todo lo que los scripts escriben queda en `data/<STAGE>/`, así que los datos de un entorno
  nunca se mezclan con los de otro. Con `STAGE=dev` solo se usa el tenant `demo`; en los demás
  entornos, `esmio` y `llevaloya` (`tenants_by_stage` en `config.yaml`).
- `ANON_SALT`: se genera una sola vez con
  `python -c "import secrets; print(secrets.token_hex(32))"` y no se cambia nunca: de él dependen
  los identificadores anónimos de todo el dataset.

Opcionales (en blanco usan el valor por defecto): `AWS_PROFILE`, `AWS_REGION`, `DB_SECRET_NAME`,
`DB_HOST`, `DB_PORT`, `DB_NAME`.

### Conexión

Antes de correr cualquier script que toque la base o S3, inicia la sesión de SSO:

```powershell
aws sso login
```

Con esa sesión los scripts leen de Secrets Manager el secreto `sisteya_db_proxy_credentials` (el
mismo que usa `mikro_back`: host, puerto, usuario y contraseña) y se conectan a `sisteya_<STAGE>`.
S3 usa la misma sesión. Si la sesión venció, el script termina con un mensaje que pide repetir
`aws sso login`.

La sesión de base de datos se abre en modo solo lectura y se verifica antes de ejecutar la primera
consulta; los scripts solo hacen `SELECT`. El usuario del secreto es el de la aplicación, que sí
tiene permisos de escritura: la protección es la sesión de solo lectura, no el usuario.

Para comprobar que el código está sano sin tocar la base: `python -m unittest discover tests`.

## Paso a paso de las fases 1 y 2

Todo se ejecuta desde la raíz del repositorio con el entorno activado y la sesión de SSO
iniciada. Ningún paso requiere correr consultas a mano: los scripts ejecutan las suyas.

Conviene hacer la primera corrida del paso 1 con `STAGE=dev`: valida las consultas sin tocar
producción. Después se cambia a `STAGE=live` y se repite desde el paso 1.

Las rutas `data/...` que aparecen abajo son relativas a la carpeta del entorno: con `STAGE=live`,
`data/manifest.csv` es `data/live/manifest.csv`.

| # | Comando | Toca | Tarda (aprox.) |
|---|---|---|---|
| 1 | `python scripts/10_check_database.py` | Base (lectura) y un objeto de S3 | 1–3 min |
| 2 | `python scripts/11_export_dataset.py --no-images` | Base (lectura) | 10–30 min |
| 3 | `python scripts/12_ocr_coverage.py` | Solo archivos locales | 1–2 min |
| 4 | `python scripts/11_export_dataset.py` | Base (lectura) y S3 | depende de la red; ~8.800 fotos |
| 5 | `streamlit run scripts/13_visual_review.py` | Solo archivos locales | manual, ~399 casos |
| 6 | `python scripts/20_crop_test.py` | Solo archivos locales | < 1 min |

Los tiempos son estimados; ninguno se ha medido todavía contra la base real.

## Qué hace cada script

### `10_check_database.py` — verificaciones previas
No escribe datos personales; solo imprime tablas y guarda los conteos en `data/checks/*.csv`.
Por cada tenant:

1. Muestra la base y el esquema en uso, confirma que la sesión es de solo lectura y muestra la
   zona horaria con la que la base evalúa las fechas.
2. Cuenta los clientes registrados desde 2025-01-01 por grupo (positivo, negativo, excluidos, sin
   uso). Con `STAGE=live` los pone al lado de los conteos del 9-nov-2026
   ([docs/context.md](docs/context.md) §4.4). Diferencias pequeñas son normales (clientes nuevos,
   créditos que se pagan); una grande no.
3. Muestra cuántos positivos y negativos produciría la exportación, y cuántos `POSSIBLE FRAUD`
   caen en la ventana configurada (referencia en `live`: 78).
4. Positivos y negativos por mes de registro: es el insumo para decidir la partición
   entrenamiento/validación.
5. Positivos y negativos por canal (app, comercio, …).
6. Cuántos clientes etiquetados tienen un crédito `NOVATED`, que hoy no cuenta como pagado.
7. En una muestra de 300 clientes por etiqueta, cuántos tienen texto de Rekognition guardado en
   el lugar donde el parser lo busca. Si no encuentra ninguno, imprime las claves reales del JSON.
8. Clientes etiquetados que también existen en la otra marca (`mikro_back` copia las fotos entre
   marcas, así que la misma foto podría estar en entrenamiento y en la prueba externa).
9. Clientes etiquetados cuyo registro común (`public.customers_commonclient`) tiene marca de fraude.
10. Lee un byte de una foto en S3 para comprobar las credenciales (`--no-s3` lo omite).

### `11_export_dataset.py` — construcción del dataset
1. Consulta los clientes de cada tenant, aplica las reglas de etiquetado y se queda con
   positivos y negativos que tengan ambas fotos.
2. Calcula `anon_id` (por tenant) y `person_id` (igual en todas las marcas) con `ANON_SALT`.
3. Elige las fotos a descargar: todos los positivos y una muestra de negativos estratificada por
   mes y canal (2.500 en ESMIO, 1.500 en Llévalo Ya; `config.yaml`).
4. Trae el OCR de **todos** los etiquetados, en lotes de 300 clientes. La base devuelve solo las
   líneas de texto con su confianza, no el JSON completo de Rekognition. Lo guarda en
   `data/ocr/<anon_id>.json`.
5. Sin `--no-images`, descarga frente y reverso de los elegidos (8 descargas en paralelo) a
   `data/images/<tenant>/<label>/` y registra ancho, alto, formato, tamaño y MD5 de cada archivo.
6. Escribe `data/manifest.csv` (una fila por cliente, sin datos personales) y
   `data/private/id_map.csv` (anon_id → número de documento; solo local).

Se puede repetir sin costo: lo que ya está en disco no se vuelve a pedir. `--refresh-ocr` fuerza a
traer el OCR otra vez; `--tenant esmio` procesa una sola marca y conserva las filas de la otra.

### `12_ocr_coverage.py` — legibilidad y validación de las reglas
No se conecta a nada. Lee el manifest y los archivos de OCR, extrae los campos del reverso
(código, serie, fechas, registrador, sexo) y el número del frente, ejecuta las seis reglas del
motor más la de reverso repetido entre clientes, e imprime:

- qué porcentaje de clientes tiene cada campo legible, por tenant y etiqueta;
- por regla y etiqueta: cobertura, número de `FAIL` y tasa. En negativos esa tasa son los falsos
  positivos; en positivos, la detección.

Guarda `data/rules_summary.csv`, `data/rules_per_customer.csv` (estados, sin valores) y
`data/private/rules_values.csv` (los valores leídos: números y series; solo local).

### `13_visual_review.py` — revisión visual de positivos
App local de Streamlit. Muestra frente y reverso de cada positivo y guarda la modalidad elegida en
`data/review/modalities.csv`. Retoma donde quedó.

### `20_crop_test.py` — prueba del recorte (fase 2)
No se conecta a nada. Toma 25 positivos y 25 negativos por tenant de las fotos descargadas, corre
el recorte de `src/id_fraud_detector/crop.py` y genera una hoja de contacto por tenant y cara en
`data/crop_test/*_sheet.jpg`: a la izquierda la foto con el cuadrilátero detectado, a la derecha la
cédula enderezada. El porcentaje que imprime solo dice si encontró un cuadrilátero; la calidad se
juzga mirando las hojas. Regla del plan: si recorta bien menos del 90 %, se entrena YOLO.

## Principios

- **Las reglas se abstienen, no penalizan**: cada regla devuelve PASS, FAIL o NOT_EVALUABLE.
  Solo FAIL suma riesgo. Una lectura rota del OCR nunca se interpreta como dato mal formado.
- **Los pesos salen de los datos**: la tasa de falsos positivos de cada regla se mide sobre todos
  los negativos y su detección sobre los positivos antes de usarla.
- **Privacidad**: identificadores anonimizados con sal; `data/` nunca se versiona; las imágenes
  no se suben a servicios públicos (Ley 1581 de 2012).
