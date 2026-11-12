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
  ocr_store.py              archivos de OCR por cliente: texto propio o tomado de la otra marca
                            cuando la imagen es idéntica
  fresh_ocr.py              OCR nuevo: Rekognition sobre la foto actual, en otra cuenta de AWS
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
  14_fresh_ocr.py           fase 1 · vuelve a pasar Rekognition por las fotos de todas las personas
  15_enlarged_ocr_test.py   fase 1 · prueba: ¿se lee mejor el reverso si se envía la cédula ampliada?
  16_check_new_positives.py fase 1 · cuenta los positivos confirmados fuera de fraud_status (solo lectura)
  17_data_summary.py        fase 1 · cifras del dataset para el informe (solo archivos locales)
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
`DB_HOST`, `DB_PORT`, `DB_NAME`. Para el OCR nuevo, además, `REKOGNITION_PROFILE` y
`REKOGNITION_ACCOUNT_ID`, que son obligatorios y no tienen valor por defecto (ver "Cuenta aparte
para Rekognition").

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

### Cuenta aparte para Rekognition

El OCR nuevo (`14_fresh_ocr.py` y la prueba `15_enlarged_ocr_test.py`) es lo único que genera
consumo, y no va a la cuenta corporativa:
se paga con otra cuenta de AWS. Las dos cuentas conviven porque cada una usa su propio perfil: la
base y S3 siguen con la sesión de SSO; Rekognition usa un perfil con nombre.

1. En la cuenta que paga, crear un usuario IAM solo para esto, con una única política que permita
   `rekognition:DetectText`, y generarle una llave de acceso.
2. Guardar la llave en un perfil: `aws configure --profile id_fraud_rekognition` (región
   `us-east-1`). Queda en `~/.aws/credentials`, fuera del repositorio.
3. En `.env`, nombrar la cuenta que paga con dos valores: el nombre del perfil
   (`REKOGNITION_PROFILE=id_fraud_rekognition`) y el número de 12 dígitos de esa cuenta
   (`REKOGNITION_ACCOUNT_ID=`; no es un secreto, aparece arriba a la derecha en la consola de
   AWS). Las llaves no van en `.env`.

Nada se asume. Antes de leer nada o de enviar una sola foto, el script termina con error si:

- falta `REKOGNITION_PROFILE` o `REKOGNITION_ACCOUNT_ID`, o el perfil no existe;
- las llaves del perfil son de una cuenta distinta de la escrita en `REKOGNITION_ACCOUNT_ID`;
- esa cuenta es la misma de la sesión de SSO.

Con el perfil dado de forma explícita, `boto3` tampoco usa las variables de entorno
`AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY` ni el perfil por defecto. Todo servicio con costo
se crea en un solo lugar, `common.rekognition_client()`.

Las fotos se leen de S3 y se envían a Rekognition dentro de la llamada; en la otra cuenta no
queda ninguna copia. Como son datos personales procesados fuera de la cuenta de la empresa, requiere su
autorización, y conviene desactivar en esa cuenta el uso de contenido para mejorar los servicios
de IA de AWS (política de exclusión de servicios de IA, en AWS Organizations).

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
| 2 | `python scripts/11_export_dataset.py --no-images` | Base (lectura) y lecturas de un byte en S3 | ~2 min (medido) |
| 3 | `python scripts/14_fresh_ocr.py` | Nada: solo informa lo pendiente y el costo | < 1 min |
| 4 | `python scripts/14_fresh_ocr.py --run` | S3 (lectura) y Rekognition en la otra cuenta | 1–3 h; ~56 USD por ~56.400 fotos |
| 5 | `python scripts/12_ocr_coverage.py` | Solo archivos locales | 1–2 min |
| 6 | `python scripts/11_export_dataset.py` | Base (lectura) y S3 | depende de la red; ~8.800 fotos |
| 7 | `streamlit run scripts/13_visual_review.py` | Solo archivos locales | manual, ~381 casos |
| 8 | `python scripts/20_crop_test.py` | Solo archivos locales | < 1 min |

Salvo el paso 2, los tiempos son estimados.

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
7. En todos los positivos y en una muestra de 300 negativos, cuántos tienen texto de Rekognition
   guardado, desglosado por origen de la etiqueta, por canal y por dónde se validó la persona la
   primera vez. Si no encuentra ninguno, imprime las claves reales del JSON.
8. Clientes etiquetados cuya persona ya estaba validada en alguna marca antes de registrarse en
   esta. A esos clientes `mikro_back` no les pide fotos de nuevo: copia las que ya tenía y no
   pasa por Rekognition, así que sus fotos son anteriores al registro y no tienen OCR aquí.
9. Clientes etiquetados que también existen en la otra marca (la misma persona podría estar en
   entrenamiento y en la prueba externa).
10. Clientes etiquetados cuyo registro común (`public.customers_commonclient`) tiene marca de
    fraude.
11. Lee un byte de una foto en S3 para comprobar las credenciales (`--no-s3` lo omite).
12. Une los etiquetados de todas las marcas por número de documento, con la misma regla de la
    exportación, y cuenta personas: cuántas quedan por etiqueta, cuántas filas repetidas se
    quitan, si alguna es positiva en una marca y negativa en otra, y de qué marca queda la fila.

### `11_export_dataset.py` — construcción del dataset
1. Consulta los clientes de cada tenant, aplica las reglas de etiquetado y se queda con
   positivos y negativos que tengan ambas fotos.
2. Deja **una fila por persona**: si el mismo número de documento está etiquetado en varias
   marcas, conserva una sola fila para no repetir la cédula. Queda, en este orden: la positiva
   antes que la negativa, la de registro más antiguo (la marca donde probablemente se tomaron las
   fotos), y el orden de las marcas en `config.yaml`. Las fotos son las de la marca de la fila
   que queda; la columna `labeled_in` dice en qué marcas está etiquetada la persona. Luego
   calcula `anon_id` y `person_id` con `ANON_SALT`.
3. Elige las fotos a descargar: todos los positivos y una muestra de negativos estratificada por
   mes y canal (2.500 en ESMIO, 1.500 en Llévalo Ya; `config.yaml`).
4. Trae el OCR guardado de **todas** las personas, en lotes de 300 clientes. La base devuelve
   solo las líneas de texto con su confianza, no el JSON completo de Rekognition. Lo guarda en
   `data/ocr/<anon_id>.json`.
5. Para los clientes que quedaron sin texto, busca a la misma persona (mismo número de documento)
   en la otra marca. Si allá sí hay texto **y la imagen de esa cara es idéntica en las dos
   marcas**, usa ese texto. Pasa cuando la persona se validó primero en la otra marca: aquí solo
   se copiaron las fotos y Rekognition no corrió. La igualdad se comprueba leyendo un byte de
   cada objeto en S3 (huella y tamaño), sin descargar nada; si la foto se volvió a tomar, no se
   usa. Se hace por cara: puede tomarse el frente y no el reverso. El manifest registra el origen
   en `front_ocr_from` y `back_ocr_from`. `--no-borrow-ocr` lo desactiva.
6. Sin `--no-images`, descarga frente y reverso de los elegidos (8 descargas en paralelo) a
   `data/images/<tenant>/<label>/` y registra ancho, alto, formato, tamaño y MD5 de cada archivo.
7. Escribe `data/manifest.csv` (una fila por persona, sin datos personales) y
   `data/private/id_map.csv` (anon_id → número de documento y rutas de las fotos en S3; solo
   local).

Se puede repetir sin costo: lo que ya está en disco no se vuelve a pedir, y cada cliente se busca
una sola vez en la otra marca. `--refresh-ocr` fuerza a traer el OCR otra vez. Siempre procesa
todas las marcas configuradas, porque el cruce por persona las necesita todas.

### `14_fresh_ocr.py` — OCR nuevo sobre las fotos actuales
El texto guardado en producción cubre a una minoría de los clientes. Este script vuelve a pasar
`detect_text` de Rekognition por el frente y el reverso de **todas** las personas del manifest,
tengan o no texto guardado, para que el texto corresponda con seguridad a la foto actual.

1. Lee el manifest y las rutas de las fotos (`data/private/id_map.csv`). No toca la base.
2. Cuenta lo pendiente y estima el costo (0,001 USD por foto).
3. Comprueba las dos cuentas: Rekognition debe estar en una cuenta distinta de la de la base y
   S3; si es la misma, termina sin enviar nada.
4. **Sin `--run` termina ahí**: solo informa. Con `--run`, por cada persona toma las dos fotos
   (del disco si ya están; si no, de S3 con la sesión de SSO), las envía a Rekognition (perfil
   `REKOGNITION_PROFILE`) y guarda en `data/ocr_fresh/<anon_id>.json` todo lo que Rekognition detectó,
   tal como llegó (líneas y palabras, con su confianza y su posición), más el MD5 y el tamaño de
   la imagen que produjo ese texto. Una imagen de más de 5 MB (límite de Rekognition) se envía
   recodificada como JPEG.
5. Imprime cuántas fotos se leyeron, el costo aproximado, el tamaño promedio de las fotos (y
   cuánto ocuparían todas en disco), los errores por tipo y, por marca y etiqueta, cuántas
   personas tienen texto en cada cara.

Las fotos pasan de a pocas (8 personas a la vez), nunca todas juntas. Por defecto no se guardan
en disco. Con `--keep-images` cada foto descargada queda en `data/images/<tenant>/<label>/`, el
mismo lugar que usa `11_export_dataset.py`: así una foto se descarga una sola vez y ni repetir el
OCR ni la descarga para el clasificador la vuelven a pedir.

Los positivos van primero. `--limit 20` procesa solo 20 personas, para una prueba. Se puede
interrumpir y repetir: el texto de cada persona se guarda apenas llega, una cara ya respondida no
se vuelve a enviar ni a cobrar, y una que falló se reintenta. Si 25 personas seguidas fallan
(sesión vencida, permisos), se detiene solo.

`--negatives 300` hace una corrida parcial: todos los positivos y solo 300 negativos. Esos
negativos se toman al azar, con semilla fija (siempre los mismos), entre los que el manifest ya
marca para imágenes (`has_images`: la muestra estratificada por mes y canal de `config.yaml`).
Sirve para medir cuánto se lee antes de pagar la lista completa.

### `12_ocr_coverage.py` — legibilidad y validación de las reglas
No se conecta a nada. Lee el manifest y los archivos de OCR, extrae los campos del reverso
(código, serie, fechas, registrador, sexo) y el número del frente, ejecuta las seis reglas del
motor más la de reverso repetido entre clientes, e imprime:

- qué porcentaje de clientes tiene cada campo legible, por tenant y etiqueta, y por etiqueta y
  canal;
- por regla y etiqueta: cobertura, número de `FAIL` y tasa. En negativos esa tasa son los falsos
  positivos; en positivos, la detección.

Por defecto usa el OCR nuevo (`data/ocr_fresh/`) y mide solo a las personas que Rekognition ya
leyó, así que sirve también después de una corrida parcial. Con `--ocr stored` usa el texto guardado en
producción (`data/ocr/`) y los archivos de salida terminan en `_stored`, para poder comparar.

Guarda `data/rules_summary.csv`, `data/rules_per_customer.csv` (estados, sin valores) y
`data/private/rules_values.csv` (los valores leídos: números y series; solo local).

### `15_enlarged_ocr_test.py` — prueba: cédula ampliada contra foto completa
Responde una sola pregunta antes de pagar la corrida grande: ¿Rekognition lee mejor el reverso si
recibe solo la cédula, ampliada? Toma 200 personas al azar (siempre las mismas) entre las que ya
tienen la foto del reverso en disco y una primera lectura, recorta la zona donde Rekognition
encontró texto, la amplía al doble y la envía de nuevo. No lee nada de S3. Guarda la respuesta en
`data/ocr_enlarged_test/` e imprime, por canal y por etiqueta, cuántos códigos se interpretan con
la foto completa y con la cédula ampliada, cuántos se ganan y cuántos se pierden. Sin `--run`
solo informa. Usa la misma comprobación de cuentas que `14_fresh_ocr.py`. Cuesta unos 0,20 USD.

### `17_data_summary.py` — cifras del dataset
No se conecta a nada. Lee el manifest, los archivos de OCR, las fotos en disco y los resultados
de los demás scripts, y escribe una tabla pequeña por tema en `data/summary/*.csv`: universo de
clientes, personas por etiqueta y origen, canal, mes, texto guardado, fotos y sus tamaños,
lectura nueva y su costo, legibilidad, reglas, detección por origen y por canal, verificador y
prueba de recorte. Solo conteos y tasas. Con esas tablas se mantiene `docs/data_summary.md` y se
hacen las gráficas del informe.

### `16_check_new_positives.py` — positivos confirmados fuera de `fraud_status` (solo lectura)
No cambia nada: solo cuenta lo que aporta cada fuente. El etiquetado lo aplica
`11_export_dataset.py`, que toma como positivos a los clientes de `confirmed_fraud_sources` y a
las cédulas que un analista confirmó como falsas en el verificador, siempre sin crédito pagado.
Revisa dos cosas:

- Las fuentes de `confirmed_fraud_sources` en `config.yaml`: clientes con crédito en un comercio
  dentro de una ventana de fechas, o con crédito desembolsado a un banco en ciertos comercios.
- El historial del verificador de cédulas del equipo, si está guardado en
  `data/private/verifier_history/casos/` (un JSON por caso): las cédulas marcadas como falsas o
  auténticas, separadas por quién puso el veredicto (un analista o la herramienta sola).

Por cada grupo imprime cuántos clientes hay, cuántos tienen ambas fotos, cuándo se registraron,
su `fraud_status`, si tienen crédito pagado, qué etiqueta tienen hoy y cuántos serían positivos
nuevos. Solo imprime conteos.

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
