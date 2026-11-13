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
  30_make_splits.py         fase 3 · reparte a las personas con fotos en cinco partes
  31_train.py               fase 3 · entrena el clasificador de una cara con validación cruzada
  32_audit_run.py           fase 3 · revisa si el modelo sigue la falsificación o la cámara y la fecha
  33_combine_sides.py       fase 3 · un puntaje por persona con frente y reverso, más las reglas
  34_plot_history.py        fase 3 · curvas de pérdida y exactitud de una corrida
  35_grad_cam.py            fase 3 · mapas de calor: dónde mira la red
  36_face_repeats.py        fase 3 · caras repetidas entre cédulas y su efecto en la detección
  50_calibrate.py           fase 5 · calibra el puntaje, pesa las reglas y arma el indicador
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
| 9 | `python scripts/30_make_splits.py` | Solo archivos locales | < 1 min |
| 10 | `python scripts/31_train.py --side back` (y `--side front`) | Solo archivos locales y la GPU | ~20 min por cara |

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
3. Elige las fotos a descargar: todos los positivos y una parte de los negativos, definida en
   `config.yaml`:
   - una muestra estratificada por mes y canal (2.500 en ESMIO, 1.500 en Llévalo Ya);
   - todos los negativos de app y todos los registrados desde septiembre de 2026, para que los
     negativos se parezcan a los positivos (`negatives_always_with_images`);
   - todos los que ya tienen sus fotos en disco. Por eso la muestra no se mueve al volver a
     exportar: solo crece.
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

### `30_make_splits.py` — las cinco partes de la validación cruzada (fase 3)
No se conecta a nada. Toma a las personas con ambas fotos en disco y las reparte en cinco partes,
que guarda en `data/splits.csv`. Cada persona queda en una sola parte; una red completa (tienda
623, AMORETT) y quienes comparten reverso quedan juntos, para que el modelo que los evalúa nunca
haya visto esa plantilla. Las partes quedan parejas en etiqueta y canal. Se corre una sola vez;
`--force` la rehace si cambia el dataset.

### `31_train.py` — entrenamiento del clasificador (fase 3)
No se conecta a nada; entrena en este equipo con PyTorch (usa la tarjeta gráfica si la hay).
Requiere `pip install -e ".[review,train]"` con PyTorch instalado desde su índice con soporte de
GPU (ver `pyproject.toml`).

```powershell
python scripts/31_train.py --side back
python scripts/31_train.py --side front
```

Por cada parte entrena un modelo con tres partes, lo vigila con una cuarta (parada temprana) y
puntúa la quinta, que nunca vio. Al final cada persona tiene un puntaje dado por un modelo que no
entrenó con ella ni con su red, y sobre esos puntajes se calculan las métricas: PR-AUC, ROC-AUC y,
la principal, cuántas falsas se detectan dejando pasar a revisión solo al 2 % de las legítimas.
Se reportan en total, por parte, por canal y por origen del positivo.

- **Modelo:** MobileNetV3 preentrenada en ImageNet (`--arch efficientnet_b0` para la otra), con
  una cabeza nueva de una salida. Primero se entrena solo la cabeza; luego se liberan los
  últimos bloques con tasa de aprendizaje baja.
- **Entrada:** la foto completa, sin recortar, reducida a 512×320. La primera vez reduce todas
  las fotos y las deja en `data/cache/`; las corridas siguientes ya no las decodifican.
- **Aumento de datos:** leve (brillo, contraste, pocos grados de giro, desplazamiento y zoom
  pequeños). Nunca voltea ni recorta la cédula.
- **Resultados:** en `runs/<nombre>/` (fuera de git): configuración, historia de cada época,
  puntajes por persona, métricas y los pesos de cada modelo.

`--limit 400 --epochs 2 --folds 0` hace una prueba rápida.

Opciones para comparar variantes (cada corrida queda en su propia carpeta de `runs/`):

| Opción | Para qué |
|---|---|
| `--size 768x480` | Entrada más grande: conserva más detalle fino de las fotos de alta resolución |
| `--arch efficientnet_b0` | Otra red preentrenada |
| `--seed 7` | Repite con otra semilla (la partición no cambia): mide cuánto varía el resultado |
| `--blocks 12`, `--lr 5e-5` | Cuántos bloques se liberan y con qué tasa de aprendizaje |
| `--patience 6` | Épocas sin mejorar antes de parar (6 por defecto desde el 13-nov-2026; las corridas del 12-nov usaron 4) |
| `--dropout 0.5`, `--weight-decay 0.01` | Más regularización contra el sobreajuste (por defecto 0,2 y 0,0001) |
| `--no-pretrained` | Solo para comparar: parte de pesos al azar, sin transferencia de aprendizaje |

### `32_audit_run.py` — ¿el modelo mira el documento o la cámara? (fase 3)
No se conecta a nada. Lee los puntajes de una corrida (`runs/<nombre>/`) y las fotos en disco:

```powershell
python scripts/32_audit_run.py runs/<nombre>
```

El canal decide la cámara y los positivos se concentran en ciertos meses, así que una red podría
puntuar bien aprendiendo eso. El script compara el clasificador contra "modelos" que no miran la
foto (solo canal, tamaño de la foto y periodo de registro) y mide cómo separa dentro de grupos
donde cámara y fecha son iguales para todos. También da detección y falsos positivos por ancho de
foto y por periodo. Solo imprime conteos y tasas.

### `33_combine_sides.py` — un puntaje por persona (fase 3)
No se conecta a nada. Toma los puntajes de una o varias corridas del frente y del reverso:

```powershell
python scripts/33_combine_sides.py --front runs/<corrida del frente> --back runs/<corrida del reverso>
python scripts/33_combine_sides.py --front runs/<frente 1> runs/<frente 2> --back runs/<reverso 1> runs/<reverso 2> --name ensemble
```

Con varias corridas de una cara, el puntaje de esa cara es el promedio simple de sus puntajes
(nada se ajusta, todas pesan igual). Todas deben venir de la misma partición.

Compara cada cara sola con tres combinaciones simples (promedio, la más alta, la más baja); se
queda con el promedio, que no se entrena. Muestra cuánto se solapan las dos caras, el puntaje
combinado por parte, canal y origen, y el clasificador junto con las reglas (se marca a una
persona si el puntaje pasa el umbral o si alguna regla falló). Lo deja en `runs/<name>/`
(`combined` si no se da `--name`), que también se puede revisar con `32_audit_run.py`.

### `34_plot_history.py` — curvas de entrenamiento (fase 3)
No se conecta a nada. Dibuja, para cada parte de una corrida, la pérdida de entrenamiento contra
la de validación por época, y marca la época cuyos pesos se conservaron:

```powershell
python scripts/34_plot_history.py runs/<corrida>
python scripts/34_plot_history.py runs/<corrida> --metric accuracy
```

Deja `curves_loss.png` o `curves_accuracy.png` en la carpeta de la corrida. Solo sirve para
corridas hechas desde el 13-nov-2026, que son las que guardan la pérdida de validación.

**Positivos que no se usan.** Si en `data/<STAGE>/review/modalities.csv` hay positivos que la
revisión visual marcó como documento auténtico (cédula real robada), `31_train.py` no los usa
para entrenar, validar ni puntuar, y lo dice al empezar. El modelo busca falsificaciones y esas
fotos no lo son.

**Modelo final.** Con `--final` entrena la red para clientes nuevos: usa a todas las personas, sin
apartar ninguna parte, y un número fijo de épocas que toma de una corrida de validación cruzada
de la misma receta (la mediana de las épocas que conservaron sus cinco partes). Deja `final.pt`.
Ese modelo no tiene resultado propio: el esperado es el de la validación cruzada.

```powershell
python scripts/31_train.py --side front --arch efficientnet_b0 --final --epochs-from runs/front_efficientnet_p6 --name front_efficientnet_final
python scripts/31_train.py --side back --arch efficientnet_b0 --final --epochs-from runs/back_efficientnet_p6 --name back_efficientnet_final
```

### `50_calibrate.py` — calibración e indicador (fase 5)
No se conecta a nada. Lee los puntajes de la validación cruzada de una corrida del frente y una
del reverso, el manifest y el resultado de las reglas:

```powershell
python scripts/50_calibrate.py --front runs/front_efficientnet_p6 --back runs/back_efficientnet_p6
```

- Le da a cada negativo con fotos el peso de los negativos que representa (por canal y mes), para
  que las cifras hablen de la población con etiqueta y no de la muestra de entrenamiento.
- Calibra el promedio de los dos puntajes a una probabilidad de falsificación (compara Platt e
  isotónica; usa la isotónica).
- Mide el peso de cada regla: cuánto multiplica un FAIL las probabilidades a favor, tomando la
  razón más baja que admiten los datos. Una regla sin fallos suficientes pesa 1.
- Reporta si el indicador está calibrado y qué manda a revisión cada corte posible.

Todo lo que reporta es anidado: lo que se aplica a una parte se ajustó sin ella. Deja en
`runs/indicator/` los parámetros para clientes nuevos (`calibration.json`), una fila por persona
(`indicator_oof.csv`) y las tablas de calibración y de cortes.

### `35_grad_cam.py` — mapas de calor (fase 3)
No se conecta a nada. Para cada persona usa el modelo de la parte que no la vio y calcula con
Grad-CAM qué zonas de la foto empujaron el puntaje hacia "falsa":

```powershell
python scripts/35_grad_cam.py runs/<corrida de una cara>
python scripts/35_grad_cam.py runs/<corrida de una cara> --queue runs/ensemble/missed_positives.csv
```

Deja dos cosas:

- En `runs/<corrida>/`: `cam_mean.png` (el mapa promedio de cuatro grupos: positivos detectados,
  positivos no detectados, negativos marcados y negativos no marcados) y `cam_summary.csv` (qué
  parte del calor cae en el borde de la foto, donde hay fondo y no cédula). Son promedios de
  muchas personas; no contienen fotos.
- En `data/<STAGE>/cam/<corrida>/<grupo>/`: por persona, la foto junto a la misma foto con el
  mapa encima. Son fotos de cédulas: quedan en `data/` y no se versionan. `--pictures` elige el
  grupo (por defecto los positivos no detectados), `--queue` una lista y `--limit` cuántas.

El mapa dice en qué zona se apoyó el modelo, no que esa zona esté falsificada.

### `36_face_repeats.py` — caras repetidas (fase 3)
No se conecta a nada. Revisa un riesgo de la validación cruzada: si la misma persona pone su cara
en varias cédulas y una queda en entrenamiento y otra en prueba, el modelo del frente podría
reconocer la cara en vez de la falsificación. Es una comprobación de la evaluación, no una parte
del indicador.

Lee la cara de cada foto del frente con dos modelos públicos de OpenCV (YuNet la encuentra, SFace
la describe con 128 números) y compara todos los pares de personas. Los dos archivos van en
`data/face_models/` y se bajan una sola vez de
[opencv_zoo](https://github.com/opencv/opencv_zoo): `face_detection_yunet_2023mar.onnx` y
`face_recognition_sface_2021dec.onnx`.

```powershell
python scripts/36_face_repeats.py --run runs/ensemble
```

Dos caras coinciden cuando su parecido supera un umbral; el umbral se elige para que dos
personas distintas (dos negativos) coincidan por azar solo una vez cada 100.000 pares
(`--chance`). Imprime solo conteos: caras encontradas por canal, coincidencias observadas contra
las esperadas por azar, grupos de personas con la misma cara y, con `--run`, la detección de los
positivos cuya cara estaba en el entrenamiento de su modelo contra la de los demás. El puntaje
del reverso sirve de control: una cara no puede ayudar al modelo del reverso.

Deja en `data/<STAGE>/faces/` (datos biométricos; nunca se versionan) las descripciones de las
caras, `face_groups.csv` y, en `pairs/`, imágenes de pares de positivos que coinciden, para que
una persona confirme que las coincidencias son reales.

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
App local de Streamlit. Muestra frente y reverso de un positivo a la vez; un clic en la
modalidad la guarda en `data/review/modalities.csv` y pasa al siguiente. Retoma donde quedó, y
"Undo last" deshace la última respuesta.

Por defecto recorre todos los positivos. Con `--queue` muestra solo a las personas de una lista,
en ese orden; así se revisan los positivos que el clasificador no detectó, que es para lo que se
usa (la revisión de todos se descartó por el tiempo que toma):

```powershell
streamlit run scripts/13_visual_review.py -- --queue runs/ensemble/missed_positives.csv
```

No muestra los puntajes del modelo, para que la respuesta no se apoye en ellos.

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
