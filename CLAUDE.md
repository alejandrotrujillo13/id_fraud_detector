# CLAUDE.md — id_fraud_detector

Proyecto de grado de Alejandro Trujillo López (Especialización en Inteligencia Artificial, UEIA):
un detector de fraude en fotos de cédulas de ciudadanía colombianas, para la plataforma de
microcréditos de SisteYa (Grupo Ya). Alejandro es ingeniero de software en SisteYa y conoce bien
los repositorios `mikro_back` y `mikro_front`; este proyecto es **independiente** de ambos.

## Qué leer antes de hacer cualquier cosa

1. **`docs/context.md`** — historia completa de la conversación en la que se diseñó el proyecto:
   qué se decidió, qué se descartó y por qué, hallazgos sobre los datos, estado actual y próximos
   pasos. Léelo entero antes de proponer o cambiar algo; casi todas las dudas ya se discutieron.
2. **`docs/plan.md`** — el plan técnico vigente (versión final). Es la fuente de verdad. Si una
   decisión cambia, actualízalo tú mismo en la misma tarea.
3. `README.md` — estructura y comandos.
4. **`docs/data_summary.md`** — todas las cifras de los datos (cuántos clientes, positivos por
   origen, canal, fotos, legibilidad, reglas), reunidas para el informe y la presentación. Se
   regenera con `python scripts/17_data_summary.py`, que deja las tablas en
   `data/<STAGE>/summary/*.csv`. Si cambia el dataset, actualiza ese documento en la misma tarea.

## Estado actual (2026-11-11) — qué existe y qué NO

- **Existe:** el paquete instalable `id_fraud_detector` (`src/id_fraud_detector/`, instalado con
  `pip install -e ".[review]"`) con las consultas y el etiquetado (`dataset.py`), el motor de
  reglas estructurales (`rules/`) y el recorte con OpenCV (`crop.py`); los scripts de la fase 1
  (`10` a `17`), de la fase 2 (`20`) y de la fase 3 (`30` a `33`) en `scripts/`; y pruebas en `tests/`
  (`python -m unittest discover tests`). Todo se probó con datos sintéticos y con una base y un
  S3 simulados.
- **Corrido contra `dev`:** el 11-nov-2026 Alejandro corrió las verificaciones previas con
  `STAGE=dev` (tenant `demo`) y terminaron sin error: la conexión por SSO y Secrets Manager y las
  tres consultas (clientes, OCR, registro común) funcionan. `demo` es muy pequeño (46 clientes
  desde 2025, 4 negativos, ningún positivo), así que solo valida que el código corre.
- **Corrido contra `live`:** `scripts/10_check_database.py` y `scripts/11_export_dataset.py
  --no-images` (11-nov-2026). Resultados y lectura en `docs/context.md` §9 y §10. Lo más
  importante: el OCR guardado depende del canal (~90 % en app y WhatsApp, ~10 % en comercio, que
  es la mayoría); tomar el texto de la otra marca recuperó solo 316 reversos, así que 137 de 403
  positivos y 4.051 de 28.234 negativos tienen texto del reverso; y 21 de los 40 positivos de
  Llévalo Ya son copias de fotos de una validación anterior en otra marca.
- **Existe en `data/live/`:** el manifest por persona (28.253 filas: 381 positivos y 27.872
  negativos, regenerado el 12-nov-2026), el OCR guardado por cliente y el OCR nuevo de 20
  positivos de ESMIO (`ocr_fresh/`). **No** se han descargado imágenes ni se ha corrido
  `12_ocr_coverage.py`.
- **Una fila por persona (decisión del 11-nov-2026, ya corrida en `live`):** la exportación cruza
  ESMIO y Llévalo Ya por número de documento y deja una sola fila por cédula
  (`dataset.one_row_per_person`). Las fotos son las de la
  marca de la fila que queda. **No** se usan las fotos del registro común
  (`public.customers_commonclient`): solo existen para quien completó una validación.
- **Prueba del OCR nuevo (12-nov-2026, `docs/context.md` §12):** 40 fotos de 20 positivos, sin
  errores. Fotos de 2,13 MB en promedio (120 GB todas), ~5,6 horas para la lista completa. En
  el reverso, Rekognition devolvió el código en 13 de 20 y con el filtro de confianza ≥ 80 solo
  se interpreta en 4.
- **Etapa intermedia (12-nov-2026, `docs/context.md` §13):** leídos los 381 positivos y 300
  negativos (fotos de 661 personas en `data/live/images/`). El código del reverso se lee en el
  60 % de las fotos de app y en el 12 % (negativos) y 21 % (positivos) de las de comercio; la
  causa es la resolución de la cámara de comercio (1280×720 o menos), no el encuadre. Reglas:
  38 de 381 positivos con algún `FAIL` (24 de los 79 de la red de la tienda 623 y 14 de los 302
  restantes), 0 de 300 negativos, pero solo 69 negativos evaluables. `R_SEX` no se evalúa en
  nadie (el patrón no encaja con el OCR real; pendiente). **La corrida completa sigue en
  pausa.**
- **OCR nuevo (decisión del 11-nov-2026, ya en el código; solo se corrió la prueba de 20):** `14_fresh_ocr.py` vuelve
  a pasar `detect_text` de Rekognition por las fotos de todas las personas, tengan o no texto
  guardado, y guarda el texto en `data/<STAGE>/ocr_fresh/`. El consumo (~56 USD) lo paga
  Alejandro con una cuenta de AWS propia, distinta de la corporativa: perfil con nombre en
  `REKOGNITION_PROFILE`, nunca la sesión de SSO. `12_ocr_coverage.py` usa ese texto por defecto
  (`--ocr stored` para el guardado). Detalle en `docs/context.md` §11. Alejandro confirmó el
  11-nov-2026 que tiene la autorización de la empresa para procesar las fotos en esa cuenta.
- **Prueba de ampliación (12-nov-2026):** `scripts/15_enlarged_ocr_test.py` envió la cédula
  recortada y ampliada al doble para 200 personas; el código legible pasó de 22,5 % a 23,5 %
  (ruido). El límite es la cámara: las fotos se envían tal como están. No volver a proponer
  recortar o ampliar para mejorar el OCR.
- **OCR nuevo de la muestra (12-nov-2026, `docs/context.md` §14):** leídos los 381 positivos y
  los 4.017 negativos marcados para imágenes (8,99 USD en total). Ningún negativo falla ninguna
  regla (0 de 977 evaluables; tope de falsos positivos de 0,31 %). Solo `R_REGISTRAR` (13) y
  `R_REPEATED_BACK` (28) detectan positivos; las demás reglas no fallan en nadie. En
  `data/live/images/` están las fotos de los 4.017 negativos y de 361 positivos (15 GB); faltan
  las de 20 positivos.
- **Revisión visual aplazada (decisión del 12-nov-2026, ya en `docs/plan.md`):** no se revisan
  los 381 positivos antes de entrenar; después del primer modelo se revisan solo los que no
  detecta, con `13_visual_review.py`.
- **Nuevas fuentes de positivos (12-nov-2026, `docs/context.md` §15 y §16), ya exportadas:**
  AMORETT (17), Finandina con Libre Destino (4 nuevos) y cédulas confirmadas como falsas por un
  analista en el verificador (8). **El manifest vigente tiene 28.292 personas: 410 positivos y
  27.882 negativos.** Con texto nuevo de Rekognition: los 410 positivos y 4.257 negativos
  (9,53 USD en total). Reglas: 49 de 410 positivos con algún fallo, 1 de 4.257 negativos. El
  historial del verificador está en `data/live/private/verifier_history/casos/` (840 casos,
  números reales, solo local). Las cifras de este bloque más arriba (381 positivos, 4.017
  negativos) son de corridas anteriores del mismo día.
- **El recorte con OpenCV no funciona con fotos reales** (`20_crop_test.py`, 12-nov-2026: 4 de
  188 fotos). La cédula ya ocupa cerca del 75 % del ancho del cuadro.
- **Decisiones de Alejandro para el clasificador (12-nov-2026; `docs/plan.md` §0 y
  `docs/context.md` §17). No las vuelvas a preguntar:**
  1. Las reglas son evidencia adicional; **el clasificador de imágenes es el componente
     principal**. No se agregan reglas ni se lee el texto de los ~23.600 negativos restantes.
  2. **Se entrena en el equipo de Alejandro** (RTX 2070 de 8 GB), con **PyTorch** (TensorFlow no
     usa la GPU en Windows). Las fotos no salen de la máquina.
  3. **Negativos emparejados con los positivos:** además de la muestra por mes y canal, todos los
     de app y todos los registrados desde septiembre de 2026 (`negatives_always_with_images` en
     `config.yaml`); unos 5.300 con fotos. El sorteo es estable: quien ya tiene fotos en disco
     no sale de la muestra.
  4. **El primer modelo va sin recorte**, con la foto completa reducida.
  5. **Validación cruzada en cinco partes por persona**, con cada red completa (tienda 623,
     AMORETT) en una sola parte. Llévalo Ya ya no es prueba externa.
  6. Las 15 cédulas "falsas" del verificador sin registro de origen **no** se incluyen.
- **Muestra emparejada exportada (12-nov-2026, `docs/context.md` §18):** 28.293 personas, 410
  positivos y 27.883 negativos; **5.707 personas con fotos y con texto nuevo de Rekognition**
  (410 positivos y 5.297 negativos). Gasto total en Rekognition: 11,61 USD. Las cifras vigentes
  están en `docs/data_summary.md`.
- **Fase 3 empezada (12-nov-2026):** existen `evaluation/splits.py`, `evaluation/metrics.py`,
  `model/data.py`, `model/net.py`, `model/train.py`, `scripts/30_make_splits.py` y
  `scripts/31_train.py`. `data/live/splits.csv` ya está hecho (cinco partes). PyTorch 2.14.1 con
  GPU está instalado en `.venv` (`pip install torch torchvision --index-url
  https://download.pytorch.org/whl/cu126`). El entrenamiento se probó con una corrida corta;
  **ningún modelo completo se ha entrenado**.
- **La tienda 623 y AMORETT son una sola red:** comparten reversos y quedaron en un grupo de 100
  personas, todo en la parte 0 de la partición. 100 de los 128 positivos de app son de esa red.
  Los resultados hay que leerlos por origen del positivo, no solo en total.
- **Primer modelo del reverso (12-nov-2026, `docs/context.md` §19;
  `runs/back_mobilenet_v3_large_512x320_20261112_1952`):** detecta el **61 % de los positivos
  con 2 % de falsos positivos** (251 de 410; PR-AUC 0,71; ROC-AUC 0,91), contra el 12 % de las
  reglas. La red tienda 623 + AMORETT, que su modelo nunca vio, se detecta en cerca del 50 %.
  `scripts/32_audit_run.py` muestra que no aprendió la cámara ni la fecha (ROC-AUC 0,92 dentro
  de grupos con igual canal, tamaño de foto y periodo; 0,74 un puntaje que solo usa esos
  datos). Falla en fotos de baja resolución (24 % en comercio con hasta 700 px de ancho).
- **Modelo del frente y puntaje combinado (12-nov-2026, `docs/context.md` §20;
  `runs/front_mobilenet_v3_large_512x320_20261112_2041` y `runs/combined`):** el frente solo
  detecta el 70 % (288 de 410); **el promedio de frente y reverso, el 79 % (322 de 410) con 2 %
  de falsos positivos** (PR-AUC 0,84; ROC-AUC 0,95); con las reglas, el 81 % (332). La red
  tienda 623 + AMORETT, nunca vista por su modelo, se detecta en 77 de 100. La revisión de
  cámara y fecha sigue limpia (ROC-AUC 0,96 dentro de grupos iguales). En comercio con fotos de
  hasta 700 px de ancho detecta el 41 %. Las cifras están en `docs/data_summary.md` §12.
- **Segunda iteración (corrida el 12 y 13-nov-2026, `docs/context.md` §22;
  `docs/data_summary.md` §13):** ocho corridas (otra semilla, entrada de 768×480,
  EfficientNet-B0 y sin transferencia, para frente y reverso). **El promedio de las cuatro
  variantes preentrenadas de cada cara (`runs/ensemble`) detecta el 87 % (356 de 410) con 2 % de
  falsos positivos** (PR-AUC 0,89; ROC-AUC 0,96); con las reglas, 360. La red nunca vista: 93
  de 100. Comercio con fotos de hasta 700 px: 59 %. Revisión de cámara y fecha limpia (0,98
  dentro de grupos iguales). Cambiar solo la semilla movió el frente 7 puntos, así que las
  diferencias entre corridas sueltas no prueban qué variante es mejor; la entrada grande no
  ayudó; sin transferencia queda en 17 %. `33_combine_sides.py` ahora recibe `--front` y
  `--back` con una o varias corridas.
- **Curvas y paciencia (13-nov-2026, `docs/context.md` §23):** `history.csv` guarda ahora pérdida
  y exactitud de entrenamiento y de validación; `scripts/34_plot_history.py` las dibuja. La
  paciencia por defecto pasó de 4 a 8 y luego a 6 (pedidos de Alejandro; hoy es 6). Las curvas muestran sobreajuste
  después de las épocas 4 a 8, controlado por la parada temprana. Con paciencia 8 el reverso de
  EfficientNet-B0 dio 71,7 % contra 68,5 % (una corrida); el frente no se ha probado con 8, y
  `runs/ensemble` sigue hecho con paciencia 4 (ver el experimento de regularización, más abajo).
- **Grad-CAM (13-nov-2026, `docs/context.md` §25):** existe `model/cam.py` y
  `scripts/35_grad_cam.py`; corrido sobre EfficientNet-B0 de frente y de reverso. En el frente,
  cuando el modelo marca a alguien el calor está en la zona de la foto de la persona; en el
  reverso se reparte en varias zonas de la cédula. En ningún caso mira el fondo. Riesgo abierto:
  si el frente aprende la foto pegada o caras repetidas; se midió después (ver "Caras repetidas"). `31_train.py` tiene `--dropout` y
  `--weight-decay`.
- **Experimento de regularización (corrido el 13-nov-2026, `docs/context.md` §26):** EfficientNet-B0
  con paciencia 6, receta actual contra regularizada (`--weight-decay 0.01 --dropout 0.5
  --blocks 3`). La regularizada es peor: las dos caras detectan 82,4 % contra 86,1 % (−3,7
  [−6,1; −0,5]) y la red nunca vista baja de 94 a 84 de 100; la pérdida de validación sigue
  subiendo igual. **Se queda la receta actual; no pruebes más variantes de regularización.**
  Paciencia: con la misma semilla solo cambia dónde se corta; 6 contra 4 da lo mismo (353 contra
  355 en las dos caras) y la mejora del reverso con 8 venía del modelo de una sola parte.
  Recomendación de Claude, pendiente de que Alejandro la confirme: no rehacer `runs/ensemble`.
- **Combinar frente y reverso (13-nov-2026, `docs/context.md` §27):** se midieron máximo, pesos
  ajustados y regresión logística sobre los dos puntajes; ninguno supera al promedio simple más
  allá del ruido. Se queda el promedio simple; no propongas un modelo de fusión.
- **Caras repetidas (13-nov-2026, `docs/context.md` §28):** existe `evaluation/faces.py` y
  `scripts/36_face_repeats.py` (modelos públicos YuNet y SFace de OpenCV en `data/face_models/`,
  todo local). Alejandro dijo que una persona puede aparecer en varios fraudes; medido: 12
  positivos comparten cara con otro positivo y solo 5 la tenían en el entrenamiento de su modelo
  (entre 5 y 11 según el umbral); la red tienda 623 + AMORETT no repite caras. No se detectan
  mejor que los demás (4 de 5 contra 83 %). **El 87 % no está inflado por caras de forma medible
  y la partición no se rehace.** Límite: no se sabe cuántas caras repetidas se le escapan al
  comparador; las 9 parejas están en `data/live/faces/pairs/` para que Alejandro las confirme.
  Es una comprobación de la evaluación, no una parte del indicador.
- **Modelo final (preferencia de Alejandro, 13-nov-2026):** los cinco modelos de cada corrida
  (`fold0.pt` a `fold4.pt`) solo miden la receta. El modelo para clientes nuevos se entrenará
  con todos los datos, cuando la validación cruzada sea aceptable y la receta quede fija (fase
  5; detalle al final de `docs/context.md` §23). No se ha entrenado.
- **Modelo final: EfficientNet-B0, una red por cara (decisión de Alejandro, 13-nov-2026, "por
  ahora"; `docs/plan.md` §0, decisión 7).** Dos redes al puntuar, promedio simple de frente y
  reverso, receta actual. Razón: detecta lo mismo que el promedio de cuatro variantes (86,1 a
  86,6 % contra 86,8 %) y como API es más simple y rápido (0,3 s por cliente en CPU contra
  0,9 s, medido en el equipo de Alejandro). El promedio de EfficientNet-B0 con varias semillas
  no se ha medido; no lo entrenes sin que él lo pida.
- **Revisión visual hecha (13-nov-2026, `docs/context.md` §24):** de los 54 positivos que el
  promedio no detecta, Alejandro ve 48 como falsificados, 3 como auténticos y 3 sin poder
  decidir. Los fallos son del modelo, no de la etiqueta. No registró el tipo de falsificación
  (no era necesario).
- **Las 3 auténticas se excluyen (decisión de Alejandro, 13-nov-2026; `docs/plan.md` §0,
  decisión 8):** son cédulas auténticas robadas (la selfie no coincidía con la foto del
  documento), no falsificaciones. `model/data.excluded_by_review` las lee de
  `data/<STAGE>/review/modalities.csv` y `31_train.py` no las usa para entrenar, validar ni
  puntuar; no pasan a negativos. Las corridas anteriores las tienen como positivos: sin ellas
  el modelo final detecta 353 de 407 (86,7 %) en vez de 353 de 410 (86,1 %). Calibración,
  backtest y modelo final deben excluirlas igual. Solo se revisaron los no detectados, así que
  puede haber más cédulas auténticas robadas entre los positivos detectados.
- **Fase 5 empezada (13-nov-2026, `docs/context.md` §31):** existen `risk.py` (calibración,
  peso de las reglas, indicador), `evaluation/population.py` (pesos de muestreo),
  `scripts/50_calibrate.py` y el modo `--final` de `31_train.py`. Calibración isotónica sobre
  `runs/front_efficientnet_p6` y `runs/back_efficientnet_p6`, en `runs/indicator/`. Las 5.704
  personas con fotos representan a 28.290 con etiqueta (falsas: 1,44 %). Pesos de las reglas:
  `R_REPEATED_BACK` 78, `R_REGISTRAR` 27,8, `R_PRODUCTION` 1,8; las demás 1. Cortes medidos:
  2 % de legítimos ↔ probabilidad 0,03 ↔ 352 de 407 falsas; 0,1 % ↔ 0,67 ↔ 221. **Los cortes
  y la forma del indicador no están decididos:** Alejandro pidió varias explicaciones y aún
  no eligió. Valores provisionales: revisión al 2 % de legítimos (lo dice el plan), retención al
  0,1 %, indicador = probabilidad × 100. Explícale con ejemplos de un solo cliente, no con
  "grupos" ni jerga. **El modelo final no se ha entrenado** (los comandos están en el README).
- **Motivo del fraude (Alejandro, 13-nov-2026):** la base no distingue cédula falsificada de
  cédula auténtica robada, así que no se puede saber cuántas robadas hay entre los positivos
  detectados; es una limitación que se reporta. Sus propuestas de mejora para la empresa
  (mejor captura de fotos, registrar el motivo del fraude, reentrenar después) están en
  `docs/plan.md` §8.1 y van en el informe como recomendaciones, no como trabajo de este proyecto.
- **Siguiente paso:** que Alejandro entrene el modelo final (dos comandos) y confirme los
  cortes; luego `pipeline.py` (una función: fotos y texto del reverso -> resultado) y el
  backtest, que necesita los montos de los créditos (script nuevo de solo lectura que él corre). Sigue pendiente que Alejandro mire las 9
  parejas de caras. No hagas búsqueda amplia de hiperparámetros: con 410
  positivos, probar muchas variantes sobre la misma validación cruzada infla el resultado.
- **NO existe:** modelo final entrenado, `pipeline.py`, backtest, ninguna integración con
  `mikro_back`.

## Qué es la solución (resumen; detalle en docs/plan.md §3 y §4)

Dos componentes que alimentan un **indicador de riesgo de falsificación 0–100** por cliente:

1. **Motor de reglas estructurales** sobre el texto del reverso de la cédula, que ya está
   extraído por Amazon Rekognition y guardado en la base (`data_back_id`). Reglas deterministas:
   registrador firmante vs. fecha de producción, número del frente = número del código, sexo,
   producción ≥ expedición y ≥ 18 años, formato del código, meses de 3 letras, reverso repetido
   entre personas distintas. **Cada regla devuelve PASS / FAIL / NOT_EVALUABLE; solo FAIL suma.**
2. **Clasificador de imágenes por transferencia de aprendizaje** (MobileNetV3 o EfficientNet
   preentrenada en ImageNet, afinada, en PyTorch), dos instancias de la misma receta: frente y
   reverso. Es el componente principal. El primer modelo usa la foto completa reducida, sin
   recorte (el recorte con OpenCV falló en fotos reales). Para lo visual: foto pegada,
   holograma, calidad de impresión, plantillas repetidas. Grad-CAM para explicar al analista.

Las reglas del punto 1 quedaron como evidencia adicional: solo 49 de 410 positivos fallan alguna,
porque en las fotos de comercio casi nunca se puede leer el código.

**Un solo tipo de modelo, no varios.** Se descartaron explícitamente (no los vuelvas a proponer
sin leer el porqué en `docs/context.md`): autoencoder para anomalías, ELA, comparación de la
imagen de la firma del registrador, línea de tiempo del consecutivo/serie, variantes de plantilla
por época, estimación de edad por rostro, juicios visuales de un LLM como regla automática,
fusor complejo de varios modelos, YOLO como primera opción de recorte.

## Convenciones estrictas

- **Código, identificadores, nombres de archivos y comentarios en inglés. Documentación
  (`README.md`, `docs/`) en español.** Nombres de carpetas y archivos con `_`, nunca con `-`.
- Python 3.11+. **Sin Django.** Este proyecto no importa nada de `mikro_back`; se conecta a
  PostgreSQL con `psycopg2` en modo solo lectura (`default_transaction_read_only=on`, verificado
  al abrir cada conexión) y a S3 con `boto3` solo con `GetObject`. **Nunca escribe en producción
  ni modifica la base.**
- **Credenciales:** ninguna en el repositorio ni en `.env`. AWS usa la sesión de SSO de Alejandro
  (`aws sso login`, perfil por defecto); el host, usuario y contraseña de la base salen del
  secreto `sisteya_db_proxy_credentials` de Secrets Manager (el mismo de `mikro_back`). El
  entorno se elige con `STAGE` en `.env` (`dev`, `test`, `live`): la base es `sisteya_<STAGE>` y
  las salidas van a `data/<STAGE>/`. Con `STAGE=dev` solo se usa el tenant `demo`
  (`tenants_by_stage` en `config.yaml`); el dataset real es `esmio` y `llevaloya` en `live`.
  Todo eso vive en `src/id_fraud_detector/common.py`. No leas ni edites `.env`.
- **Rekognition va en otra cuenta.** Es el único servicio con costo y no se cobra a la empresa:
  usa el perfil de AWS nombrado en `REKOGNITION_PROFILE` (cuenta de Alejandro; sus llaves están
  en `~/.aws/credentials`, no en el repositorio ni en `.env`) y exige además el número de esa
  cuenta en `REKOGNITION_ACCOUNT_ID`. **Ninguno de los dos tiene valor por defecto** (pedido de
  Alejandro, 12-nov-2026: que nunca se use la cuenta corporativa por accidente).
  `common.rekognition_client()` es el único lugar donde se crea un cliente de un servicio con
  costo, y se niega a correr si falta alguno de los dos, si las llaves del perfil son de otra
  cuenta o si es la misma cuenta de la sesión de SSO. No agregues valores por defecto ni rutas
  alternas a esa comprobación. No llames a
  Rekognition ni a ningún otro servicio de pago con la sesión corporativa. Las fotos van de S3 a
  memoria y de ahí a Rekognition; no se guardan en la otra cuenta.
- Tablas (por esquema de tenant, `django-tenants`): `users_client`, `credits_credit`,
  `users_customeridvalidationresult`. Esquemas que se usan: `esmio` y `llevaloya`.
- **Privacidad (Ley 1581/2012):** `data/` está en `.gitignore` y no se versiona nunca. Los
  clientes se identifican con `anon_id = sha256(ANON_SALT:tenant:document_number)[:16]`.
  `data/<STAGE>/private/` guarda documentos reales (`id_map.csv`, `rules_values.csv`): solo
  local. Nunca subir fotos a Colab
  público, Roboflow público ni servicios similares. No escribir números de cédula ni nombres de
  personas en código, docs ni commits.
- Reglas (`src/id_fraud_detector/rules/engine.py`): una lectura rota del OCR es `NOT_EVALUABLE`,
  nunca `FAIL`. Los pesos del indicador **salen de las tasas medidas** con
  `scripts/12_ocr_coverage.py` (falsos positivos sobre los negativos, detección sobre los
  positivos); no se inventan.
- **Organización (detalle en docs/plan.md §6b):** un solo repositorio para todo el proyecto. La
  lógica reutilizable va en el paquete `src/id_fraud_detector/` y se importa como
  `from id_fraud_detector...` (nunca con `sys.path`). `scripts/` solo tiene puntos de entrada
  delgados, nombrados `FN_name.py` donde `F` es la fase (`10`–`19` fase 1, `20`–`29` fase 2, …).
  Dependencias en `pyproject.toml`, por grupos. Las carpetas de fases futuras (`model/`,
  `evaluation/`, `service/`, `infra/`) se crean cuando empieza la fase, no antes. Los cambios a
  `mikro_back` (fase 6) no viven aquí.
- Ante una decisión que cambie el plan (nuevo modelo, nueva regla, nueva fuente de datos),
  **pregunta a Alejandro antes**; no la implementes y luego la expliques.

## Etiquetado del dataset (config.yaml; decisión cerrada)

- **POSITIVE (cédula falsa):** `fraud_status = 'FRAUD'` sin ningún crédito en estado `PAID`;
  o `fraud_status = 'POSSIBLE FRAUD'` del tenant `esmio` registrado entre **2026-09-13 y
  2026-10-04** sin crédito `PAID` (red de fraude de la tienda 623, 79 clientes verificados).
- **POSITIVE, fuentes confirmadas por el equipo de fraude (decisión del 2026-11-12):** con las
  mismas condiciones (sin crédito `PAID`, registro desde 2025, ambas fotos), sea cual sea su
  `fraud_status`: clientes de `esmio` con crédito en el comercio AMORETT del 14-oct al 10-nov de
  2026; clientes de `esmio` con crédito desembolsado a BANCO FINANDINA en un comercio "LIBRE
  DESTINO" (sin fechas); y cédulas que un **analista** confirmó como falsas en el verificador del
  equipo (`data/<STAGE>/private/verifier_history/casos/`, `real_origen = Analista`). Las dos
  primeras están en `confirmed_fraud_sources` de `config.yaml`. No se usan las cédulas que el
  verificador marcó solo por puntaje (`real_origen = Claude`) ni las que no registran origen. Uno
  de estos clientes con crédito `PAID` queda sin etiqueta (ni positivo ni negativo).
- **NEGATIVE (cédula legítima):** sin `fraud_status` y con al menos un crédito `PAID`.
- **Fuera del dataset:** `VERIFICATION REQUIRED`, `VALIDATION REQUEST DENIED`, los demás
  `POSSIBLE FRAUD` (incluidos ~170 posteriores al 4-oct-2026: el usuario decidió NO incluirlos),
  y clientes sin marca y sin crédito pagado.
- Solo registrados desde **2025-01-01**. Conteos reales al 12-nov-2026, una fila por persona:
  410 positivos (390 esmio, 20 llevaloya) y 27.882 negativos (15.921 esmio, 11.961 llevaloya).
  Detalle en `docs/data_summary.md`.
- La fecha de registro es `terms_and_conditions_timestamp` (`users_client` no tiene `created_at`).

## Contexto de los sistemas de SisteYa (solo lectura; detalle en docs/plan.md Anexo A)

- Repos hermanos en el equipo de Alejandro: `C:\Users\Alejandro\mikro\mikro_back` (Django 5.1 +
  DRF, multi-tenant, Lambda/Zappa) y `C:\Users\Alejandro\mikro\mikro_front` (Next.js). **No los
  modifiques desde aquí**; sirven solo como referencia de cómo se capturan y guardan las fotos.
- Fotos en S3, bucket `sisteya-media`. La copia con historial de versiones:
  `{live|dev}/{tenant}/clients/front_ids/front_id_image_{client_id}.png` y
  `back_ids/back_id_image_{client_id}.png`; la key completa está en las columnas
  `users_client.front_id_image` / `back_id_image`. La copia cruda `clients/id-validations/
  frontal-id-photos/{document_number}.png` se sobreescribe en cada intento (no usarla).
- OCR de Rekognition (`detect_text`, `detect_labels`, `detect_faces`) ya guardado en
  `users_customeridvalidationresult.data_front_id` y `data_back_id` (JSON crudo).
- La app toma la foto con `react-webcam`, captura el **cuadro completo de la cámara**, lo pasa
  por canvas y lo envía como PNG: sin EXIF, con fondo alrededor de la cédula, resolución variable.
  La guía visual en pantalla (marco con forma de cédula) **no recorta ni valida nada**.
- Validación actual en `mikro_back` (reglas, no modelo): etiqueta "Document/Id Cards", texto con
  nombre y número, rostro presente en el frente, similitud rostro-selfie, liveness. Los fraudes
  conocidos pasan todas esas pruebas.
- Existe un motor de puntaje de fraude en `mikro_back/docs/fraud-score/` (reglas FR01–FR14, modos
  SHADOW/ENFORCE). La integración futura del indicador sería una regla más (`FV_DOC_IMAGE_RISK`),
  en modo SHADOW. Eso es fase 6, opcional.

## Cómo trabajar con Alejandro

- Habla en español. Es directo y hace preguntas de fondo ("¿esto es necesario o lo pones por
  cubrir temas de clase?"): responde con honestidad, separa lo necesario de lo opcional.
- Prefiere soluciones simples y justificadas con datos sobre arquitecturas grandes.
- No le gustan los nombres con guion; usa `_`.
- No ejecuta nada en producción sin revisarlo, y **no quiere correr consultas a mano en pgAdmin**:
  toda consulta va dentro de un script de solo lectura (`scripts/10_check_database.py` o uno
  nuevo) que él mismo ejecuta. Claude no corre scripts contra la base ni contra S3. Cuando
  comparta resultados, regístralos en `docs/context.md`.
- El documento espejo de `docs/plan.md` vive también en el proyecto de Claude.ai
  ("ESP_IA Detector de fraudes"); si cambias `docs/plan.md`, avísale para que sincronice.
