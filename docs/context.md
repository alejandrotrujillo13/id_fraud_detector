# Contexto: cómo se llegó al diseño actual

Resumen fiel de la conversación entre Alejandro y Claude (Cowork) del 5 al 9 de noviembre de 2026
en la que se diseñó este proyecto. Está escrito para que otro asistente (Claude Code en VS Code) o
una persona nueva entienda **por qué** las cosas son como son, no solo qué son. El plan técnico
vigente está en `docs/plan.md`; este archivo cuenta la historia y las razones.

---

## 1. El problema de negocio

SisteYa opera una plataforma de microcréditos multi-tenant (marcas como ESMIO y Llévalo Ya). En el
registro, el cliente toma tres fotos con el celular: frente de la cédula, reverso y selfie con
prueba de vida. El backend (`mikro_back`) valida con Amazon Rekognition que la imagen parezca un
documento, que el texto contenga nombre y número, que haya un rostro en el frente y que el rostro
coincida con la selfie.

Ese esquema falla contra las falsificaciones reales. En septiembre–octubre de 2026 una red de
fraude en ESMIO (tienda 623, créditos de libre inversión) registró 65 identidades robadas de
personas mayores con buen historial, tomó 128 créditos por COP 103,8 millones y nunca pagó. El
análisis interno (`mikro_back/docs/fraud-score/01-problema.html`) encontró que las cédulas eran
**composiciones digitales sobre plantilla**: el rostro del estafador va en la cédula (por eso la
similitud con la selfie es ~100 y la prueba de vida pasa), el OCR sale limpio, y el reverso
reutiliza seriales de plantilla (el serial 22746003 aparece en 14 casos; 00359735 y 0014238152A
en 13 cada uno). Nada automático los detuvo; solo un analista humano mirando la foto podría
haberlo hecho, y no siempre hay uno.

El proyecto de grado (documento "ENTREGA", presentación y cronograma en la carpeta "Proyecto de
grado") propone un prototipo de IA/visión por computador que analice las fotos de la cédula y
entregue un **indicador de riesgo de falsificación** que apoye al analista de validación. Objetivos
específicos: dataset limpio y etiquetado; preprocesamiento y extracción de características; modelo
inicial de clasificación válido/sospechoso, evaluado e iterado. Metodología: CRISP-DM.

## 2. Qué se vio en la materia de Deep Learning (y qué se usa)

Notebooks de la carpeta "6. Deep Learning": MLP y CNN en Keras (FashionMNIST), CNN binaria con
`ImageDataGenerator` (gatos/perros), transfer learning con VGG16/InceptionV3/MobileNetV3 (Taller 1,
hojas de árboles), YOLO con ultralytics y Roboflow (Taller 2), autoencoders para denoising y
detección de anomalías con umbral μ+2σ (Taller en clase 1), embeddings de imágenes en Orange.

Alejandro fue explícito: **no hay que usar todo lo visto en clase, sino lo que mejor resuelva el
problema.** Se usa transfer learning (indispensable), la receta de carga por carpetas y métricas, y
Grad-CAM. YOLO solo como respaldo del recorte. El autoencoder se descartó (ver §5).

## 3. Cómo se guardan las fotos (hallazgos en `mikro_back` y `mikro_front`)

- S3, bucket `sisteya-media`, prefijo por entorno (`live`/`dev`) y tenant. Dos copias de cada foto:
  - `clients/id-validations/frontal-id-photos/{document_number}.png` (y `back-id-photos/`,
    `selfie-photos/`): subida cruda con `s3transfer`; el nombre es el número de documento, así que
    **se sobreescribe en cada intento**. No sirve como fuente del dataset.
  - `clients/front_ids/front_id_image_{client_id}.png` y `back_ids/...`: `ImageField` del modelo
    `Client` con `django-s3-storage` y `AWS_S3_FILE_OVERWRITE=False`, que agrega sufijo y
    **conserva versiones**. La key completa está en las columnas `users_client.front_id_image` /
    `back_id_image`. **Esta es la fuente del dataset.**
- `users_customeridvalidationresult` (1:1 con `users_client`) guarda la salida cruda de Rekognition
  en `data_front_id` y `data_back_id` (`detect_labels`, `detect_text` con cajas y confianza por
  fragmento, `detect_faces`), más `data_face_match`. Es el OCR que usa el motor de reglas, sin
  llamar a Rekognition otra vez.
- Etiquetas de fraude: `users_client.fraud_status` ∈ {FRAUD, POSSIBLE FRAUD, VERIFICATION REQUIRED,
  VALIDATION REQUEST DENIED} con historial en `users_customerhistoricalfraudstatus`.
- Captura en `mikro_front` (`components/pages/id-validation/StepFrontIdPicture.js`, comercio en
  `commerce/create-customer/FrontIdModal.js`): `react-webcam` pide un stream de 1210 px con
  proporción 213/338 y lo muestra en una caja con proporción de cédula (338/213 ≈ 1,587) con
  `overflow-hidden`, un ícono de persona donde va la foto del titular y, en comercio, una imagen
  guía superpuesta. **Es solo visual; no hay lógica de encuadre.** `doCustomScreenShot` captura el
  cuadro completo del video (`videoWidth × videoHeight`), lo dibuja en un canvas y lo exporta como
  PNG. Consecuencias: la foto trae más fondo del que el usuario vio (muchos celulares ignoran
  `aspectRatio`), no tiene EXIF, y la resolución depende del dispositivo. El usuario puede girar a
  la cámara frontal ("Girar cámara"), de menor calidad.
- Validación actual (`client_id_validation_views.py`): labels aceptados "Document", "Id Cards",
  "Driving License", "License"; texto debe contener nombre, apellido y número; rostro en el frente
  con confianza > umbral; ningún rostro en el reverso; umbrales en `IdValidationConfig`. Hay flujos
  de app (`validate_id_labels_front/back`) y de comercio (`check_frontal_id_picture`,
  `check_back_id_picture`); en comercio existe el permiso `upload_id_validation_photos` (fotos
  subidas como archivo, no capturadas).
- Ya existe un **motor de puntaje de fraude** (`docs/fraud-score/`): 21 variables, reglas FR01–FR14
  y KO, umbrales 4/10, modos SHADOW y ENFORCE, política `COMERCIO_*` en ENFORCE desde el 20-oct-2026.
  Incluye la regla FV_BACK_SERIAL_SHARED_DOCS (serial del reverso compartido, vía `IdBackSerial`).

## 4. Evolución del diseño (qué se propuso, qué preguntó Alejandro, qué quedó)

### 4.1 Primera propuesta (5-nov): pipeline de cuatro etapas y varios modelos
Localización con YOLO + homografía; autoencoder sobre parches para "liveness" del documento
(recaptura de pantalla/impresión); clasificador por transferencia por zonas con rama de residuo de
ruido (ELA/SRM) y embeddings del serial contra vecinos cercanos; fusor logístico → riesgo 0–100.
Justificación: señales locales y finas, pocos positivos, fraudes heterogéneos.

### 4.2 Alejandro preguntó si los varios modelos eran necesarios o por cubrir temas de clase
Respuesta honesta: en parte era cobertura. Se separó:
- **Indispensable:** transfer learning (con decenas o cientos de positivos no se entrena una CNN
  desde cero) y un dataset bien construido con evaluación sin fugas.
- **Depende de los datos:** localización/rectificación. Primero CV clásico; YOLO solo si falla.
- **Solo si hace falta:** análisis por zonas; empezar con el documento completo a 384–512 px.
- **No necesario:** autoencoder. En fotos reales el error de reconstrucción sigue al desenfoque, la
  luz y la textura, no a la falsedad; marcaría fotos malas, no cédulas falsas. Funciona en
  MNIST/ECG porque lo "normal" es simple. Además el fraude documentado es composición digital, no
  recaptura.
- **Experimento futuro:** rama de residuo de ruido; embeddings del serial (la regla OCR ya atrapa
  el 52 %); fusor solo si hay varios scores.
Resultado: **un solo enfoque supervisado** (misma receta para frente y reverso).

### 4.3 Reglas de etiquetado (decisiones de Alejandro)
- Un cliente FRAUD que pagó un crédito no fue fraude por foto → excluido.
- Solo FRAUD cuenta como positivo; los otros estados no se sabe si fueron por foto → fuera del
  dataset (ni positivos ni negativos).
- Negativo = sin marca y con al menos un crédito pagado (pagar es la mejor evidencia de identidad
  real).
- Solo tenants `esmio` y `llevaloya`; solo registrados desde 2025-01-01 (fotos anteriores pueden
  venir de versiones viejas de la app).
- Claude estimó "pocas decenas" de positivos a partir de los 65 del documento interno. Alejandro
  preguntó de dónde salía el número y aclaró que **no había autorizado consultas a la base**.
  Aclaración: no se ejecutó ninguna consulta; era una estimación del documento. Se le entregó SQL
  para pgAdmin y él lo corrió.

### 4.4 Conteos reales (ejecutados por Alejandro, 9-nov-2026, registrados desde 2025-01-01)

| Tenant | Grupo | Clientes | Con frente y reverso |
|---|---|---|---|
| esmio | POSITIVO (FRAUD sin crédito pago) | 296 | 281 |
| esmio | NEGATIVO (sin marca, con crédito pago) | 16.014 | 16.011 |
| esmio | EXCLUIDO otro estado | 13.936 | 11.736 |
| esmio | SIN USO | 112.286 | 88.954 |
| llevaloya | POSITIVO | 45 | 40 |
| llevaloya | EXCLUIDO fraud con pago | 1 | 1 |
| llevaloya | NEGATIVO | 12.077 | 12.075 |
| llevaloya | EXCLUIDO otro estado | 13.300 | 12.597 |
| llevaloya | SIN USO | 109.483 | 94.654 |

Positivos de esmio por mes de registro: dos olas (abr–jun 2025: 25/37/30; feb–may 2026:
9/17/27/18) y goteo constante; **sep–nov 2026 solo 2+5+1**. Conclusión: los 65 fraudes confirmados
de la red de sep–oct 2026 están como POSSIBLE FRAUD, no como FRAUD, y la regla los excluía.

### 4.5 Qué hacer con los 65 de la red (decisión de Alejandro)
- Primero dijo "FRAUD los 65"; luego precisó: **no se cambia la marca en la base**, se incluyen por
  ventana de fechas como parámetro del comando de exportación.
- Consulta por semana de POSSIBLE FRAUD en esmio desde sep-2026: semanas del 14, 21 y 28 de
  septiembre suman 14 + 16 + 49 = **79 clientes (78 con ambas fotos)**, consistente con 65
  confirmados + 16 marcados en la investigación. Ventana confirmada: **2026-09-13 a 2026-10-04**.
- Después del 4-oct hay ~160 POSSIBLE FRAUD más de la app (32/37/44/25/21 por semana) y 9 de
  comercio. Mezcla probable de red que siguió y de clientes retenidos por la política de fraude en
  ENFORCE (con legítimos dentro; la proporción con ambas fotos cae a la mitad). **Alejandro decidió
  no incluirlos.** Quedan fuera del dataset; existen por si el equipo los confirma más adelante.
- Total positivos con ambas fotos: 281 + 78 + 40 ≈ **399**. Negativos ≈ 28.000.

### 4.6 Recorte de la cédula (varias rondas de preguntas)
Alejandro preguntó qué era "usar YOLO", qué significaba "fotos enmarcadas" y por qué se mencionaban
las cajas de `detect_text`. Aclaraciones que quedaron:
- YOLO no clasifica fraude; solo encontraría la cédula dentro de la foto para recortarla.
- Las cajas de `detect_text` encierran palabras, no el plástico: **no sirven para recortar** (el
  texto no llega a los bordes, no dan esquinas). A lo sumo acotan la búsqueda.
- La app muestra un marco visual sin lógica: la cédula puede salir pequeña o desplazada, así que el
  recorte fijo no basta, pero hay un prior fuerte (centro, casi horizontal, ocupa buena parte).
- Plan: recorte grueso a la región del marco + 20 % de margen; dentro, bordes Canny → contorno de
  cuatro lados más grande con proporción ID-1 (≈1,58) → homografía a 1000×630. Sin contorno válido
  → cola de revisión manual. Medir en 50 fotos por tenant; **YOLO solo si recorta bien < 90 %.**
- Revisar en esa muestra si los positivos son fotos reales (fondo, perspectiva) o imágenes planas
  inyectadas sin pasar por la cámara; una captura legítima es PNG de canvas con dimensiones de
  stream de cámara. Ancho, alto, formato y tamaño son features de costo cero.

### 4.7 Legibilidad del texto (preocupación de Alejandro: cámaras malas)
Las reglas se diseñan para **abstenerse, no penalizar**: tres estados PASS / FAIL / NOT_EVALUABLE;
solo FAIL suma. Umbral de confianza del OCR por fragmento (Rekognition la entrega) y forma esperada
(regex del código); si no encaja, es "no leído", no "mal formado". Normalización de confusiones
O/0, I/l/1, S/5, B/8 y tolerancia a un carácter con redundancia. Redundancia: el número aparece en
frente, código del reverso y código de barras; el registrador se fija con un apellido legible.
OCR dirigido como mejora: recortar y ampliar la franja del código tras la rectificación y volver a
pasarla por OCR. Medir cobertura por campo en los 28.000 negativos antes de nada. Opción de
producto (fuera de la tesis): pedir repetir la foto si el código no se lee.

### 4.8 El "Verificador de Cédulas" del equipo y el Excel de firmas (9-nov)
Un compañero de verificación manual construyó un artifact de Claude ("Verificador de Cédulas")
que recibe frente, reverso y selfie, lee el PDF417 con zxing y le pasa a Claude un catálogo de ~30
patrones, 8 familias de fraude (F1 reverso clonado … F8 foto reemplazada), una lista negra de
reversos clonados y la memoria de casos. Se construyó iterando sobre ~90 casos: **3 falsas y 2
auténticas de entrenamiento, 81 sin etiqueta**; pesos de criterio, no validados. Alejandro pidió
analizarlo porque no sabía si todo era válido. Evaluación:
- **Sólido y determinista:** reglas estructurales del código del reverso (número frente = código,
  sexo, producción ≥ expedición y ≥ 18 años, formato 1-7-8-M/F-10-8, meses de 3 letras);
  **registrador firmante vs. fecha de producción**; reverso repetido entre personas distintas con
  la excepción "mismo lote" (mismo consecutivo y misma fecha con series cercanas < 50.000 y finales
  distintos es normal); PDF417 vs. impreso y plantilla de huella repetida (si el código se lee);
  dígitos de control ICAO de la MRZ en la cédula digital.
- **Tabla de registradores** (coincide con el Excel y la prensa): Iván Duque Escobar 6-dic-1999 →
  24-ago-2002; Alma Beatriz Rengifo 25-ago-2002 → dic-2006; Juan Carlos Galindo 1-ene-2007 →
  dic-2007 (primer periodo, confirmado); Carlos Ariel Sánchez 6-dic-2007 → dic-2015; Galindo
  4-dic-2015 → dic-2019; Alexander Vega Rocha 6-dic-2019 → dic-2023; Hernán Penagos desde dic-2023.
  Único punto en duda: el Excel y el catálogo cierran a Vega el 21-nov-2023, pero la prensa dice
  que Penagos asumió funciones el 6-dic-2023; la quincena 21-nov a 6-dic-2023 se trata como zona
  gris que no penaliza. Los registradores anteriores a 1999 del Excel no importan (cédulas
  anteriores a la amarilla perdieron validez en 2010).
- **En duda, excluido:** línea de tiempo del consecutivo y la serie (33 y 18 anclas, márgenes
  arbitrarios ya corregidos varias veces); variantes de plantilla por época (tildes, posición de
  "NUMERO") inferidas de pocos casos.
- **No se sostiene como regla automática, excluido:** juicios visuales del LLM (foto pegada,
  huella empastada, holograma plano, firma con lapicero, impresión casera), estimación de edad por
  rostro (±5–8 años, ajustada tras falsos positivos), reglas de datos de contacto, pesos y
  umbrales del catálogo. Lo visual le corresponde a la CNN.
- **Firmas del Excel:** no se compara la imagen de la firma; el nombre va impreso debajo y el OCR
  lo lee. En las composiciones firma y nombre vienen juntos de la plantilla; lo que falla es la
  fecha, y eso lo cubre la regla de registrador.
- **Privacidad:** el artifact publicado contiene números de cédula y nombres reales; debería
  anonimizarse antes de compartirlo.
- **Valor principal:** la taxonomía de modalidades para clasificar los positivos, y la posibilidad
  de validar cada regla contra los 28.000 negativos (falsos positivos) y 399 positivos (detección),
  con el mismo método de `docs/fraud-score/01-problema.html`. Alejandro pidió **incluir solo lo
  valioso y excluir lo dudoso**; así quedó en `docs/plan.md` §3.1b.

### 4.9 Decisiones de organización del código
- Proyecto **aparte**, no dentro de `mikro_back`; carpeta en "Proyecto de grado", que Alejandro
  convertirá en repositorio. Sin Django; conexión directa a PostgreSQL (solo lectura) y S3.
- **Código en inglés**, documentación en español.
- Nombres con `_`, no con `-` (la carpeta pasó de `detector_fraude_cedulas` → `id-fraud-detector`
  → `id_fraud_detector`). Puede quedar una carpeta vacía `detector_fraude_cedulas` por borrar.

## 5. Lista explícita de lo descartado y por qué (para no volver a proponerlo)

| Descartado | Razón |
|---|---|
| Clasificador end-to-end único sobre la foto completa a 224 px | Pierde señales locales finas; con pocos positivos aprende el fondo, el celular o la fecha. Se conserva solo como **línea base** para comparar. |
| Varios modelos especializados + fusor | Con la regla de etiquetado los positivos son una sola modalidad; un enfoque supervisado basta. Parte de la propuesta inicial era cobertura de temas de clase. |
| Autoencoder / detección de anomalías | En fotos reales el error de reconstrucción mide desenfoque y luz, no falsedad. El fraude documentado no es recaptura. |
| ELA (Error Level Analysis) | La captura es PNG generado en canvas; no hay cuantización JPEG que analizar. |
| YOLO como primer paso | El encuadre de la app da un prior fuerte; CV clásico primero. YOLO solo si recorta bien < 90 %. |
| Cajas de `detect_text` para recortar | Encierran palabras, no el plástico; no dan esquinas. |
| Comparación de imagen de firma del registrador | El nombre va impreso y el OCR lo lee; la señal está en la fecha. |
| Línea de tiempo consecutivo/serie, variantes de plantilla por época | Pocos puntos de referencia, excepciones abiertas. |
| Juicios visuales de un LLM, estimación de edad, reglas de contacto, pesos del catálogo | No reproducibles o fuera del alcance de imagen; los pesos se miden. |
| Incluir los ~170 POSSIBLE FRAUD posteriores al 4-oct-2026 | No confirmados; mezcla de red y falsos positivos de la política. Decisión de Alejandro. |
| Cambiar la marca de los 65 a FRAUD en la base | Decisión de Alejandro: la base no se toca; se incluyen por ventana de fechas en config. |
| Positivos sintéticos como base | Con ~399 positivos reales no hacen falta para arrancar; quedan para iteración si una modalidad es escasa. |

## 6. Lo que SÍ quedó (resumen ejecutable)

1. Dataset: `scripts/11_export_dataset.py` con las reglas de `config.yaml`; manifest anonimizado,
   OCR por cliente, imágenes de todos los positivos y de 4.000 negativos muestreados (todos los
   negativos entran con metadatos y OCR para medir falsos positivos).
2. Reglas: `src/id_fraud_detector/rules/` + `scripts/12_ocr_coverage.py` → legibilidad por campo y tabla de
   falsos positivos / detección por regla. Primer resultado publicable de la tesis.
3. Recorte: `src/id_fraud_detector/crop.py` + `scripts/20_crop_test.py` → hojas de contacto para decidir si hace
   falta YOLO.
4. Revisión visual: `scripts/13_visual_review.py` (Streamlit) → modalidad de cada positivo
   (cloned_back, assembled_back, replaced_photo, pasted_photo_template,
   screen_or_print_recapture, genuine_doc_stolen_identity, other, unsure). Los
   `genuine_doc_stolen_identity` pasan a negativos difíciles.
5. Modelo (fase 3, no empezada): transfer learning frente y reverso a 384–512 px, aumentos
   conservadores (sin flips), `class_weight`, partición por cliente y tiempo (entrenar 2025,
   validar 2026), **Llévalo Ya como prueba externa**, evaluación de FPR contra todos los negativos,
   métrica principal recall a FPR ≤ 2 %, Grad-CAM, línea base end-to-end a 224 px.
6. Indicador (fase 5): calibración 0–100, umbrales revisión/retención, backtest histórico con el
   método de `05-backtest-esmio.html`.
7. Integración (fase 6, opcional): campo `data_doc_risk` en `CustomerIdValidationResult`, regla
   `FV_DOC_IMAGE_RISK` en SHADOW.

## 7. Pendientes al 9-nov-2026

- Usuario de solo lectura a `sisteya_live` y credenciales de S3 para correr `11_export_dataset.py`
  (primero con `--no-images` y comparar conteos con la tabla de §4.4).
- Confirmar con la Registraduría el día exacto del relevo Vega → Penagos (21-nov vs 6-dic-2023).
- Decidir dónde se entrena (local con GPU, AWS de la empresa, Colab Pro con datos cifrados).
- Revisión visual de los ~399 positivos con `13_visual_review.py` (Alejandro o el compañero de
  verificación).
- Borrar la carpeta vacía `detector_fraude_cedulas` en "Proyecto de grado".
- Sincronizar `docs/plan.md` con el documento del proyecto en Claude.ai cuando cambie.

## 8. Cambios tras la migración a Claude Code (11-nov-2026)

Revisión del código y de `mikro_back` (solo lectura) ya en VS Code. Nada de esto se ha corrido
todavía contra la base real.

**Preferencia de trabajo de Alejandro (cambia lo dicho en §4.3):** no quiere correr consultas a
mano en pgAdmin. Las consultas van dentro de scripts de solo lectura que él ejecuta; por eso se
creó `scripts/10_check_database.py`.

**Conexión (decisión de Alejandro, 11-nov):** sin llaves de acceso ni contraseñas en `.env`. Los
scripts usan su sesión de SSO (`aws sso login`), leen las credenciales de la base del secreto
`sisteya_db_proxy_credentials` de Secrets Manager (el mismo de `mikro_back`) y se conectan a
`sisteya_<STAGE>`, con `STAGE` definido en `.env`. Las salidas quedan en `data/<STAGE>/` para que
los datos de `dev` no se mezclen con los de `live`. El usuario del secreto es el de la
aplicación (con escritura); la protección es la sesión de solo lectura, que se verifica al abrir
cada conexión.

**Errores corregidos**
- El OCR está en `data_back_id["detected_text"]["TextDetections"]`; el parser lo buscaba un nivel
  más arriba y todas las reglas habrían salido `NOT_EVALUABLE` sin ningún error.
- Cuatro casos en los que un OCR roto daba `FAIL`: fechas entregadas en otro orden
  (`R_PRODUCTION`), mes con una letra mal leída (`R_MONTHS`), lugar de nacimiento "LA VEGA" con el
  registrador ilegible (`R_REGISTRAR`) y número del frente partido por el OCR (`R_NUMBER`).
  Detalle en `docs/plan.md` §3.1b, "Precisiones de implementación".
- Tres pérdidas de cobertura: `l` leída por `1`, serial con letra distinta de A, y cédulas de 19 y
  20 millones descartadas por parecer fechas.
- La exportación traía el JSON completo de Rekognition de todos los clientes en una sola consulta
  (varios GB; no cabe en memoria). Ahora la base devuelve solo las líneas de texto con su
  confianza, en lotes de 300 clientes, y la exportación se puede retomar.
- Los valores leídos de las cédulas (números, series) ya no quedan en `data/rules_per_customer.csv`
  sino en `data/private/rules_values.csv`.

**Hallazgos en `mikro_back` (pendientes de medir y decidir; `docs/plan.md` §5.4 punto 3)**
- `users_customeridvalidationresult` es 1:1 con el cliente (`OneToOneField`) y su `timestamp` es
  `auto_now`: es la última modificación, no la fecha de la validación.
- `validate_from_common_client` copia las fotos entre marcas sin volver a pasar por Rekognition.
- `Client.get_fraud_status()` lee primero la marca del registro común (`CommonClient`, esquema
  `public`) y después la del tenant.
- Un crédito saldado con un pago BNPL queda `NOVATED`, no `PAID`.
- La regla de serial compartido ya está en producción como FR01
  (`users/utils/id_back_serial_utils.py`).

**Primera corrida real (11-nov, `STAGE=dev`, tenant `demo`).** Las verificaciones previas
terminaron sin error: conexión por SSO y Secrets Manager, sesión de solo lectura y las tres
consultas (clientes, OCR, registro común). Conteos de `demo` desde 2025-01-01: 46 clientes, 4
negativos (los 4 con ambas fotos), 2 con otro estado, ningún positivo; texto del reverso legible
en 3 de los 4 negativos, lo que confirma la ruta `detected_text` → `TextDetections`; 1 de los 4
negativos tiene un crédito `NOVATED`. Es una base de pruebas: valida que el código corre, no
dice nada sobre el dataset.

**Organización del repositorio (decisión de Alejandro, 11-nov).** Preguntó si convenía planear
todo el proyecto en el mismo repositorio, incluidos el clasificador y la infraestructura. Quedó
que sí: un solo repositorio, con la lógica en un paquete instalable (`src/id_fraud_detector/`,
`pip install -e ".[review]"`), dependencias por grupo en `pyproject.toml` (reemplaza a
`requirements.txt`) y scripts numerados por fase. Por eso los scripts cambiaron de nombre y este
documento ya usa los nuevos: `00_check_database` → `10_check_database`, `01_export_dataset` →
`11_export_dataset`, `02_ocr_coverage` → `12_ocr_coverage`, `04_visual_review` →
`13_visual_review`, `03_crop_test` → `20_crop_test`. Las carpetas de fases futuras se crean cuando
empiece cada fase. Estructura completa en `docs/plan.md` §6b. También se inició el repositorio
git (primer commit local, sin remoto).
