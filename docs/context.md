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

## 9. Primera corrida en `live` (11-nov-2026, `10_check_database.py`)

Solo verificaciones; no se exportó nada. Conteos en `data/live/checks/` (agregados).

**Conteos (clientes registrados desde 2025-01-01, con ambas fotos entre paréntesis)**

| Tenant | Positivos FRAUD | Ventana POSSIBLE FRAUD | Negativos |
|---|---|---|---|
| esmio | 299 (284) | 79 con ambas fotos | 16.086 (16.083) |
| llevaloya | 45 (40) | — | 12.149 (12.147) |

Coinciden con los del 9-nov salvo la deriva normal. Positivos con ambas fotos: 284 + 79 + 40 = 403.

**Positivos de ESMIO por año de registro:** 202 en 2025 y 82 en 2026 (FRAUD), más los 79 de la
ventana (52 en septiembre y 27 en octubre de 2026). Con "entrenar 2025, validar 2026" quedan 202
para entrenar y 161 para validar, de los cuales 79 son de una sola red.

**Negativos recientes escasos:** en ESMIO hay 117 negativos registrados en septiembre de 2026, 43
en octubre y 3 en noviembre, frente a 79 + 6 positivos en septiembre–octubre. Una muestra de
negativos proporcional al mes dejaría unos 25 negativos de ese periodo.

**Canal:** en ESMIO el 27 % de los positivos viene de la app (99 de 363) frente al 5 % de los
negativos (870 de 16.083); el resto es comercio y WhatsApp. En Llévalo Ya, 39 de 40 positivos y
el 97 % de los negativos son de comercio. El canal es un confusor para el clasificador.

**OCR guardado (muestra de 300 por etiqueta; todos tienen fila de validación):**

| Tenant | Etiqueta | Con texto del reverso |
|---|---|---|
| esmio | positivos | 114 de 300 (38 %) |
| esmio | negativos | 49 de 300 (16 %) |
| llevaloya | positivos | 2 de 40 (5 %) |
| llevaloya | negativos | 35 de 300 (12 %) |

Es el hallazgo que más afecta al plan: las reglas sobre el OCR guardado solo podrían evaluarse en
una fracción de los clientes.

**Segunda corrida (mismo día), con el OCR desglosado. La causa es el canal, no las fotos copiadas.**
La hipótesis inicial (clientes ya validados en otra marca, a los que `validate_from_common_client`
les copia las fotos sin pasar por Rekognition) quedó descartada como causa principal: solo el 2 %
de los negativos de ESMIO (343) y el 3,6 % de los de Llévalo Ya (439) están en ese caso. Esos sí
tienen 0 % de OCR, pero son pocos.

| Tenant | Etiqueta | APP | WHATSAPP | COMMERCE |
|---|---|---|---|---|
| esmio | positivos | 94 de 99 (95 %) | 7 de 9 | 32 de 255 (13 %) |
| esmio | negativos (muestra) | 21 de 26 (81 %) | 7 de 7 | 26 de 267 (10 %) |
| llevaloya | positivos | 1 de 1 | — | 1 de 39 (3 %) |
| llevaloya | negativos (muestra) | 2 de 6 | 8 de 8 | 25 de 286 (9 %) |

- En el flujo de comercio solo ~10 % de los clientes tiene texto guardado, y comercio es el 94 % de
  los negativos y el 73 % de los positivos. **No se sabe por qué**: el endpoint de comercio
  (`check_frontal_id_picture` / `check_back_id_picture`) sí llama a `detect_text` y lo guarda.
  Pendiente de que Alejandro lo revise en `mikro_back`. Consecuencia fuera de la tesis: la regla
  FR01 de serial compartido lee ese mismo texto, así que tampoco puede evaluarse donde falta.
- Por origen de la etiqueta en ESMIO: los 79 de la ventana tienen texto (100 %); de los 284 FRAUD,
  54 (19 %). En total, 135 de los 403 positivos tienen texto del reverso.
- Negativos con texto: del orden de 4.000 entre las dos marcas (estimado a partir de la muestra).

**Personas validadas antes en otra marca (todas las etiquetadas):**

| Tenant | Etiqueta | Clientes | Validada antes en otra marca | De esas, antes de 2025 |
|---|---|---|---|---|
| esmio | negativos | 16.083 | 343 (2,1 %) | 107 |
| esmio | positivos | 363 | 0 | 0 |
| llevaloya | negativos | 12.148 | 439 (3,6 %) | 142 |
| llevaloya | positivos | 40 | 21 (52,5 %) | 0 |

21 de los 40 positivos de Llévalo Ya no tienen fotos propias: son copias de las de una validación
anterior en otra marca (casi con seguridad ESMIO, donde 22 de ellos son positivos). Solo 249
negativos y ningún positivo tienen fotos anteriores a 2025.

**Acceso a S3:** verificado en las dos marcas.

**Texto tomado de la otra marca (decisión de Alejandro, 11-nov).** Alejandro buscó en pgAdmin un
caso de comercio sin OCR en ESMIO y vio que el cliente se había validado en Llévalo Ya: ESMIO
tomó la validación del registro común y solo copió las fotos. Pidió que, en esos casos, la
exportación busque el texto entre ESMIO y Llévalo Ya, **solo si las imágenes son iguales en las
dos marcas**. Quedó así en `11_export_dataset.py` (`src/id_fraud_detector/ocr_store.py`): para
cada cliente sin texto se busca el mismo número de documento en la otra marca, sin filtro de
fecha de registro; si allá hay texto y la imagen de esa cara tiene la misma huella (ETag) y el
mismo tamaño en S3, se usa ese texto. Se decide por cara y el manifest guarda el origen
(`front_ocr_from`, `back_ocr_from`). No se llama a Rekognition ni se descarga la imagen.

Ese caso también muestra que la columna `validated_elsewhere_first` se queda corta: solo marca a
quien se validó en otra marca **antes de registrarse** aquí. Quien se registró aquí primero, se
validó después en la otra marca y luego volvió, no queda marcado. El número real de textos
recuperados se midió en la exportación (§10): son pocos. La búsqueda solo cubre ESMIO y Llévalo
Ya: si la persona se validó primero en una tercera marca, su texto no se recupera.

**Misma persona en las dos marcas:**
- 22 de los 40 positivos de Llévalo Ya son también positivos de ESMIO (24 de los 40 existen en
  ESMIO con fotos). La "prueba externa nunca vista" solo tiene 18 positivos que no están en ESMIO.
- 399 personas son negativos en las dos marcas.
- Además, 37 de los 40 positivos de Llévalo Ya son de 2025 (sobre todo abril–junio), el mismo
  periodo de la primera ola de ESMIO.

**Marca de fraude en el registro común:** coincide con la del tenant. Solo 5 negativos (3 en
ESMIO, 2 en Llévalo Ya) tienen alguna marca en el registro común, ninguna FRAUD ni POSSIBLE FRAUD.

**Créditos `NOVATED`:** solo 1 positivo tiene un crédito `NOVATED`; 287 negativos lo tienen, pero
ya son negativos por tener además un crédito `PAID`. Cambiar la regla no movería el dataset.

**Pendiente de decisión de Alejandro (nada de esto se ha aplicado):**
1. Qué hacer con la baja cobertura del OCR: medir las reglas solo donde hay texto o volver a pasar
   OCR sobre las fotos. Recuperar el texto de la otra marca ya se hizo y aporta poco (§10).
   **Decidido por Alejandro el 11-nov-2026:** volver a pasar Rekognition sobre las fotos de todos
   los clientes, tengan o no texto guardado, para que el texto corresponda con seguridad a la
   imagen del cliente. El consumo no puede ir a la cuenta corporativa: lo paga él con otra cuenta
   de AWS, separada de la sesión de SSO que se usa para la base y S3. Estimado: unos 57 USD por
   `detect_text` sobre las dos caras de los 28.637 clientes (0,001 USD por imagen). Falta
   confirmar el alcance y el tipo de cuenta; todavía no hay código.
2. Cómo evaluar, dado que Llévalo Ya no es independiente de ESMIO: asignar cada persona a una
   sola partición es obligatorio; falta decidir si Llévalo Ya sigue siendo la prueba externa.
3. Muestra de negativos con imágenes: emparejarla con los positivos por canal y periodo en vez de
   hacerla proporcional.
4. Si las fotos copiadas de validaciones anteriores a 2025 siguen en el dataset.

## 10. Exportación sin imágenes en `live` (11-nov-2026, `11_export_dataset.py --no-images`)

Terminó sin error en unos dos minutos. Existen `data/live/manifest.csv` (28.637 filas), un archivo
de OCR por cliente en `data/live/ocr/` y `data/live/private/id_map.csv`. No se descargó ninguna
imagen.

**Etiquetados:** ESMIO 363 positivos (284 FRAUD + 79 de la ventana) y 16.085 negativos; Llévalo Ya
40 positivos y 12.149 negativos. Con la configuración actual, la muestra para imágenes sería de
2.866 clientes en ESMIO y 1.549 en Llévalo Ya (pendiente la decisión 3 del §9).

**Texto tomado de la otra marca:**

| | ESMIO (busca en Llévalo Ya) | Llévalo Ya (busca en ESMIO) |
|---|---|---|
| Clientes sin texto en alguna cara | 13.860 de 16.448 (84 %) | 10.905 de 12.189 (89 %) |
| De esos, existen en la otra marca | 898 (6,5 %) | 937 (8,6 %) |
| Frente: la otra marca tiene texto | 138 | 197 |
| Frente: imagen idéntica, texto tomado | 136 | 192 |
| Frente: imagen distinta, no se toma | 2 | 5 |
| Reverso: la otra marca tiene texto | 134 | 187 |
| Reverso: imagen idéntica, texto tomado | 132 | 184 |
| Reverso: imagen distinta, no se toma | 2 | 3 |

**Texto del reverso disponible al final (propio + tomado):**

| Tenant | Etiqueta | Clientes | Con texto del reverso | De esos, tomados de la otra marca |
|---|---|---|---|---|
| esmio | negativos | 16.085 | 2.587 (16 %) | 132 |
| esmio | positivos | 363 | 133 (37 %) | 0 |
| llevaloya | negativos | 12.149 | 1.464 (12 %) | 182 |
| llevaloya | positivos | 40 | 4 (10 %) | 2 |

En total: 137 de 403 positivos (34 %) y 4.051 de 28.234 negativos (14 %).

**Lectura:**
- La comparación por huella funciona: cuando la otra marca tiene texto, la imagen es idéntica en
  el 98 % de los casos (328 de 335 frentes, 316 de 321 reversos). No hace falta descargar y
  comparar contenido.
- El caso que encontró Alejandro es real pero raro. Se recuperaron 316 reversos de unos 24.700
  que faltaban (1,3 %). Solo el 7 % de los clientes sin texto existe en la otra marca, y de esos
  cerca del 80 % tampoco tiene texto allá.
- Los positivos casi no cambian: de 135 a 137 con texto del reverso. Los positivos de Llévalo Ya
  solo recuperaron 2 textos, aunque 21 tienen fotos copiadas; lo más probable es que la copia de
  ESMIO tampoco tenga texto (solo el 19 % de los FRAUD de ESMIO lo tiene). No está verificado.
- La causa principal sigue siendo que el flujo de comercio no guarda el texto, en ninguna de las
  dos marcas. Sigue sin saberse por qué.
- De los 137 positivos con texto, 79 son de la red de la tienda 623. La tasa de detección de las
  reglas hay que leerla separada por origen de la etiqueta, no solo en conjunto.

## 11. Decisiones de Alejandro tras la exportación (11-nov-2026)

1. **Volver a pasar Rekognition por todas las fotos**, tengan o no texto guardado (detalle y
   costo en el §9, decisión 1). El consumo lo paga Alejandro con una cuenta de AWS distinta de la
   corporativa: es una cuenta suya, antigua, que paga él (la abrió como educativa; no es un
   laboratorio de AWS Academy). Estimado: `detect_text` sobre dos caras cuesta 0,001 USD por
   imagen, unos 56 USD para toda la lista. Alcance con el que quedó el script, por confirmar por
   Alejandro: solo texto (las reglas no usan etiquetas ni rostro), frente y reverso, todas las
   personas. Alejandro confirmó el mismo día que tiene la autorización de la empresa para
   procesar las fotos en esa cuenta. Antes de la corrida completa debe activar en su cuenta la
   exclusión de servicios de IA (AWS Organizations, "Opt out from all services"), para que AWS
   no use las imágenes para mejorar sus servicios.

   Quedó en `scripts/14_fresh_ocr.py` (`src/id_fraud_detector/fresh_ocr.py`): lee cada foto de
   S3 a memoria con la sesión de SSO, la envía a Rekognition con el perfil `REKOGNITION_PROFILE`
   y guarda en `data/<STAGE>/ocr_fresh/` todas las detecciones tal como llegan (líneas y
   palabras, con confianza y posición), más el MD5 y el tamaño de la imagen que produjo el
   texto. Sin `--run` solo informa lo pendiente y el costo. Se
   niega a correr si el perfil de Rekognition es de la misma cuenta que la sesión de SSO. Es
   reanudable y no vuelve a cobrar lo ya leído. Las rutas de las fotos salen de
   `private/id_map.csv`, que ahora las incluye. `12_ocr_coverage.py` usa este texto por defecto y
   el guardado con `--ocr stored`. Probado con S3 y Rekognition simulados; no se ha corrido.

   A Alejandro le preocupaba tener que volver a descargar las fotos si algo falla o hay que
   repetir. Por eso el script tiene `--keep-images`: guarda cada foto descargada en
   `data/<STAGE>/images/`, la misma ruta de `11_export_dataset.py`, y todos los scripts leen del
   disco antes de ir a S3. Sin la opción, las fotos pasan por memoria de a 8 personas y no se
   guardan. Falta que decida si la usa en la corrida completa: la prueba de 20 personas imprime
   el tamaño promedio y cuánto ocuparían todas en disco.

   El 12-nov-2026 Alejandro pidió que no se pierda nada de lo que se paga: el archivo guarda la
   lista completa de detecciones de Rekognition (líneas y palabras con su geometría), no solo
   las líneas. Las reglas siguen leyendo solo las líneas. `ocr_fresh/` es la única copia de ese
   texto (`data/` no se versiona): al terminar la corrida conviene copiarla a otro disco
   cifrado, nunca a una nube pública.

   El mismo día pidió que el script falle si no ha dicho con qué cuenta corren los servicios con
   costo, sin asumir nada, para no usar la corporativa por accidente. El perfil ya era
   obligatorio; ahora también lo es el número de la cuenta (`REKOGNITION_ACCOUNT_ID`), y el
   script comprueba antes de cualquier otra cosa que las llaves del perfil sean de esa cuenta y
   que no sea la de la sesión de SSO. Esto cierra un caso que la comparación sola no cubría: un
   perfil de otra cuenta de la empresa, distinta de la de la sesión. Lo que sí sigue cobrándose
   a la cuenta corporativa es la descarga de las fotos desde su S3 (transferencia de salida), y
   eso no se puede mover sin cambiar el bucket.
2. **Una fila por persona, no por cliente de cada marca.** Los etiquetados de ESMIO y Llévalo Ya
   se cruzan por número de documento para que nadie quede repetido. Con los conteos del §9
   quedarían unas 381 personas positivas (363 + 40 − 22) y unas 27.835 negativas
   (16.085 + 12.149 − 399), unas 28.200 en total. No hay ninguna persona positiva en una marca y
   negativa en la otra.
3. **Las fotos son las de cada marca, no las del registro común.** Alejandro propuso primero
   tomarlas de `public.customers_commonclient` y lo descartó el mismo día: `mikro_back` guarda
   esas fotos una sola vez, cuando la persona completa una validación
   (`Client.check_user_validation`). Quien subió las fotos pero no terminó la validación, o a
   quien se le rechazó, no tiene fotos allí, y eso es más probable justo en los positivos.
   Además, la primera validación pudo ser en una tercera marca o antes de 2025.

Hasta estas decisiones la exportación tenía una fila por cliente de cada marca (28.637), con
`person_id` igual entre marcas.

**Cómo quedó el cruce** (`dataset.one_row_per_person`, usado por `11_export_dataset.py` y por la
sección 12 de `10_check_database.py`): se juntan los etiquetados de todas las marcas y, por cada
número de documento, queda una sola fila. Orden de preferencia: la positiva antes que la
negativa, la de registro más antiguo (la marca donde probablemente se tomaron las fotos), el
orden de las marcas en `config.yaml` y el id más bajo. Las fotos y los metadatos (canal, fecha de
registro) son los de la fila que queda. El manifest guarda `labeled_in` (marcas donde la persona
está etiquetada) y `label_conflict`. La opción `--tenant` de la exportación se quitó, porque el
cruce necesita todas las marcas. Los archivos de OCR de las filas que ya no están en el manifest
quedan en disco sin uso.

El cruce también cubre un caso que antes no estaba controlado: si dos clientes de la misma marca
tuvieran el mismo documento, compartirían `anon_id` y uno pisaría el archivo de OCR del otro. No
se sabe si ocurre en `live`.

Probado con datos sintéticos y con la base y el S3 simulados; **pendiente** correr de nuevo
`11_export_dataset.py --no-images` en `live` para regenerar el manifest por persona. (Hecho el
12-nov-2026; ver §12.)

## 12. Prueba del OCR nuevo con 20 personas en `live` (12-nov-2026)

Alejandro regeneró el manifest por persona y corrió `14_fresh_ocr.py --run --limit 20`.

**Manifest por persona:** 28.253 personas (56.506 fotos).

| Marca de la fila | Positivos | Negativos |
|---|---|---|
| ESMIO | 362 | 15.913 |
| Llévalo Ya | 19 | 11.959 |
| Total | 381 | 27.872 |

**La prueba:** las dos cuentas quedaron separadas (Rekognition en la de Alejandro, base y S3 en
la corporativa). Se leyeron 40 fotos de 20 positivos de ESMIO, sin errores y todas con texto,
por 0,04 USD. Medido:

- Velocidad: 1,39 personas por segundo con 8 hilos; a ese ritmo la lista completa tarda unas
  5,6 horas.
- Tamaño promedio de foto: 2,13 MB. Todas las fotos en disco serían unos 120 GB, y esa misma
  cantidad sale del S3 corporativo (transferencia de salida a cargo de la empresa).
- Archivo de texto por persona: 38,6 KB en promedio, cerca de 1,1 GB para toda la lista.
- Ninguna foto superó los 5 MB; todas se enviaron sin recodificar. Modelo de texto 3.0.

**Lo que se lee del reverso en esas 20 personas** (conteos sacados de los archivos locales, sin
mostrar texto):

| | De 20 |
|---|---|
| Rekognition devolvió una línea con forma de código | 13 |
| El código se puede interpretar sin filtrar por confianza | 8 |
| El código se puede interpretar con el filtro actual (confianza ≥ 80) | 4 |
| Registrador reconocido (confianza ≥ 80 / sin filtro) | 6 / 7 |
| Sexo impreso leído | 0 |
| Alguna fecha impresa leída | 18 |
| Número leído en el frente | 19 |

Lectura, con las reservas de que son solo 20 personas, todas positivas y de una marca:

- En 7 de 20 Rekognition no devolvió el código en absoluto. No se sabe si es por el tamaño del
  texto dentro del cuadro completo de la cámara, por la calidad de la foto o porque son cédulas
  falsas.
- La línea del código llega muchas veces con confianza baja (entre 12 y 78 en 8 de las 13
  líneas). El filtro de 80 descarta lecturas que sí tienen la forma exacta del código. Bajarlo
  solo para el código es un cambio del motor de reglas, sin costo, que se puede decidir con los
  datos completos porque el archivo guarda todas las líneas.
- El sexo impreso no se leyó en ninguna, aunque la palabra "SEXO" aparece en 15: el patrón que
  lo busca no encaja con la forma en que Rekognition entrega esa parte. Por revisar.
- Ninguna de las 20 es cédula digital (sin líneas de lectura mecánica).

**Etapa intermedia (aprobada por Alejandro el 12-nov-2026):** antes de gastar los ~56 USD hay
que saber cuánto se lee el código en cédulas legítimas y si enviar la cédula recortada lo
mejora. Se leen todos los positivos y 300 negativos, guardando las fotos (~1,3 USD, unos 3 GB),
se mide con `12_ocr_coverage.py` y, sobre esas mismas fotos, se compara foto completa contra
cédula recortada. La corrida completa queda en pausa hasta ver esos números.

- Los 300 negativos salen al azar, con la semilla de `config.yaml`, de los 4.017 negativos que
  el manifest ya marca para imágenes (`has_images`: 2.505 de ESMIO y 1.512 de Llévalo Ya, la
  muestra estratificada por mes de registro y canal, proporcional al tamaño de cada grupo). Así
  las fotos guardadas sirven después para el clasificador si esa muestra se mantiene. Como la
  muestra es proporcional, casi todos serán de comercio y habrá unos 10 de app: no alcanza para
  comparar canales en negativos.
- Quedó en el código: `14_fresh_ocr.py --negatives N` (corrida parcial) y `12_ocr_coverage.py`
  ahora mide solo a las personas ya leídas y agrega la legibilidad por canal.
- Falta escribir la comparación contra la cédula recortada; depende de ver primero cómo recorta
  `20_crop_test.py` sobre fotos reales, que tampoco se ha corrido.
- La decisión 3 del §9 (muestra de negativos para el clasificador: proporcional o emparejada
  con los positivos por canal y periodo) sigue abierta. Tampoco está decidido si la corrida
  completa guarda las fotos (120 GB).

## 13. Etapa intermedia en `live`: 381 positivos y 300 negativos (12-nov-2026)

Alejandro corrió `14_fresh_ocr.py --run --negatives 300 --keep-images` y `12_ocr_coverage.py`.
Se leyeron 1.322 fotos sin errores (1,32 USD), a 2,08 personas por segundo (la lista completa
tardaría unas 3,7 horas). Foto promedio de 1,98 MB (todas serían unos 112 GB). Quedaron en
disco las fotos de 661 personas; las de los 20 positivos de la primera prueba no se guardaron.

**Lo que se lee, por canal** (con el filtro de confianza ≥ 80, salida de `12_ocr_coverage.py`):

| Etiqueta | Canal | n | Número del frente | Código del reverso | Serie | Registrador |
|---|---|---|---|---|---|---|
| Negativo | App | 10 | 100 % | 60 % | 70 % | 80 % |
| Negativo | Comercio | 288 | 80 % | 12 % | 25 % | 25 % |
| Positivo | App | 100 | 100 % | 60 % | 43 % | 60 % |
| Positivo | Comercio | 272 | 82 % | 21 % | 36 % | 26 % |
| Positivo | WhatsApp | 9 | 89 % | 0 % | 0 % | 0 % |

El sexo impreso no se lee en nadie: Rekognition entrega la palabra "SEXO" como una línea aparte
y el valor en otra, así que el patrón actual nunca encaja. Arreglarlo exige usar la posición de
las líneas (solo disponible en el OCR nuevo). Pendiente.

**Por qué se lee tan poco en comercio** (conteos sacados de los archivos y fotos locales):

- La causa es la resolución de la foto. En la app casi todas son de 1920×1210; en comercio la
  mayoría es de 1280×720 y hay un grupo de 640×480. El código se interpreta con confianza ≥ 80
  en el 46 % de las fotos de 1401 a 2000 px de ancho y en el 8 % de las de 1400 px o menos.
- No es que la cédula quede pequeña dentro del cuadro: el texto ocupa cerca del 75 % del ancho
  de la foto en todos los canales. Recortar, por sí solo, aporta poco.
- En comercio Rekognition devuelve una línea con forma de código en el 69 % de las fotos, pero
  solo se interpreta en el 27 % (negativos) y el 39 % (positivos) sin filtrar por confianza, y
  en el 12 % y el 21 % con el filtro.
- El filtro de confianza está justificado: en los códigos leídos con confianza ≥ 80 el número
  del código coincide con el del frente en 152 de 152; en los que solo se leen por debajo de 80,
  en el 75 % (77 de 102). Bajar el filtro sin más produciría fallos falsos.

**Reglas** (381 positivos, 300 negativos):

| | Positivos | Negativos |
|---|---|---|
| Con alguna regla del código evaluable | 156 (41 %) | 69 (23 %) |
| Con al menos una regla en `FAIL` | 38 (10 %) | 0 |
| `R_REGISTRAR` en `FAIL` / evaluables | 13 / 127 | 0 / 64 |
| `R_REPEATED_BACK` en `FAIL` / evaluables | 28 / 121 | 0 / 40 |

Las demás reglas no fallan en nadie. Por origen de la etiqueta: 24 de los 79 de la red de la
tienda 623 tienen algún `FAIL` (30 %) y 14 de los 302 `FRAUD` restantes (5 %). Por canal: 26 de
100 positivos de app y 12 de 272 de comercio.

Lectura:

- Ningún falso positivo, pero sobre muy pocos negativos evaluables: con 0 de 69 la tasa real
  puede ser de hasta un 4 % (regla de tres). No alcanza para fijar pesos.
- Las reglas detectan sobre todo a la red de la tienda 623. Fuera de ella detectan el 5 % de
  los positivos. El resto del trabajo recae en el clasificador de imágenes.
- `R_REPEATED_BACK` compara entre las personas leídas (681). Con más personas puede encontrar
  más repeticiones, también entre negativos.
- La cobertura de las reglas está limitada por la legibilidad, y la legibilidad por la cámara
  del canal de comercio. Con la corrida completa tal como está, el código quedaría legible en
  cerca del 12 % de los negativos de comercio.

**De dónde sale la resolución de comercio** (leído en `mikro_front`, sin modificarlo): los dos
flujos piden a la cámara 1920 px de ancho (`StepBackIdPicture.js` en la app, `BackIdModal.js` en
comercio) y capturan el cuadro al tamaño real del video. La petición es una preferencia, no una
exigencia: el navegador entrega lo que la cámara soporta. En comercio el equipo es el de la
tienda, y por eso salen 1280×720 y 640×480. Además, el flujo de comercio elige por defecto la
primera cámara que lista el dispositivo; no está verificado si eso hace que en algunos equipos
se use una cámara peor que la disponible. El problema seguirá existiendo en producción mientras
las tiendas usen esas cámaras.

Alejandro preguntó (12-nov-2026) si, dado eso, conviene no usar el texto en el producto final.
Recomendación de Claude, pendiente de su decisión: mantener las reglas como evidencia adicional
(solo suman cuando fallan, no han dado falsos positivos y explican el motivo al analista), pero
tratar al clasificador como el componente principal, no invertir más en reglas y limitar la
lectura de negativos a los 4.017 marcados para imágenes.

**Siguiente prueba (escrita, sin correr):** `scripts/15_enlarged_ocr_test.py` toma 200 personas
al azar con la foto del reverso en disco, recorta la zona donde Rekognition encontró texto, la
amplía al doble y la vuelve a enviar (unos 0,20 USD; no lee nada de S3). Compara cuántos códigos
se interpretan con la foto completa y con la cédula ampliada. Decide si la corrida completa se
envía ampliada. Probado con Rekognition simulado.

**Resultado de la prueba de ampliación (12-nov-2026, 200 personas, 0,20 USD):** ampliar no
mejora la lectura.

| | Foto completa | Cédula ampliada |
|---|---|---|
| Código legible, app (41) | 56,1 % | 58,5 % |
| Código legible, comercio (159) | 13,8 % | 14,5 % |
| Código legible, todos (200) | 22,5 % | 23,5 % |
| Serie | 28,0 % | 31,5 % |
| Registrador | 26,5 % | 26,0 % |
| Número del código igual al del frente | 43 de 43 | 43 de 45 |

Se ganaron 6 códigos y se perdieron 4: es ruido. Además, dos de los códigos leídos en la
versión ampliada no coinciden con el número del frente, cosa que no pasó en ninguna lectura de
la foto completa. Conclusión: el límite es la cámara, no el tamaño con que el texto llega a
Rekognition. Las fotos se envían tal como están y `15_enlarged_ocr_test.py` queda como
evidencia de la prueba, sin más uso.

## 14. OCR nuevo de los 4.017 negativos de la muestra (12-nov-2026)

Alejandro corrió `14_fresh_ocr.py --run --negatives 4017 --keep-images` (7.434 fotos, sin
errores, 7,43 USD, 27 minutos) y `12_ocr_coverage.py`. Quedan leídas 4.398 personas: los 381
positivos y los 4.017 negativos marcados para imágenes. Gasto total en Rekognition hasta aquí:
8,99 USD. En disco hay 8.756 fotos (15 GB): las de los 4.017 negativos y las de 361 positivos;
faltan las de los 20 positivos de la primera prueba.

**Legibilidad por canal** (confianza ≥ 80):

| Etiqueta | Canal | n | Número del frente | Código del reverso | Serie | Registrador |
|---|---|---|---|---|---|---|
| Negativo | App | 156 | 92 % | 35 % | 48 % | 71 % |
| Negativo | Comercio | 3.743 | 77 % | 13 % | 25 % | 25 % |
| Negativo | WhatsApp | 117 | 80 % | 30 % | 42 % | 34 % |
| Positivo | App | 100 | 100 % | 60 % | 43 % | 60 % |
| Positivo | Comercio | 272 | 82 % | 21 % | 36 % | 26 % |

En app el código se lee más en positivos (60 %) que en negativos (35 %). No se sabe por qué;
puede ser el periodo o el tipo de equipo. Hay que tenerlo presente al comparar tasas.

**Reglas** (381 positivos, 4.017 negativos):

| Regla | Positivos: fallos / evaluables | Negativos: fallos / evaluables | Tope de falsos positivos (95 %) |
|---|---|---|---|
| `R_REGISTRAR` | 13 / 127 | 0 / 836 | 0,36 % |
| `R_REPEATED_BACK` | 28 / 121 | 0 / 594 | 0,51 % |
| `R_NUMBER` | 0 / 114 | 0 / 550 | 0,55 % |
| `R_PRODUCTION` | 0 / 112 | 0 / 558 | 0,54 % |
| `R_CODE_FORMAT` | 0 / 118 | 0 / 594 | — |
| `R_MONTHS` | 0 / 345 | 0 / 2.724 | 0,11 % |
| `R_SEX` | 0 / 0 | 0 / 0 | sin medir |

El tope es la regla de tres (3 / evaluables): con cero fallos observados, la tasa real está por
debajo de ese valor con 95 % de confianza.

- Ningún negativo falla ninguna regla: 0 de 977 negativos con alguna regla del código
  evaluable (tope de 0,31 %).
- La detección no cambió: 38 de 381 positivos (24 de los 79 de la red de la tienda 623, 14 de
  los 302 restantes; 26 de 100 en app, 12 de 272 en comercio).
- Los 28 reversos repetidos siguen siendo 28 al agregar 4.017 negativos: las repeticiones son
  entre positivos, ninguna con un negativo.
- Solo dos reglas aportan detección (`R_REGISTRAR` y `R_REPEATED_BACK`). `R_NUMBER`,
  `R_PRODUCTION`, `R_CODE_FORMAT` y `R_MONTHS` no fallan en nadie, ni positivo ni negativo: con
  estos datos no hay de dónde sacarles un peso. `R_SEX` sigue sin poder evaluarse.

**Pendiente de decisión de Alejandro:** si se leen los 23.855 negativos restantes (~48 USD).
Claude recomendó no hacerlo: los 4.017 ya leídos acotan los falsos positivos por debajo del 1 %.

## 15. Revisión visual aplazada y nuevas fuentes de positivos (12-nov-2026)

**Revisión visual.** `13_visual_review.py` tenía un error (Streamlit no interpreta una expresión
suelta partida en dos líneas) y quedó corregido. Alejandro decidió no revisar los 381 positivos
antes de entrenar, porque toma demasiado tiempo: el primer modelo se entrena con todos y después
se revisan solo los que el modelo no detecta. `docs/plan.md` quedó actualizado.

**Nuevas fuentes de positivos que le pasaron a Alejandro** (ESMIO; casos confirmados aunque
`fraud_status` todavía no lo diga):

1. Todos los clientes que sacaron créditos en el comercio AMORETT entre el 14-oct y el
   10-nov de 2026.
2. Todos los clientes con crédito y desembolso a una cuenta de BANCO FINANDINA cuyo comercio de
   la solicitud contenga "LIBRE DESTINO" (hay tres). No dio fechas.
3. El historial del verificador de cédulas del equipo (el artifact de Claude.ai), filtrando por
   "Confirmada falsa".

Quedaron en `config.yaml` (`confirmed_fraud_sources`).
`scripts/16_check_new_positives.py` (solo lectura) cuenta cada grupo: cuántos
clientes hay, con ambas fotos, cuándo se registraron, su `fraud_status`, si tienen crédito
pagado y qué etiqueta tienen hoy. Las tablas salen de `mikro_back`: `credits_credit`,
`credits_creditrequest` (con `payout_id` y `credit_id`), `credits_payout.bank_name` y
`commerces_commerce` (`name` y `commercial_name`). Las consultas corrieron bien en `live`.

Dudas que los conteos deben resolver antes de etiquetar: si se exige que no tengan crédito
pagado (como en los demás positivos), qué hacer con los registrados antes de 2025, si Finandina
lleva ventana de fechas, y cuántos son negativos hoy (pasarían de negativo a positivo). Además,
"fraude confirmado" no es lo mismo que "cédula falsa": si la identidad es robada pero el
documento es auténtico, la foto no tiene nada falso.

**El historial del verificador** se descargó el 12-nov-2026 a
`data/live/private/verifier_history/casos/` (840 casos, 700 cédulas distintas, del 19-oct al
12-nov de 2026; solo local, con números reales). Lo que mostró:

- El veredicto "Confirmada falsa" **no siempre lo puso una persona**. La herramienta marca sola
  como falsa toda cédula con puntaje de 61 o más, y como auténtica toda la de 30 o menos; el
  campo `real_origen` dice quién lo puso.

| Veredicto | Lo puso un analista | Sin registro de quién | Lo puso la herramienta |
|---|---|---|---|
| Falsa | 13 casos | 19 casos | 102 casos |
| Auténtica | 19 casos | 34 casos | 606 casos |

- Entre analista y sin registro hay 25 personas con cédula falsa; solo 2 están en el dataset (ya
  como positivas). Las otras 23 no están en el manifest: habrá que ver con la base si existen
  como clientes y tienen ambas fotos.
- Las 87 personas marcadas como falsas solo por la herramienta no sirven como etiqueta: es el
  puntaje de otro modelo, no una confirmación.

**Conteos en `live` (`16_check_new_positives.py`, 12-nov-2026).** Ninguno de estos clientes
tiene crédito pagado ni es negativo hoy, salvo donde se indica.

| Grupo | Clientes | Con ambas fotos | Ya positivos | Positivos nuevos |
|---|---|---|---|---|
| AMORETT (ESMIO) | 17 | 17 | 0 | 17 |
| Finandina + Libre Destino (ESMIO) | 68 | 68 | 64 | 4 |
| Verificador, falsa por analista: ESMIO | 9 | 8 | 1 | 7 |
| Verificador, falsa por analista: Llévalo Ya | 3 | 1 | 0 | 1 |
| Verificador, falsa sin registro de origen: ESMIO | 15 | 15 | 1 | 14 |
| Verificador, falsa por puntaje: ESMIO / Llévalo Ya | 66 / 24 | 64 / 20 | 6 / 0 | 55 / 20 (no se usan) |

- AMORETT: 17 créditos, todos de clientes de la app registrados en oct–nov de 2026; 16 con
  `POSSIBLE FRAUD` y 1 sin marca.
- Finandina: solo coincidió el comercio "LIBRE DESTINO BOT" (125 créditos de sep–oct de 2026,
  68 clientes de la app). 64 ya eran positivos por la ventana de la tienda 623.
- Los grupos pueden compartir personas; el total de positivos nuevos sale de la exportación.
- Las cédulas que un analista confirmó como **auténticas** (14 números; 10 clientes en ESMIO y
  4 en Llévalo Ya) casi no están en el dataset: son recientes y no tienen crédito pagado. No se
  agregaron como negativos; nadie lo ha pedido.

**Decisiones de Alejandro (12-nov-2026):** misma regla de créditos pagados que los demás
positivos; se descartan los registrados antes de 2025; los casos de Finandina eran cédulas
falsas y van sin ventana de fechas; un cliente que hoy fuera negativo pasa a positivo (con la
regla de créditos pagados eso no puede ocurrir: quien tiene un crédito pagado queda sin
etiqueta; hoy no hay ninguno).

**Cómo quedó en el código:** `dataset.confirmed_customers` reúne los clientes de cada fuente y
`dataset.label_clients` los etiqueta como positivos con `label_source` `CONFIRMED_AMORETT`,
`CONFIRMED_FINANDINA_LIBRE_DESTINO` o `VERIFIER_ANALYST`; quien ya era positivo por
`fraud_status` conserva su origen. Lo usan `10_check_database.py` y `11_export_dataset.py`. El
verificador se lee con `id_fraud_detector/verifier.py`. Solo entran las cédulas con veredicto de
un analista; las 15 sin registro de origen quedan fuera (decisión de Alejandro del
12-nov-2026: no se sabe quién las confirmó). Probado con datos simulados; exportado en `live` el mismo día (§16).

## 16. Exportación con el etiquetado nuevo y cifras consolidadas (12-nov-2026)

Alejandro corrió `11_export_dataset.py` (con imágenes), `14_fresh_ocr.py` para los positivos
nuevos y `12_ocr_coverage.py`. Todas las cifras quedaron reunidas en `docs/data_summary.md`,
que se regenera con `scripts/17_data_summary.py` (solo lee archivos locales y deja las tablas en
`data/live/summary/`). Alejandro lo pidió para armar después la presentación.

**Dataset:** 28.716 clientes etiquetados → 28.292 personas (424 filas repetidas, 0 conflictos):
**410 positivos y 27.882 negativos**. Positivos por origen: `FRAUD` 302, ventana de la tienda
623 79, AMORETT 17, verificador (analista) 8, Finandina 4. Las fotos de los 410 positivos y de
la muestra de negativos se descargaron sin errores.

**La muestra de negativos se movió al volver a exportar.** El sorteo es por grupo de mes y
canal, y al entrar clientes nuevos cambió parte de los elegidos: de los 4.019 marcados ahora,
unos 240 no estaban en la muestra anterior. Esos se leyeron con Rekognition (0,48 USD). Quedan
4.257 negativos con texto nuevo y 4.271 con fotos en disco. Pendiente: hacer que el sorteo sea
estable (por ejemplo, conservar a quien ya tiene fotos), para que una exportación futura no
vuelva a mover la muestra.

**Reglas con los 410 positivos y 4.257 negativos:**

- Detección: 49 de 410 positivos con algún fallo (12 %). Por origen: tienda 623, 26 de 79;
  AMORETT, 8 de 17; `FRAUD`, 14 de 302; Finandina (nuevos), 1 de 4; verificador, 0 de 8.
- AMORETT: 8 de los 17 fallan alguna regla (6 con reverso repetido, 5 con registrador y 2 con
  fecha de producción). Es evidencia de que en ese grupo las cédulas sí eran falsificadas.
- `R_PRODUCTION` detectó por primera vez (2 positivos, los dos de AMORETT).
- Primer falso positivo: 1 negativo de comercio falla `R_REGISTRAR` (1 de 883 evaluables,
  0,11 %). Los demás negativos no fallan nada (1 de 4.257 con algún fallo).
- Por canal: 37 de 128 positivos de app con algún fallo, 12 de 273 de comercio.

**Recorte (`20_crop_test.py`, corrido por Alejandro el 12-nov-2026):** encontró el contorno de
la cédula en 4 de 188 fotos (2 %). El recorte con OpenCV, como está, no funciona con fotos
reales. Hay que replantearlo en la fase 2; un dato a favor de no recortar: el texto ocupa cerca
del 75 % del ancho de la foto, o sea que la cédula ya llena casi todo el cuadro.

**Revisión visual:** Alejandro alcanzó a etiquetar 6 positivos antes de aplazarla (4 "other" y
2 "replaced_photo").

**Gasto total en Rekognition:** 9,53 USD (9.334 fotos más 200 de la prueba de ampliación).

## 17. Decisiones para el clasificador (12-nov-2026)

Alejandro decidió que las 15 cédulas "falsas" del verificador sin registro de origen no entran
(no se sabe quién las confirmó) y aceptó las cinco recomendaciones de Claude para seguir. Quedó
todo en `docs/plan.md` §0.

1. **Dónde se entrena: en su equipo.** NVIDIA RTX 2070 de 8 GB, 12 núcleos y 153 GB libres
   (medido). Las fotos no salen de la máquina y no hay costo.
   - **Marco: PyTorch**, no Keras/TensorFlow como en la materia. Motivo: TensorFlow dejó de usar
     la tarjeta gráfica en Windows nativo (exige WSL2), y Grad-CAM, que es un entregable, es
     directo en PyTorch. La receta es la misma del Taller 1: base preentrenada congelada, cabeza
     nueva y luego afinar las últimas capas. Alejandro aceptó "instalar PyTorch"; conviene
     recordárselo cuando vea el código, porque en clase vio Keras.
2. **Negativos emparejados con los positivos.** La muestra proporcional se parecía a los
   negativos (94 % comercio, 1 % registrado desde septiembre de 2026) y no a los positivos
   (31 % app, 28 % desde septiembre de 2026). Ahora, además de la muestra por mes y canal,
   entran todos los negativos de app (980) y todos los registrados desde el 1-sep-2026 (301):
   `negatives_always_with_images` en `config.yaml`.
   - **El sorteo quedó estable** (`dataset.sample_negatives`): quien ya tiene fotos en disco no
     sale de la muestra y cuenta para la cuota de su mes y canal; lo que falte se toma en un
     orden fijo que depende solo de la persona y de la semilla. Antes cada exportación volvía a
     sortear y movió unos 240.
   - Calculado con el manifest local: quedarían 5.297 negativos con fotos (3.497 de ESMIO y
     1.800 de Llévalo Ya) y faltan por bajar las de 1.026 personas (824 de app), unas 2.052
     fotos. Solo descarga; Rekognition es opcional (unos 2 USD si se quiere medir las reglas en
     esos negativos).
3. **Sin recorte en el primer modelo.** Foto completa reducida. Si queda corto, se compara
   contra una versión recortada.
4. **Evaluación: validación cruzada en cinco partes por persona**, con cada red completa
   (tienda 623, AMORETT, y quienes comparten reverso) en una sola parte. Se descartó la
   partición por tiempo y Llévalo Ya como prueba externa (20 positivos).
5. **Papel de las reglas:** evidencia adicional; el clasificador es el componente principal.

Alejandro quiere mantener al día la copia de `docs/plan.md` del proyecto de Claude.ai, por si
arma la presentación desde allá: hay que avisarle cada vez que el plan cambie. Hizo commit de
todo lo anterior el 12-nov-2026 (`53711e4`).

## 18. Muestra emparejada en `live` e inicio de la fase 3 (12-nov-2026)

**Exportación y lectura.** Alejandro corrió `11_export_dataset.py` con la muestra nueva:
28.293 personas (410 positivos, 27.883 negativos), **5.297 negativos con fotos** (3.497 de ESMIO
y 1.800 de Llévalo Ya), sin errores de descarga. Luego leyó con Rekognition a los 1.040 que
faltaban (2,08 USD). Gasto total en Rekognition: **11,61 USD**. Hoy tienen fotos y texto nuevo
las mismas 5.707 personas: 410 positivos y 5.297 negativos.

**Reglas con más negativos de app** (980 en vez de 156): siguen sin fallar. 1 negativo con
algún fallo entre 5.297 (el mismo de comercio, por `R_REGISTRAR`: 1 de 1.330 evaluables,
0,08 %). Con alguna regla del código evaluable: 1.500 negativos (484 de app). Topes de falsos
positivos de las reglas sin fallos: entre 0,08 % y 0,32 %. En negativos de app el código se lee
en el 38 % y el registrador en el 69 %.

**PyTorch** 2.14.1 con GPU quedó instalado en `.venv` y comprobado con la RTX 2070.

**Código de la fase 3** (nuevo, en PyTorch):

- `evaluation/splits.py` y `scripts/30_make_splits.py`: cinco partes por persona, redes y
  reversos compartidos juntos, parejas en etiqueta y canal. Claude lo corrió (solo lee archivos
  locales) y dejó `data/live/splits.csv`.
- `evaluation/metrics.py`: PR-AUC, ROC-AUC y detección con un 2 % de falsos positivos, sin
  scikit-learn.
- `model/data.py`: reduce cada foto una vez a 512×320 (cubre el cuadro y corta el exceso por
  igual a ambos lados; no agrega barras, porque delatarían la forma del cuadro de la cámara y
  con ella el canal) y la guarda en `data/live/cache/`.
- `model/net.py` y `model/train.py`: MobileNetV3 o EfficientNet-B0 de torchvision con una
  salida; primero solo la cabeza, luego los últimos 6 bloques con tasa baja; pérdida con peso
  para los positivos; parada temprana por PR-AUC de validación; aumento leve en la GPU.
- `scripts/31_train.py`: por cada parte entrena con tres, valida con una y puntúa la quinta;
  deja todo en `runs/<nombre>/`.

Probado con una corrida corta de 400 personas y una parte en la GPU: corre de punta a punta.
Velocidad medida: unas 465 fotos por segundo entrenando, o sea unos 7 segundos por época de
3.400 fotos. La corrida completa de una cara debería tardar alrededor de 20 minutos, la mayor
parte reduciendo las fotos la primera vez. **Ningún modelo completo se ha entrenado todavía.**

**Hallazgo de la partición: la tienda 623 y AMORETT son una sola red.** Comparten reversos
entre sí, así que quedaron en un solo grupo de 100 personas (79 + 17 + los 4 nuevos de
Finandina), completo en la parte 0. Otros 11 positivos `FRAUD` comparten reverso en 5 grupos
pequeños. Consecuencias para leer los resultados:

- 100 de los 128 positivos de app son de esa red; fuera de ella hay 28 positivos de app (7 por
  parte). La parte 0 tiene 133 positivos y las demás unos 69.
- El resultado de la parte 0 responde "¿el modelo habría detectado una red que nunca vio?". El
  de las otras cuatro mide la detección de los `FRAUD` sueltos, casi todos de comercio.
- Cuando la parte 0 está en entrenamiento, la red pesa mucho (100 de unos 240 positivos). Hay
  que mirar los resultados por origen, no solo el total.

## 19. Primer modelo del reverso (12-nov-2026)

Alejandro corrió `python scripts/31_train.py --side back` (MobileNetV3, 512×320, foto completa,
cinco partes). Corrida `runs/back_mobilenet_v3_large_512x320_20261112_1952`. Cada puntaje lo da
un modelo que no entrenó con esa persona ni con su red.

**Resultado, 410 positivos y 5.297 negativos:**

| | Valor |
|---|---|
| Detección dejando pasar a revisión al 2 % de las legítimas | **61,2 %** (251 de 410) |
| PR-AUC | 0,71 |
| ROC-AUC | 0,91 |

Las reglas solas detectan el 12 %. Por parte: 65 %, 83 %, 62 %, 67 % y 67 % (cada parte con su
propio umbral del 2 %). Entrenó entre 13 y 23 épocas por parte.

**Por canal** (un solo umbral para todos): app 59 % (75 de 128), comercio 62 % (169 de 273),
WhatsApp 7 de 9. Los falsos positivos quedan cerca del 2 % en app y en comercio, así que el
umbral no castiga a un canal. Dentro de cada canal el modelo separa bien: ROC-AUC de 0,92 en
app y 0,90 en comercio.

**Por origen del positivo:** `FRAUD` 64 % (192 de 302); red de la tienda 623, 53 % (42 de 79);
AMORETT 41 % (7 de 17); Finandina 3 de 4; verificador 7 de 8. La red (tienda 623 + AMORETT)
estaba completa en la parte 0, o sea que el modelo que la puntuó nunca la había visto: detecta
cerca de la mitad de una red nueva.

**¿Aprendió la cámara o la fecha?** No, según la revisión de `scripts/32_audit_run.py`:

- Un "modelo" que no mira la foto y solo conoce canal, tamaño de la foto y periodo de registro
  llega a ROC-AUC 0,74 y detecta el 25 % al 2 %. Solo con canal y tamaño no detecta nada
  (ROC-AUC por debajo de 0,5). El clasificador llega a 0,91 y 61 %.
- Dentro de grupos donde canal, tamaño exacto de la foto y periodo son iguales para todos (337
  de los 410 positivos caen en grupos con suficiente gente), el ROC-AUC ponderado es 0,92: sigue
  separando cuando la cámara y la fecha no pueden ayudarle.
- El tamaño del archivo solo da ROC-AUC 0,58.

**Dónde falla:** en fotos de baja resolución. En comercio con fotos de hasta 700 px de ancho
detecta el 24 % (7 de 29); con 701 a 1.400 px, el 61 %; con más de 1.400 px, el 75 %. La
cámara limita también al clasificador, no solo a las reglas.

**Reservas:**

- En comercio, 1920×1210, enero a agosto de 2026 hay 60 positivos y detecta el 92 % (ROC-AUC
  0,99). Parece una oleada con una plantilla común que no quedó agrupada, porque sus códigos no
  se leen: positivos de esa oleada están en entrenamiento y en prueba. Reconocer una plantilla
  ya vista es útil en producción, pero infla la estimación frente a falsificaciones nuevas. El
  dato más honesto para "una red que nunca se vio" es el de la parte 0: alrededor del 50 %.
- No se sabe cuántos positivos son cédulas auténticas con identidad robada; esos no se pueden
  detectar por la foto y bajan el techo. Los puntajes son muy polarizados: la mitad de los
  positivos puntúa por encima de 0,96 y una cuarta parte por debajo de 0,11.
- Es una sola corrida con una semilla. Las partes van de 62 % a 83 %.

Falta: el modelo del frente, combinar las dos caras, Grad-CAM y revisar a ojo los positivos que
no detecta.

## 20. Modelo del frente y puntaje combinado (12-nov-2026)

Alejandro corrió `python scripts/31_train.py --side front`
(`runs/front_mobilenet_v3_large_512x320_20261112_2041`). Claude combinó las dos caras con
`scripts/33_combine_sides.py` (nuevo; solo lee los puntajes ya guardados) y dejó el resultado en
`runs/combined/`.

**Detección con 2 % de falsos positivos, 410 positivos y 5.297 negativos:**

| Puntaje | Detectados | PR-AUC | ROC-AUC |
|---|---|---|---|
| Reglas solas | 49 (12 %), con 1 falso positivo | — | — |
| Reverso solo | 251 (61 %) | 0,71 | 0,91 |
| Frente solo | 288 (70 %) | 0,78 | 0,93 |
| **Promedio de frente y reverso** | **322 (79 %)** | **0,84** | **0,95** |
| Promedio de las dos caras, o alguna regla en fallo | 332 (81 %), con 106 falsos positivos (2,0 %) | — | — |

- El promedio no se entrena, así que no hay nada que sobreajustar. El máximo de las dos caras
  da 77 % y el mínimo 71 %.
- **Las dos caras se complementan:** con el umbral propio de cada una, 204 positivos los
  detectan ambas, 84 solo el frente, 47 solo el reverso y 75 ninguna.
- **Las reglas agregan 10 positivos** que el clasificador no detecta, a cambio de 1 negativo.
- Por parte (promedio): 80 %, 86 %, 77 %, 83 % y 75 %.
- Por canal: app 80 % (102 de 128), comercio 77 % (211 de 273), WhatsApp 9 de 9; falsos
  positivos de 1,8 % y 2,0 % en app y comercio.
- Por origen: `FRAUD` 79 % (238 de 302); red de la tienda 623, 80 % (63 de 79); AMORETT 65 %
  (11 de 17); Finandina 3 de 4; verificador 7 de 8. La red estaba completa en la parte 0: su
  modelo nunca la vio y detecta 77 de sus 100 personas.
- Por periodo de registro: 76 % en el primer semestre de 2025, 73 % en el segundo, 91 % de
  enero a agosto de 2026 y 78 % de septiembre a noviembre de 2026.

**Revisión de cámara y fecha** (`32_audit_run.py` sobre `runs/combined` y sobre el frente): el
puntaje que no mira la foto sigue en ROC-AUC 0,74 y 25 % de detección. Dentro de grupos con
igual canal, tamaño de foto y periodo, el ROC-AUC ponderado es 0,96 con el frente y 0,96 con el
promedio. No aprendió la cámara ni la fecha.

**La resolución sigue siendo el límite:** en comercio, con fotos de hasta 700 px de ancho el
promedio detecta el 41 % (12 de 29); con 701 a 1.400 px, el 81 %; con más de 1.400 px, el 83 %.

**Reservas, las mismas del §19:** la oleada de comercio de enero a agosto de 2026 con fotos de
1920×1210 se detecta en el 98 % (59 de 60), lo que apunta a una plantilla repetida entre
entrenamiento y prueba; no se sabe cuántos de los 88 positivos no detectados son cédulas
auténticas; es una sola corrida con una semilla y un solo tamaño de entrada.

Falta: Grad-CAM, revisar a ojo los 88 positivos no detectados, calibrar el puntaje a 0–100 y el
backtest.

## 21. Segunda iteración y revisión de los no detectados (12-nov-2026)

Alejandro preguntó si el modelo se puede mejorar y si valdría la pena no usar transferencia de
aprendizaje. Respuesta de Claude: mantener la transferencia (con 410 positivos, una red desde
cero casi con seguridad queda muy por debajo; es una expectativa, no una medición) y correr una
vez la versión sin ella como comparación para el informe. Para mejorar, en orden de lo que se
espera que aporte: limpiar las etiquetas revisando los no detectados, una entrada más grande
(768×480), promediar varios modelos, otra red (EfficientNet-B0) y más positivos cuando el equipo
de fraude confirme los de octubre y noviembre.

Advertencia acordada: con 410 positivos no se hace búsqueda amplia de variantes. Se decide una
lista corta de antemano, se corre completa y se reportan todos los resultados.

`31_train.py` ganó las opciones `--no-pretrained`, `--seed` y `--lr` (probadas en una corrida
corta). Ningún experimento de la segunda iteración se ha corrido.

**Revisión visual de los no detectados.** Alejandro aceptó revisar a ojo los 88 positivos que el
puntaje combinado no marca (10 de ellos sí fallan alguna regla). `33_combine_sides.py` deja la
lista en `runs/combined/missed_positives.csv`, del puntaje más bajo al más alto, y
`13_visual_review.py` se reescribió: `--queue` para mostrar solo esa lista, un clic por caso,
botón para deshacer y sin mostrar los puntajes del modelo. La primera opción es "Looks genuine"
(documento auténtico, el fraude estuvo en otra parte), que es la pregunta de fondo: cuántos
fallos son del modelo y cuántos de la etiqueta. Probada sin interfaz contra los archivos
locales, guardando en un archivo temporal.

## 22. Resultados de la segunda iteración (13-nov-2026)

Alejandro corrió las ocho corridas de la lista (otra semilla, 768×480, EfficientNet-B0 y sin
transferencia, para frente y reverso). La revisión visual de los no detectados no se ha hecho.
Las tablas completas están en `docs/data_summary.md` §13.

**Resultados** (detección con 2 % de falsos positivos; frente, reverso y promedio de ambas):

- Primera corrida: 70,2 %, 61,2 % y 78,5 % (322 de 410).
- Otra semilla: 77,3 %, 62,4 % y 79,5 %.
- Entrada de 768×480: 73,2 %, 60,7 % y 78,0 %.
- EfficientNet-B0: 78,8 %, 68,5 % y 86,6 % (355).
- Sin transferencia: 11,7 %, 23,2 % y 16,6 %.
- **Promedio de las cuatro preentrenadas: 81,2 %, 71,7 % y 86,8 % (356 de 410)**; PR-AUC 0,89,
  ROC-AUC 0,96. Con las reglas, 360 de 410 y 106 de 5.297 negativos.

**Lectura.**

- La semilla sola mueve el frente 7 puntos. Las diferencias entre corridas sueltas no alcanzan
  para decir que una variante es mejor; lo que sí se sostiene es el promedio de varias, que mejora
  8,3 puntos sobre la primera corrida (intervalo del 95 % por remuestreo: de 5,6 a 11,2).
- El promedio usa las cuatro variantes preentrenadas sin escoger entre ellas. Se dejó así a
  propósito: elegir la mejor combinación sobre la misma validación cruzada inflaría la cifra.
- La entrada más grande no ayudó. EfficientNet-B0 salió mejor en las dos caras, pero con una
  corrida por cara no se separa del efecto de la semilla.
- Sin transferencia el modelo casi no aprende y es inestable entre partes. Varias partes pararon
  temprano (entre 7 y 13 épocas), así que esto mide esta receta con estos datos, no un tope de lo
  que una red desde cero podría lograr con más datos.
- La red tienda 623 + AMORETT, nunca vista por su modelo: 93 de 100 (antes 77). Comercio con
  fotos de hasta 700 px: 17 de 29 (antes 41 %).
- Revisión de cámara y fecha sobre el promedio: ROC-AUC 0,98 dentro de grupos con igual canal,
  tamaño y periodo; el puntaje que solo usa esos datos sigue en 0,74.
- Quedan 54 positivos sin detectar (4 fallan alguna regla). De los 88 de antes se recuperan 35
  y se pierde 1.

**Cambios en el código.** `33_combine_sides.py` recibe ahora `--front` y `--back` con una o varias
corridas y promedia las de cada cara; con la primera pareja reproduce `runs/combined` idéntico.
El promedio de las cuatro quedó en `runs/ensemble`, y la lista para la revisión visual en
`runs/ensemble/missed_positives.csv`.

**Pendiente de decidir por Alejandro:** si el modelo final es el promedio de las cuatro variantes
(ocho redes al puntuar una persona) o solo EfficientNet-B0 (dos redes). `docs/plan.md` no se
cambió.

**Pérdida y exactitud en el historial (13-nov-2026).** Alejandro preguntó por las curvas de
pérdida de entrenamiento y de validación, que son las que en clase se usan para ver el
sobreajuste. Hasta aquí `history.csv` solo guardaba la pérdida de entrenamiento y el PR-AUC de
validación. `model/train.py` guarda ahora, por época, `train_loss`, `train_accuracy`, `val_loss`,
`val_accuracy` y `val_pr_auc`; la pérdida de validación pesa los positivos igual que la de
entrenamiento, para que las dos curvas se puedan comparar. La época que se conserva la sigue
decidiendo el PR-AUC de validación: nada del entrenamiento cambió. Las corridas anteriores no
tienen esas columnas; para la gráfica hay que repetir una. Probado con una corrida corta.

**Paciencia de la parada temprana: de 4 a 8 (pedido de Alejandro, 13-nov-2026).** Es el valor por
defecto de `31_train.py` y de `TrainConfig`, también para las corridas sin transferencia (antes
6). Las diez corridas del 12 y 13-nov-2026 (incluida `runs/ensemble`) se hicieron con 4, o con 6
las dos sin transferencia; cada `config.json` guarda la que usó. Las corridas nuevas no se
comparan directamente con esas: la diferencia mezcla la paciencia con el azar de la GPU.

## 23. Curvas de pérdida y prueba de la paciencia (13-nov-2026)

Alejandro repitió EfficientNet-B0 con las mediciones nuevas: el frente con paciencia 4 (arrancó
antes del cambio) en `runs/front_efficientnet_b0_curves`, y el reverso con paciencia 8 en
`runs/back_efficientnet_b0_curves`. `scripts/34_plot_history.py` (nuevo) dibuja las curvas y las
deja en `curves_loss.png` y `curves_accuracy.png` de cada corrida; `matplotlib` quedó en el grupo
`train` de `pyproject.toml`.

**Lo que muestran las curvas.**

- En las diez partes hay sobreajuste: la pérdida de entrenamiento baja hasta 0,01–0,09 y la de
  validación toca su mínimo entre las épocas 4 y 8 (contando las 3 de la capa final) y después
  sube, hasta más del doble en varias partes.
- La época que se conserva (mejor PR-AUC de validación) casi siempre queda después del mínimo de
  la pérdida de validación, a veces muy después (reverso, parte 1: época 20, pérdida 1,24 contra
  un mínimo de 0,70). El orden de los puntajes sigue mejorando mientras los puntajes se vuelven
  demasiado seguros. No afecta la detección al 2 %, que solo depende del orden, pero confirma que
  los puntajes no sirven como probabilidad sin la calibración de la fase 5.
- El frente reprodujo la corrida de ayer (78,8 %): agregar las mediciones no cambió el
  entrenamiento.

**Paciencia 8 contra 4 (solo el reverso de EfficientNet-B0, una corrida):** 71,7 % contra 68,5 %,
3,2 puntos más (intervalo por remuestreo: de 0,5 a 4,6). Cambiaron tres partes; la mejora viene
casi toda de la parte 1 (de 71 % a 84 %), donde el mejor PR-AUC de validación apareció en la
época 17 del afinado. Claude había dicho que no esperaba mejora; en el reverso sí la hubo. Con
una corrida no alcanza para darla por firme, y el frente no se ha probado con 8.

**Pendiente de decidir por Alejandro:** si se repiten las cuatro variantes preentrenadas con
paciencia 8 para rehacer el promedio. `runs/ensemble` (87 %) sigue hecho con paciencia 4.

**Modelo final (preferencia de Alejandro, 13-nov-2026).** Cada corrida deja cinco modelos, uno
por parte; sirven para medir la receta, no son el modelo que se usaría con clientes nuevos.
Alejandro prefiere entrenar un modelo final con todos los datos, una vez que la validación
cruzada dé un resultado aceptable y la receta quede fija. Queda para la fase 5. Puntos a
resolver en ese momento: cuántas épocas entrenar sin parte de validación (la mediana de las
mejores épocas de la validación cruzada), que la calibración y el umbral salen de los puntajes
de la validación cruzada y no del modelo final, y que el backtest no puede hacerse sobre
personas con las que el modelo final entrenó.

## 24. Revisión visual de los no detectados (13-nov-2026)

Alejandro revisó en la app los 54 positivos que el promedio de las cuatro variantes no detecta
(`runs/ensemble/missed_positives.csv`). Preguntó si hacía falta el motivo; se le dijo que no,
que bastaba con decir si la cédula se ve auténtica o falsificada. Resultado
(`data/live/review/modalities.csv`, 60 respuestas con las 6 de antes):

- **48 se ven falsificadas** (41 `FRAUD`, 6 de la tienda 623, 1 de AMORETT; 39 de comercio y 9
  de app; 11 con fotos de hasta 700 px). Las 4 que fallan alguna regla están aquí.
- **3 se ven auténticas:** las tres son `FRAUD` de comercio registrados en 2025.
- **3 sin poder decidir.**

**Lectura.** La hipótesis de que buena parte de los no detectados fueran documentos auténticos con
el fraude en otra parte no se cumplió: son 3 de 54. Claude había señalado como candidatos a los
21 con puntaje casi cero en las dos caras; la mayoría también se ven falsificados. Entonces el
tope del modelo no está en las etiquetas: hay 48 falsificaciones que una persona reconoce y el
modelo no. No quedó registrado qué las delata, porque no se marcó el tipo ni se escribieron
notas.

**Reservas.** La revisión no fue a ciegas (todos los casos mostrados estaban marcados como
fraude) y solo cubre a los no detectados, así que no sirve para corregir la cifra principal: se
sigue reportando 356 de 410. Sin las 3 auténticas sería 356 de 407 (87,5 %).

**Pendiente de decidir por Alejandro:** si las 3 que se ven auténticas salen de los positivos al
entrenar el modelo final. Son muy pocas para repetir la validación cruzada solo por eso.

**Qué delata a las 48 (dicho por Alejandro, 13-nov-2026; sin conteo por tipo):**

1. Impresiones en papel.
2. Láminas reemplazadas con datos nuevos: la cédula se ve vieja o borrada de fondo, pero los
   datos se ven muy claros y negros.
3. Fotos borrosas a propósito. Se sabe que son fraude porque el equipo reconoce por la foto a
   personas que lo intentan con frecuencia.
4. Foto o datos reemplazados: borroso detrás de los datos para tapar los anteriores, y el marco
   de la foto con inconsistencias.

**Dos comprobaciones de Claude sobre archivos locales (solo conteos):**

- *¿El texto le llega muy pequeño al modelo?* Con las cajas de texto que Rekognition ya había
  devuelto: una línea de texto mide en mediana 37 px en la foto original del frente, 14 px en la
  entrada de 512×320 y 22 px en la de 768×480; recortando por las cajas de texto llegaría a 30 px
  (reverso: 32, 12, 18 y 21 px). La ganancia de recortar es pequeña frente a la entrada grande,
  que ya se probó sin mejora. Además el recorte abriría un atajo: hay tres o más líneas legibles
  en el reverso del 98,5 % de los positivos y solo del 88 % de los negativos. **No se recomienda
  recortar por las cajas de texto.**
- *¿Los no detectados son las fotos borrosas?* No. Midiendo la nitidez de la foto reducida, solo
  2 de las 48 están entre las más borrosas (el 10 % menos nítido de los negativos), y en ese
  grupo el modelo detecta 19 de 21 positivos. La medida es gruesa (la reducción esconde parte del
  desenfoque), pero no señala la nitidez como la causa.

**Lectura.** Los fallos son de cuatro tipos distintos, cada uno con pocos ejemplos; no hay una
causa única que un ajuste resuelva. Dos puntos para el informe: el tipo 3 no es detectable por la
foto del documento (el indicio es la cara de una persona conocida, no la cédula), y el producto
debería tratar una foto ilegible como "no evaluable" y no como legítima. Riesgo anotado: si una
misma persona aparece con varias identidades en partes distintas de la validación cruzada, el
modelo del frente puede estar reconociendo la cara, y eso inflaría su resultado. La partición
agrupa por red y por reverso compartido, no por cara. No se ha medido.

## 25. Mapas de calor y regularización (13-nov-2026)

**Grad-CAM.** `src/id_fraud_detector/model/cam.py` y `scripts/35_grad_cam.py` (nuevos). Para cada
persona se usa el modelo de la parte que no la vio. Corrido sobre EfficientNet-B0 del frente y
del reverso (las corridas que están en el promedio). Claude solo miró los mapas promedio por
grupo (`runs/<corrida>/cam_mean.png`), que no contienen fotos; las imágenes por persona de los 54
no detectados quedaron en `data/live/cam/<corrida>/missed_positives/` para que las vea Alejandro.

Parte del calor que cae en el borde de la foto (un calor repartido por igual daría 0,40):

| Grupo | Frente | Reverso |
|---|---|---|
| Positivos detectados | 0,14 | 0,30 |
| Positivos no detectados | 0,37 | 0,41 |
| Negativos marcados | 0,21 | 0,33 |
| Negativos no marcados | 0,46 | 0,48 |

- **Frente:** cuando el modelo marca a alguien (positivo o negativo), el calor es una sola mancha
  en el centro-derecha de la foto, donde va la foto de la persona en la cédula. Cuando no marca,
  el calor es difuso y evita esa zona.
- **Reverso:** el calor de los detectados se reparte en cuatro o cinco zonas de la cédula
  (centro, parte superior izquierda y las dos esquinas inferiores).
- **Ninguno de los dos se apoya en el fondo.** Es otra evidencia, además de la revisión de cámara
  y fecha, de que el modelo mira el documento.

**Riesgo abierto del frente.** Que se apoye en la foto de la persona admite dos lecturas: que
detecta la foto pegada o reemplazada (lo que Alejandro describió en los tipos 2 y 4), o que
reconoce caras que se repiten. A favor de la primera: de las 100 personas de la red tienda 623 +
AMORETT, que ningún modelo vio al entrenar, el frente solo detecta 85. No lo descarta para el
resto de los positivos. Sigue pendiente que Alejandro diga si las mismas personas aparecen en
muchos fraudes.

**Límite del método.** El mapa se escala por su propio máximo, así que muestra dónde miró el
modelo y no con cuánta fuerza; en una foto con puntaje casi cero el mapa no significa mucho. El
puntaje recalculado difiere hasta 0,02 del guardado, porque aquí no se usa precisión mixta.

**Sobreajuste.** Alejandro preguntó cómo mejorar la red para evitarlo antes de repetir con
paciencia 8. `31_train.py` ganó `--dropout` y `--weight-decay` (probados en una corrida corta);
las opciones `--blocks` y `--lr` ya existían. Propuesta de Claude: un solo experimento decidido
de antemano con más regularización (decaimiento de pesos 0,01, dropout 0,5 y 3 bloques liberados
en vez de 6) sobre EfficientNet-B0, las dos caras, comparado con la misma red y paciencia.
Ninguno se ha corrido.

**Paciencia 6 (decisión de Alejandro, 13-nov-2026).** Para el experimento de regularización
prefirió 6 en vez de 8; quedó como valor por defecto de `31_train.py` y `TrainConfig`. No hay
corrida de referencia de EfficientNet-B0 con paciencia 6 (el frente existe con 4; el reverso con
4 y con 8), así que el experimento son cuatro corridas: la receta actual y la regularizada
(`--weight-decay 0.01 --dropout 0.5 --blocks 3`), cada una para frente y reverso.

## 26. Experimento de regularización y lectura de la paciencia (13-nov-2026)

Alejandro corrió las cuatro corridas de EfficientNet-B0 con paciencia 6: `runs/front_efficientnet_p6`
y `runs/back_efficientnet_p6` (receta actual) y `runs/front_efficientnet_p6_reg` y
`runs/back_efficientnet_p6_reg` (decaimiento de pesos 0,01, dropout 0,5 y 3 bloques liberados).
Detección con 2 % de falsos positivos, sobre los 410 positivos:

| | Receta actual | Regularizada | Diferencia [IC 95 %] |
|---|---|---|---|
| Frente | 79,0 % (324) | 75,9 % (311) | −3,2 [−7,1; +0,7] |
| Reverso | 68,5 % (281) | 62,7 % (257) | −5,9 [−10,3; −1,0] |
| Promedio de las dos caras | 86,1 % (353) | 82,4 % (338) | −3,7 [−6,1; −0,5] |
| Red nunca vista (100) | 94 | 84 | |
| Resto de positivos (310) | 259 | 254 | |

**La regularización empeoró el resultado y no quitó el sobreajuste.** Con la receta regularizada
la pérdida de entrenamiento también baja a 0,03–0,09 y la de validación tiene su mínimo en las
épocas 5 a 8 y después sube, igual que antes. Lo único que mejora es la pérdida de validación del
reverso en la época conservada (0,86 contra 1,07): el modelo queda menos seguro de más, pero
ordena peor. La mayor caída está en la red que el modelo nunca vio (94 a 84 de 100). Los tres
cambios se probaron juntos, así que no se sabe cuál de ellos hace daño; no se van a separar
(sería búsqueda de hiperparámetros sobre la misma validación cruzada). **Se queda la receta
actual.** El sobreajuste lo controla la parada temprana; el exceso de confianza del puntaje se
corrige en la calibración de la fase 5.

**Lo que mostraron las corridas sobre la paciencia.** Con la misma semilla el entrenamiento sigue
el mismo camino, así que cambiar la paciencia solo cambia dónde se corta:

- Reverso, paciencia 6 contra 4: se conserva la misma época en las cinco partes; el resultado es
  idéntico (281 de 410).
- Frente, paciencia 6 contra 4: cambia la época en dos partes (19 en vez de 8; 21 en vez de 13);
  324 contra 323.
- Reverso, paciencia 8: la mejora de 3,2 puntos medida en §23 viene del modelo de una sola parte
  (la 1, que siguió hasta la época 20 en vez de quedarse en la 11). No es evidencia de que más
  paciencia sea mejor en general.
- Las dos caras de EfficientNet-B0: 355 con paciencia 4 y 353 con paciencia 6 (−0,5 [−1,5; +1,7]).

Recomendación de Claude: dejar la paciencia en 6 y **no rehacer el promedio de las cuatro
variantes**; `runs/ensemble` (paciencia 4, 356 de 410) sigue siendo el resultado de la validación
cruzada. Para el modelo final la paciencia no se usa: se entrena un número fijo de épocas. Con
esto la receta queda fija y lo que sigue es la fase 5.

## 27. Formas de combinar frente y reverso, y caras repetidas (13-nov-2026)

**Caras repetidas.** Alejandro respondió la pregunta que estaba abierta desde §24: una misma
persona puede aparecer en varios fraudes, pero no sabe cuántas veces pasa. El riesgo queda sin
medir: si la misma cara está en la parte de entrenamiento y en la de prueba, el modelo del
frente puede reconocer la cara en vez de la falsificación y la detección medida queda inflada
para esas personas. Lo único medido a favor es la red tienda 623 + AMORETT (100 personas, todas
en una parte): el frente detecta 85 a 89 sin haber visto a ninguna.

**¿Un modelo de ensamble sobre los dos puntajes?** Alejandro preguntó si convenía un modelo que
combine los resultados del frente y del reverso. Se midió sin entrenar nada, con los puntajes
guardados de `runs/ensemble` y de EfficientNet-B0 con paciencia 6. Lo que tiene pesos ajustados
se ajustó con cuatro partes y se aplicó a la quinta. Detección con 2 % de falsos positivos:

| Forma de combinar | Cuatro variantes | EfficientNet-B0 sola |
|---|---|---|
| Promedio simple (lo que se usa) | 86,8 % (356) | 86,1 % (353) |
| Máximo de los dos | 84,9 % (348) | 84,1 % (345) |
| Promedio con pesos ajustados | 86,6 % (355) | 82,2 % (337) |
| Regresión logística sobre los dos puntajes | 87,1 % (357) | 84,9 % (348) |
| Regresión logística sobre los logits | 86,1 % (353) | 86,3 % (354) |
| Promedio de los logits | 86,3 % (354) | 87,3 % (358) |
| Regresión logística con la regla como tercera entrada | 87,1 % (357) | 84,9 % (348) |

Ninguna forma supera al promedio simple más allá del ruido (la mejor diferencia es +1,2
[−0,5; +3,2]) y ninguna gana en las dos columnas a la vez. La regresión aprende pesos casi
iguales para las dos caras (5,7–6,7 y 5,0–5,9), que es lo mismo que promediar. **Se queda el
promedio simple.** Las dos caras sí se complementan: de los 410 positivos, 267 los marcan las dos,
66 solo el frente, 27 solo el reverso y 50 ninguna; el promedio ya aprovecha eso.

Otra lectura del mismo cálculo: EfficientNet-B0 más MobileNetV3 da 85,1 %, y agregar corridas
hasta las cuatro lleva a 86,8 %. Promediar más corridas ayuda poco una vez está EfficientNet-B0.

## 28. Caras repetidas: medición (13-nov-2026)

Alejandro aprobó construir la comprobación. `src/id_fraud_detector/evaluation/faces.py` y
`scripts/36_face_repeats.py` (nuevos) leen la cara de cada foto del frente con dos modelos
públicos de OpenCV (YuNet y SFace, en `data/face_models/`; no agregan dependencias) y comparan
todos los pares de personas. Todo corre en el equipo; Claude solo vio conteos. Las descripciones
de las caras quedan en `data/live/faces/`.

**Caras encontradas:** en 394 de los 410 positivos (96 %) y en 4.497 de los 5.297 negativos
(85 %). En app: 100 % de los positivos y 84 % de los negativos.

**Umbral.** Dos caras "coinciden" si su parecido supera un umbral. Como no hay pares que se sepa
que son la misma persona, el umbral se fijó por el otro lado: que dos negativos (personas
distintas) coincidan por azar una vez cada 100.000 pares (parecido 0,546). Coincidencias entre
positivos contra las esperadas por azar:

| Azar permitido | Parecido | Pares de positivos | Esperados por azar |
|---|---|---|---|
| 1 en 1.000.000 | 0,613 | 7 | 0,1 |
| 1 en 100.000 | 0,546 | 9 | 0,8 |
| 1 en 10.000 | 0,487 | 23 | 7,7 |
| 1 en 1.000 | 0,421 | 101 | 77,4 |

**Sí hay caras repetidas, pero son pocas.** Con el umbral de 1 en 100.000, 12 positivos comparten
cara con otro positivo, todos de origen `FRAUD`; 7 con uno de otra parte y 5 con uno que estaba
en el entrenamiento de su propio modelo. Con el umbral de 1 en 10.000 son 20 con uno del
entrenamiento, de los que 9 se esperan por azar. Con umbrales más flojos ya no hay más
coincidencias que las del azar. Queda entre 5 y 11 positivos de 410 (1 a 3 %) cuya cara pudo
verse en el entrenamiento. **En la red tienda 623 + AMORETT no hay ninguna cara repetida**: la
red repite reversos, no caras.

**No se detectan mejor.** Con `runs/ensemble`, de los 5 positivos cuya cara estaba en el
entrenamiento el frente detecta 4 (80 %); de los 382 sin cara repetida, el 83 %. Con el umbral de
1 en 10.000: 85 % contra 83 % en el frente, y la misma diferencia en el reverso (75 % contra
73 %), que no puede usar la cara. Al ordenar los positivos por el parecido de su cara con la más
parecida del entrenamiento, la detección del frente sube poco (81 % a 88 % del cuarto más bajo
al más alto, sin la red) y la del reverso sube más (64 % a 77 %): es calidad de la foto, no
memoria de caras. Sin los 5 positivos, la detección queda en 86,9 %.

**Conclusión.** El 87 % no está inflado por caras repetidas en una cantidad que se pueda medir;
en el peor caso (que esos 5 a 11 se detectaran solo por la cara) serían 1 a 3 puntos. No se
rehace la partición. Límite: no se midió cuántas caras repetidas se le escapan al comparador
(las caras de las cédulas son pequeñas y las fotos, de mala calidad). Las 9 parejas están en
`data/live/faces/pairs/` para que Alejandro confirme que son la misma persona.
`face_groups.csv` no sirve tal cual para agrupar la partición: entre los negativos trae unas 100
parejas que coinciden por azar.

**Hallazgo aparte.** El detector encuentra la cara más a menudo y con más confianza en los
positivos: la confianza del detector, sola, separa positivos de negativos con ROC-AUC 0,70. Va
en la misma dirección que Grad-CAM (el frente mira la foto de la persona) y que lo que Alejandro
describió (foto reemplazada o impresa, que se ve más nítida que la de una cédula auténtica). En
los 16 positivos sin cara encontrada el frente detecta 5.

## 29. Modelo final: EfficientNet-B0, una red por cara (13-nov-2026)

Alejandro preguntó qué significaba elegir entre "cuatro variantes" y "solo EfficientNet-B0", y
si varias redes harían lenta la evaluación, porque al final esto corre como API. Tiempo de una
predicción medido en su equipo, en CPU con un hilo y una imagen sintética: EfficientNet-B0
143 ms; MobileNetV3 67 ms a 512×320 y 158 ms a 768×480; el mapa de calor de EfficientNet-B0,
153 ms con la predicción incluida. Por cliente (frente y reverso): 0,3 s con dos redes y 0,9 s
con seis u ocho. Cada red pesa 16 MB. En Lambda no se ha medido; Claude estima 2 a 3 veces más.

**Decisión de Alejandro ("por ahora"):** el modelo final es EfficientNet-B0 sola, una red por
cara, con promedio simple de frente y reverso. Quedó como decisión 7 en `docs/plan.md` §0. No
se midió el promedio de EfficientNet-B0 con varias semillas (4 a 6 redes); queda como opción si
una sola red resulta inestable. La cifra de validación cruzada de este modelo es 86,1 %
(paciencia 6) a 86,6 % (paciencia 4); el 86,8 % de `runs/ensemble` es de las cuatro variantes.

## 30. Las tres cédulas auténticas: robadas, no falsificadas (13-nov-2026)

Alejandro explicó por qué las 3 cédulas que vio auténticas en la revisión (§24) están marcadas
como `FRAUD`: la selfie que se tomó el cliente no coincidía con la foto de la cédula. Son
documentos auténticos usados por otra persona. Su razonamiento: este modelo es para
falsificaciones, así que esas fotos no son ejemplos de lo que busca; el fraude lo deben alertar
las demás verificaciones cuando el indicador se combine con ellas. Propuso usarlas como
negativos y dejó a criterio de Claude excluirlas.

**Quedan excluidas** (decisión 8 de `docs/plan.md` §0). Con los puntajes que ya existen da lo
mismo excluirlas que pasarlas a negativos, y excluirlas mantiene una sola definición de negativo
(crédito pagado y sin marca) en vez de mezclar tres casos decididos a ojo:

| Modelo | Como están en la base | Excluidas | Como negativos |
|---|---|---|---|
| EfficientNet-B0, paciencia 6 (modelo final) | 353 de 410 (86,1 %) | 353 de 407 (86,7 %) | 353 de 407 (86,7 %) |
| EfficientNet-B0, paciencia 4 | 355 de 410 (86,6 %) | 355 de 407 (87,2 %) | 355 de 407 (87,2 %) |
| Promedio de cuatro variantes | 356 de 410 (86,8 %) | 356 de 407 (87,5 %) | 356 de 407 (87,5 %) |

Ningún modelo marcaba a las tres. `model/data.excluded_by_review` lee las respuestas de la
revisión y `31_train.py` deja a esas personas fuera del entrenamiento, de la validación y de
los puntajes, sin tocar la caché de fotos. No se volvió a entrenar nada: las corridas
existentes las tienen como positivos y basta quitarlas al calcular.

**Reserva que sigue en pie.** Solo se revisaron los 54 positivos no detectados. Entre los 353
detectados puede haber más cédulas auténticas robadas que el modelo marca; contarían como
aciertos sin serlo. Para saberlo habría que conocer el motivo de cada `FRAUD`.

**Respuesta de Alejandro sobre el motivo (13-nov-2026).** La base no distingue los casos: no hay
un campo que diga si un `FRAUD` fue por cédula falsificada o por cédula auténtica robada. La
reserva queda como limitación del proyecto. Como propuesta de mejora, Alejandro plantea mejorar
la captura de las fotos y registrar el motivo del fraude o de la falsificación, para reentrenar
un modelo mejor más adelante. Quedó en `docs/plan.md` §8.1 con las cifras medidas que respaldan
cada propuesta.

## 31. Fase 5: calibración del indicador (13-nov-2026)

Alejandro preguntó qué es calibrar y si no hacía falta primero el modelo final. No: el modelo
final ve a todas las personas y no deja puntajes limpios; se calibra con los de la validación
cruzada y la función se le aplica después. Supuesto: el modelo final se comporta como los cinco
de la validación cruzada.

Existen `src/id_fraud_detector/risk.py` (calibración de Platt e isotónica, peso de las reglas,
indicador), `evaluation/population.py` (pesos de muestreo) y `scripts/50_calibrate.py`. Corrido
sobre `runs/front_efficientnet_p6` y `runs/back_efficientnet_p6`; resultado en `runs/indicator/`.
Todo lo que sigue es "anidado": la calibración y los pesos que se aplican a una parte se
ajustaron sin ella.

**Población.** Las 5.704 personas con fotos (407 positivos; 3 excluidos por la revisión)
representan a 28.290 personas con etiqueta: cada negativo pesa lo que representa su celda de
canal y mes (entre 1 y 7,3). Las cédulas falsas son el 1,44 % de esa población y el 7,14 % de la
muestra. El 2 % de falsos positivos medido en la muestra es 1,98 % en la población: la muestra
emparejada no distorsiona esa cifra.

**Método.** Pérdida logarítmica ponderada (menor es mejor): decir siempre 1,44 %, 0,0753; el
puntaje crudo, 0,0610; Platt, 0,0287; isotónica, 0,0264. Se usa la **isotónica**: el promedio de
dos redes que casi siempre dicen 0 o 1 se amontona en 0, 0,5 y 1, y la probabilidad salta de 5 %
a 25 % justo al pasar de 0,50 a 0,52 (una cara dice "falsa" y la otra empieza a dudar). Una
curva logística no puede seguir ese salto; Platt quedaba mal en la mitad de la escala (12 de
indicador contra 26 falsas por cada 100).

**¿Los cinco modelos se calibran igual?** Cuatro sí (un puntaje de 0,9 equivale a 62–71 % de
probabilidad); el de la parte 4 no (26 %). El supuesto de que la calibración se traslada al
modelo final es aproximado; habrá que recalibrar con los casos que acumule el modo SHADOW.

**Peso de las reglas.** Un FAIL multiplica las probabilidades a favor (odds) por el peso de la
regla. El peso es la razón más baja que admiten los datos: límite inferior (Wilson, 95 %) de la
tasa de fallo entre falsas dividido por el límite superior entre auténticas, nunca menor que 1.

| Regla | Falsas: fallan / evaluadas | Auténticas: fallan / evaluadas | Peso |
|---|---|---|---|
| `R_REPEATED_BACK` | 37 / 137 | 0 / 990 | 78,0 |
| `R_REGISTRAR` | 19 / 142 | 1 / 1.330 | 27,8 |
| `R_PRODUCTION` | 2 / 126 | 0 / 929 | 1,8 |
| Las demás | 0 | 0 | 1 (no cambian nada) |

Medidos sin la parte 0 (donde está la red), los pesos bajan a 28,1 y 1,9: la mayoría de los
fallos son de la red. Las reglas solas llevan a revisión a 3 positivos más y a ningún negativo.

**Cortes posibles** (indicador = probabilidad calibrada × 100, con reglas):

| Legítimos por encima | Indicador | Falsas por encima | Falsas por cada 100 marcados |
|---|---|---|---|
| 2 % (107 de la muestra) | 3,1 | 352 de 407 (86,5 %) | 39 |
| 1 % (54) | 11,6 | 343 (84,3 %) | 55 |
| 0,5 % (23) | 40,9 | 280 (68,8 %) | 68 |
| 0,2 % (9) | 52,2 | 250 (61,4 %) | 83 |
| 0,1 % (4) | 67,3 | 221 (54,3 %) | 89 |
| 0,02 % (1) | 78,7 | 200 (49,1 %) | 97 |

Por canal con el corte del 2 %: app 119 de 128 y 1,94 % de legítimos; comercio 224 de 270 y
1,94 %; WhatsApp 9 de 9 y 3,65 %.

**Lo que hay que decidir.** Con el número como probabilidad, la revisión empezaría en 3, no en
40 como decía el plan; y la proporción real de fraude entre los clientes nuevos no se conoce
(1,44 % entre los que tienen etiqueta; cerca de 0,15 % si se cuenta a todos los registrados).

**Explicaciones pedidas por Alejandro (13-nov-2026).** Las preguntas sobre la escala y los
cortes no se entendieron a la primera; pidió varias aclaraciones: qué entrega el modelo (un
número entre 0 y 1 por foto), qué devuelve el producto (franja, indicador, puntajes, reglas y
mapas de calor), qué diferencia hay entre probabilidad calibrada e indicador (ninguna si el
indicador es la probabilidad) y qué significa una probabilidad para un solo cliente. Claude
cambió su recomendación: el indicador debe ser la probabilidad calibrada por 100, sin una
segunda escala estirada a 40 y 70. Alejandro aún no eligió los cortes; quedan provisionales:
revisión al 2 % de legítimos (probabilidad 0,03) y retención al 0,1 % (0,67).

**Modo `--final` de `31_train.py`.** Entrena con todas las personas (sin las 3 excluidas), sin
parte de validación, un número fijo de épocas: la mediana de las épocas conservadas por las
cinco partes de la corrida de referencia. Para EfficientNet-B0 con paciencia 6 son 13 en las
dos caras (3 de la capa final y 10 de ajuste; frente [7, 19, 8, 13, 21], reverso
[12, 11, 23, 14, 13]). Probado con 300 personas; el modelo final completo no se ha entrenado.
