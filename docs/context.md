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
