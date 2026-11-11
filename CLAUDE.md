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

## Estado actual (2026-11-11) — qué existe y qué NO

- **Existe:** el paquete instalable `id_fraud_detector` (`src/id_fraud_detector/`, instalado con
  `pip install -e ".[review]"`) con las consultas y el etiquetado (`dataset.py`), el motor de
  reglas estructurales (`rules/`) y el recorte con OpenCV (`crop.py`); los scripts de la fase 1
  (`10` a `13`) y de la fase 2 (`20`) en `scripts/`; y pruebas en `tests/`
  (`python -m unittest discover tests`). Todo se probó con datos sintéticos y con una base y un
  S3 simulados.
- **Corrido contra `dev`:** el 11-nov-2026 Alejandro corrió las verificaciones previas con
  `STAGE=dev` (tenant `demo`) y terminaron sin error: la conexión por SSO y Secrets Manager y las
  tres consultas (clientes, OCR, registro común) funcionan. `demo` es muy pequeño (46 clientes
  desde 2025, 4 negativos, ningún positivo), así que solo valida que el código corre.
- **NO existe todavía:** el dataset. **Nada se ha corrido contra `live`.**
- **Siguiente paso:** Alejandro corre `scripts/10_check_database.py` con `STAGE=live` y comparte
  la salida (solo conteos). Con esos números se deciden los pendientes de `docs/plan.md` §5.4
  punto 3.
- **NO existe:** ningún modelo entrenado, ningún notebook de entrenamiento, ninguna integración
  con `mikro_back`. Eso es fase 3 en adelante.
- **Pendiente de decisión del usuario:** dónde se entrena (máquina local, AWS de la empresa o
  Colab Pro con datos cifrados). No asumas ninguna.

## Qué es la solución (resumen; detalle en docs/plan.md §3 y §4)

Dos componentes que alimentan un **indicador de riesgo de falsificación 0–100** por cliente:

1. **Motor de reglas estructurales** sobre el texto del reverso de la cédula, que ya está
   extraído por Amazon Rekognition y guardado en la base (`data_back_id`). Reglas deterministas:
   registrador firmante vs. fecha de producción, número del frente = número del código, sexo,
   producción ≥ expedición y ≥ 18 años, formato del código, meses de 3 letras, reverso repetido
   entre personas distintas. **Cada regla devuelve PASS / FAIL / NOT_EVALUABLE; solo FAIL suma.**
2. **Clasificador de imágenes por transferencia de aprendizaje** (MobileNetV3 o EfficientNet
   preentrenada en ImageNet, afinada), dos instancias de la misma receta: frente y reverso, sobre
   la cédula recortada y enderezada a 384–512 px. Para lo visual: foto pegada, holograma,
   calidad de impresión, plantillas repetidas. Grad-CAM para explicar al analista.

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
- **NEGATIVE (cédula legítima):** sin `fraud_status` y con al menos un crédito `PAID`.
- **Fuera del dataset:** `VERIFICATION REQUIRED`, `VALIDATION REQUEST DENIED`, los demás
  `POSSIBLE FRAUD` (incluidos ~170 posteriores al 4-oct-2026: el usuario decidió NO incluirlos),
  y clientes sin marca y sin crédito pagado.
- Solo registrados desde **2025-01-01**. Conteos reales: ~399 positivos con ambas fotos
  (281 + 78 esmio, 40 llevaloya) y ~28.000 negativos (16.011 esmio, 12.075 llevaloya).
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
