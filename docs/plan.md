# Detector de cédulas falsas — Plan final del proyecto de grado

**Autor:** Alejandro Trujillo López · **Versión final:** 2026-11-09 · **Última actualización:** 2026-11-11
**Fuentes:** carpeta "Proyecto de grado", notebooks de Deep Learning, repositorios `mikro_back` y `mikro_front`, conteos ejecutados en `sisteya_live` por Alejandro.

---

## 1. En una página

**Problema.** La validación de identidad de la app de microcréditos comprueba que la foto de la cédula tenga texto correcto, parezca un documento y el rostro coincida con la selfie. Las cédulas falsificadas que ha visto la empresa pasan todas esas pruebas, porque son composiciones digitales bien hechas: el estafador pone su propia cara sobre una plantilla con los datos de una víctima real. Hoy eso solo lo detecta un analista humano mirando la foto, y no siempre hay uno disponible.

**Solución.** Dos componentes que alimentan un **indicador de riesgo de falsificación de 0 a 100**: un **motor de reglas estructurales** determinista sobre el texto del reverso (registrador vs. fecha, coherencia del código, lista negra de reversos clonados), validado estadísticamente con los datos históricos, y un **clasificador de imágenes** para lo visual. Se construye con **transferencia de aprendizaje**: una red convolucional ya entrenada en millones de imágenes (MobileNetV3 o EfficientNet) a la que se le reemplaza la última capa y se afina con las cédulas reales y falsas de SisteYa. Antes de entrar a la red, la foto pasa por un preprocesamiento clásico con OpenCV que recorta y endereza la cédula para que la red vea solo el documento. El indicador se integra al motor de puntaje de fraude que ya existe, primero en modo observación.

**Datos.** ~399 cédulas de fraude confirmado sin crédito pagado (281 FRAUD + 78 de la red de sep–oct 2026 en ESMIO, 40 en Llévalo Ya) y unas 28.000 cédulas de clientes que pagaron al menos un crédito, todas con frente y reverso, registradas desde 2025. Las fotos ya están en S3; las etiquetas ya están en la base.

**Resultado esperado.** Un prototipo que, sobre los fraudes históricos, muestre cuántos habría enviado a revisión y a qué costo en falsos positivos, con una explicación visual (mapa de calor) de por qué cada cédula es sospechosa.

---

## 2. Qué pasa hoy con la foto de una cédula

Cuando un cliente se registra en la app, toma tres fotos con la cámara del celular: frente de la cédula, reverso y selfie con prueba de vida. El front (`mikro_front`) muestra una caja con la forma de una cédula como guía visual, pero no obliga a nada: captura el cuadro completo de la cámara, lo convierte a PNG y lo envía. El backend (`mikro_back`) la sube a S3 y le pide a Amazon Rekognition tres cosas: etiquetas (¿es un documento?), texto (¿aparecen el nombre y el número?) y rostros (¿hay una cara en el frente y ninguna en el reverso?). Luego compara el rostro de la cédula con la selfie.

El análisis interno de octubre de 2026 (`docs/fraud-score/01-problema.html`) mostró que ninguna de esas pruebas separó el fraude: el número coincidía en el 100 % de los casos, la similitud facial era casi perfecta y la prueba de vida pasaba. Lo único que delató las falsificaciones fue que el reverso repetía los mismos números de serie entre distintos "clientes" (plantilla reutilizada). El modelo propuesto busca aprender esas y otras señales visuales directamente de las imágenes.

---

## 3. La solución

### 3.1 Arquitectura

```
                    ┌──────────────────────────────────────────────────────────┐
  foto frente ─────►│ 1. Chequeos de metadatos (dimensiones, formato, cámara)  │──► señal_meta
  foto reverso ────►│                                                          │
                    └──────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │ 2. Recorte y enderezado (OpenCV: bordes → contorno →      │
                    │    4 esquinas → homografía → cédula rectangular 1000×630) │
                    └──────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                    ┌──────────────────────────────────────────────────────────┐
  OCR del reverso ─►│ 2b. Reglas estructurales sobre el texto ya extraído por  │──► señal_reglas
  (ya guardado)     │     Rekognition: PASS / FAIL / NOT_EVALUABLE (§3.1b)     │
                    └──────────────────────────────────────────────────────────┘
                                   │                        │
                                   ▼                        ▼
                    ┌────────────────────────┐  ┌────────────────────────┐
                    │ 3a. Red FRENTE          │  │ 3b. Red REVERSO         │
                    │ MobileNetV3/EfficientNet│  │ misma receta            │
                    │ afinada → p_frente      │  │ → p_reverso             │
                    │ + Grad-CAM              │  │ + Grad-CAM              │
                    └────────────────────────┘  └────────────────────────┘
                                   │                        │
                                   ▼                        ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │ 4. Combinación y calibración → RIESGO 0–100              │
                    │    umbrales: < 40 normal · 40–70 revisión · > 70 rechazo │
                    └──────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │ 5. Guardado en CustomerIdValidationResult.data_doc_risk  │
                    │    y regla FV_DOC_IMAGE_RISK del motor de fraude (SHADOW)│
                    └──────────────────────────────────────────────────────────┘
```

### 3.1b Segundo componente: motor de reglas estructurales (agregado 2026-11-09)

Origen: el "Verificador de Cédulas" del equipo de verificación manual (artifact de Claude) y el Excel de firmas de registradores. De ese material se toma **solo lo que es verificable de forma determinista o contra fuentes públicas**; lo que depende del juicio de un LLM, de pocos casos o de pesos de criterio se deja por fuera.

**Reglas que entran** (corren sobre el texto del reverso que Rekognition ya guarda en `data_back_id.detected_text`, sin modelo ni LLM):

| Regla | Qué verifica | Por qué entra |
|---|---|---|
| Registrador vs. fecha | El registrador cuyo nombre va impreso bajo la firma del reverso debe ser el que estaba en el cargo en la **fecha de producción** (AAAAMMDD del código). Regla direccional: si la fecha de expedición es posterior al periodo del firmante, es fraude; si es anterior, es normal (duplicados y reimpresiones llevan la firma de quien estaba al producirlas). | Verificable con fuentes públicas. Tabla: Iván Duque Escobar 6-dic-1999 → 24-ago-2002 · Alma Beatriz Rengifo 25-ago-2002 → dic-2006 · Juan Carlos Galindo 1-ene-2007 → dic-2007 · Carlos Ariel Sánchez 6-dic-2007 → dic-2015 · Juan Carlos Galindo 4-dic-2015 → dic-2019 · Alexander Vega Rocha 6-dic-2019 → dic-2023 · Hernán Penagos desde dic-2023. Único punto por confirmar con la Registraduría: el día exacto del cambio Vega → Penagos (el Excel dice 21-nov-2023; la prensa, 6-dic-2023); mientras tanto, la quincena 21-nov a 6-dic-2023 se trata como zona gris y no penaliza. |
| Número del frente = número del código del reverso | Los 10 dígitos del código (con ceros a la izquierda) deben coincidir con el número impreso en el frente. | Determinista. |
| Sexo coherente | La letra M/F del código debe coincidir con el sexo impreso en el reverso. | Determinista. |
| Producción ≥ expedición y ≥ 18 años | La fecha de producción del código no puede ser anterior a la fecha de expedición ni anterior a los 18 años del titular. | Determinista. |
| Formato del código | 1 letra (P, A o R) · lugar de 7 dígitos · consecutivo de 8 · M/F · número de 10 · fecha de 8. Longitudes distintas = código mal armado. | Determinista. Se valida primero sobre los 28.000 negativos para confirmar que el formato es universal en el periodo cubierto. |
| Meses de 3 letras | ENE, FEB, MAR, ABR, MAY, JUN, JUL, AGO, SEP, OCT, NOV, DIC. (En la cédula digital se permiten 4 letras.) | Determinista. |
| Reverso repetido entre personas distintas | Mismo consecutivo con **otra fecha**, o misma serie o mismo número final, en clientes con número de cédula distinto. Excepción "mismo lote": mismo consecutivo **y misma fecha** con series cercanas (diferencia < 50.000) y finales distintos es normal (cédulas producidas el mismo día). | Es la señal que más fraudes atrapó en el análisis interno (seriales de plantilla: 14 y 13 casos) y ya existe en forma parcial como `IdBackSerial`. Se calcula sobre la base completa de clientes, no sobre una lista fija. Los valores concretos de la "lista negra" del Verificador se cargan como semilla y se confirman buscándolos en la base. |
| Código de barras PDF417 (cuando se lee) | Serie, número de cédula, fecha de nacimiento y sexo decodificados del código de barras deben coincidir con lo impreso; la plantilla de huella contenida en el código no debe repetirse en personas distintas. | Determinista si el código es legible (zxing). Se mide primero en qué porcentaje de las fotos se puede leer; si es bajo, queda como control oportunista. |
| Cédula digital: dígitos de control de la MRZ | Los dígitos de control de las tres líneas de la MRZ siguen el estándar ICAO 9303 (pesos 7-3-1); NUIP, nacimiento, sexo y expiración de la MRZ deben coincidir con el frente. | Estándar público. |

**Legibilidad del texto (criterio de diseño, 2026-11-09).** En producción habrá cámaras malas y texto que no se lee. Las reglas se diseñan para abstenerse, no para penalizar:
- Cada regla devuelve **pasa / falla / no evaluable**. Solo "falla" suma puntos. "No evaluable" se registra como métrica de cobertura y el caso se decide con la red convolucional (entrenada con aumentos de desenfoque y baja resolución) y las reglas que sí se evaluaron.
- Una regla se evalúa solo si sus campos superan un umbral de **confianza del OCR** (Rekognition la entrega por fragmento) y encajan en la **forma esperada** (expresión regular del código). Si lo leído no encaja, es "no leído", no "mal formado".
- Normalización de confusiones típicas del OCR antes de comparar (O/0, I/l/1, S/5, B/8) y tolerancia a un carácter cuando hay redundancia.
- **Redundancia**: el número de cédula aparece en el frente, en el código del reverso y en el código de barras; se evalúa con dos de tres lecturas coincidentes. El registrador se fija con un apellido completo legible. 
- **OCR dirigido**: tras el recorte y enderezado, la franja del código del reverso se recorta, se amplía y se pasa de nuevo por OCR (Rekognition o Tesseract/PaddleOCR). Segundo intento sobre la zona conocida, no un segundo sistema.
- **Medir primero**: sobre los 28.000 negativos, porcentaje de legibilidad de cada campo con el OCR ya guardado; eso define la cobertura real de las reglas. Revisar si la ilegibilidad correlaciona con fraude: solo entonces podría ser señal débil.
- Opción de producto (fuera de la tesis): si el código no se lee en una validación en vivo, la app pide repetir la foto del reverso, como hoy repite cuando no detecta rostro.

**Taxonomía de modalidades** para la revisión visual de los ~399 positivos (tomada de las familias del Verificador, útil como lista de categorías aunque sus conteos no se usan): reverso clonado completo · reverso armado con segmentos de cédulas reales · datos reales con foto reemplazada · foto pegada sobre plantilla · recaptura de pantalla o impresión · documento real con identidad robada · otro.

**Validación antes de usar cualquier regla**: cada una se ejecuta sobre los ~28.000 negativos (tasa de falsos positivos) y sobre los ~399 positivos (tasa de detección), con el mismo método de `docs/fraud-score/01-problema.html`. Los pesos del indicador salen de esas tasas, no del catálogo. Una regla con falsos positivos altos se baja a "informativa" o se elimina.

**Precisiones de implementación (2026-11-11)**, todas para cumplir que una lectura rota del OCR sea "no evaluable" y nunca "falla":
- *Número*: se comparan todos los números legibles del frente contra el del código. Si lo leído en el frente es un trozo del número del código (el OCR partió el número), la regla se abstiene.
- *Producción*: la cédula amarilla imprime dos fechas; la más antigua es el nacimiento y la otra la expedición, sin depender del orden en que el OCR las entregue. Con un número distinto de fechas legibles, la regla se abstiene.
- *Meses*: un mes a una sola letra de distancia de uno válido (p. ej. `MAH`) se trata como error de lectura y la regla se abstiene; falla solo con meses de 4 letras o que difieren en dos o más letras.
- *Registrador*: los apellidos se buscan como palabra completa. "VEGA" solo no basta, porque también es lugar de nacimiento o expedición ("LA VEGA", "VEGACHÍ"); ese registrador se fija con "VEGA ROCHA", "ALEXANDER VEGA" o "ROCHA".
- *Formato del código*: en la práctica mide cobertura; un código que no encaja en la forma no se puede distinguir de un OCR roto, así que nunca da "falla".

**Lo que se excluye y por qué**: la línea de tiempo del consecutivo y de la serie (33 y 18 puntos de referencia, márgenes arbitrarios ya corregidos varias veces); las variantes de plantilla por época (tildes, posición de "NUMERO"), inferidas de pocos casos y con excepciones abiertas; los juicios visuales del LLM (foto pegada, huella empastada, holograma plano, firma con lapicero, impresión casera), que no son reproducibles y que en este proyecto le corresponden a la red convolucional; la estimación de edad por rostro (error de ±5–8 años, ajustada tras falsos positivos); las reglas de datos de contacto (fuera del alcance de imagen); los pesos y umbrales del catálogo; y la comparación de la imagen de la firma del registrador (el nombre va impreso debajo y lo lee el OCR; en las composiciones firma y nombre vienen juntos de la plantilla, lo que falla es la fecha).

**Nota de privacidad**: el artifact publicado contiene números de cédula y nombres reales en el catálogo y en la memoria compartida; debería anonimizarse antes de seguirlo compartiendo.

### 3.2 Por qué este diseño y no otro

**Un solo tipo de modelo, aplicado a las dos caras.** Las falsificaciones conocidas son una sola modalidad (composición digital sobre plantilla), así que un clasificador supervisado es lo adecuado. Frente y reverso se tratan con dos instancias de la misma receta porque son documentos visualmente distintos y las señales de fraude de cada uno son diferentes (la foto pegada en el frente, el serial de plantilla en el reverso).

**Transferencia de aprendizaje, no una red desde cero.** Con ~399 positivos no es posible entrenar una red convolucional desde cero; los pesos preentrenados en ImageNet ya saben reconocer bordes, texturas y formas, y solo hay que enseñarles qué distingue una cédula falsa. Se congela la base, se entrena la cabeza, y luego se liberan las últimas capas con una tasa de aprendizaje baja (receta del Taller 1 con MobileNetV3).

**Recorte clásico, no una red detectora.** La cédula siempre está cerca del centro, casi horizontal y ocupa buena parte del cuadro (por la guía visual de la app), así que detectar su contorno con OpenCV es suficiente. Se mide en una muestra de 50 fotos por tenant; si recorta bien menos del 90 %, se entrena YOLO (Taller 2) como respaldo. El recorte importa porque sin él la red podría aprender el fondo (mesa, mano) en vez del documento.

**Lo que se descartó y por qué.** El autoencoder para anomalías (en fotos reales el error de reconstrucción sigue al desenfoque y la luz, no a la falsedad; funciona en MNIST, no aquí). El ELA (análisis de recompresión JPEG: la captura es un PNG generado en canvas, sin JPEG). Un fusor complejo de varios modelos (con una sola receta, la probabilidad calibrada es el indicador). Quedan como trabajo futuro: análisis por zonas del documento, rama de residuo de ruido, y comparación de embeddings del serial entre clientes.

### 3.3 Detalles de entrenamiento

- Entrada: cédula rectificada a **384–512 px** de ancho (más que los 224 habituales, para no perder tipografía y bordes finos).
- Aumento de datos conservador: brillo, contraste, desenfoque leve, rotación de ±5°, compresión. **Nunca** volteos ni recortes agresivos, porque destruyen la señal.
- `class_weight` para el desbalance; muestra de 3.000–5.000 negativos variados en mes y canal para entrenar; `ModelCheckpoint` + `EarlyStopping` sobre PR-AUC de validación.
- Línea base de comparación: la misma red sobre la foto completa sin recortar, a 224 px. Sirve para demostrar en la tesis cuánto aporta el preprocesamiento.
- Calibración de la probabilidad (Platt o isotónica) sobre el conjunto de validación, para que "70" signifique lo mismo en los dos tenants.

---

## 4. Qué se le hace a un cliente nuevo (flujo de inferencia)

Cuando un cliente termina de tomar las fotos, el sistema haría lo siguiente, en orden, en unos pocos segundos:

**Paso 1 — Chequeos baratos sobre el archivo.** Antes de mirar los píxeles se revisan datos del archivo: ancho y alto (una captura legítima tiene dimensiones de cámara, como 1280×720 o 1920×1080; un archivo subido a mano saltándose la app tendría medidas arbitrarias), formato (la app genera PNG), tamaño y si se usó la cámara frontal. Nada de esto decide solo, pero una anomalía aquí suma al riesgo y queda registrada.

**Paso 2 — Encontrar y enderezar la cédula.** Con OpenCV se buscan los bordes, se toma el cuadrilátero más grande con proporción de cédula (≈1,58:1) y se transforma la perspectiva para que quede como si estuviera escaneada: rectangular, derecha, siempre del mismo tamaño. Si no se encuentra un contorno válido, la foto se marca "no analizable" y pasa a revisión humana (eso ya es información: una foto donde no se distingue el documento no debería aprobarse sola).

**Paso 2b — Reglas estructurales sobre el texto del reverso.** Con el OCR que Rekognition ya extrajo se verifica: que el número del código coincida con el del frente, que el sexo coincida, que la fecha de producción no sea anterior a la expedición ni a los 18 años, que los meses tengan 3 letras, que el código tenga el formato 1-7-8-M/F-10-8, que el registrador impreso corresponda a la fecha de producción, y que consecutivo, serie o número final no se repitan en otro cliente con cédula distinta (salvo "mismo lote"). Si el código de barras se lee, se compara con lo impreso. Cada regla suma puntos según su tasa de falsos positivos medida en los datos históricos; las que resultan con falsos positivos casi nulos (número distinto, sexo distinto, registrador imposible, reverso repetido) llevan el caso directo a revisión o retención.

**Paso 3 — Análisis del frente.** La cédula rectificada entra a la red del frente, que devuelve una probabilidad de falsificación. Lo que la red aprende a mirar, según lo que se ve en los fraudes conocidos: discontinuidades alrededor de la fotografía del titular (borde del pegado, iluminación de la cara distinta a la del resto del documento, textura de la foto más lisa o más nítida que el plástico), tipografía o alineación de los datos que no corresponde a la del documento oficial, ausencia o irregularidad del holograma y los microtextos, y repetición de fondos o elementos idénticos entre documentos de distintas personas. Grad-CAM genera un mapa de calor sobre la cédula señalando qué región pesó más en la decisión.

**Paso 4 — Análisis del reverso.** La red del reverso hace lo mismo sobre la zona de huella, firma, código de barras y seriales. En los fraudes conocidos el reverso venía de una plantilla reutilizada, así que la red aprende a reconocer esas plantillas y las anomalías de impresión asociadas. También produce su mapa de calor.

**Paso 5 — Riesgo final.** Las dos probabilidades y las señales del paso 1 se combinan (promedio ponderado o regresión logística pequeña) y se calibran a una escala 0–100. Con umbrales iniciales: menos de 40 sigue el flujo normal, 40–70 va a validación manual con los mapas de calor adjuntos, más de 70 se retiene. Los umbrales se fijan sobre datos históricos para que la franja de revisión no supere el ~2 % de clientes legítimos (hoy la política de fraude manda a revisión el 1,6 %).

**Paso 6 — Registro e integración.** El riesgo, las dos probabilidades, los mapas de calor y las señales de metadatos se guardan en `CustomerIdValidationResult` (nuevo campo JSON `data_doc_risk`). El motor de puntaje de fraude lee ese valor como una regla más (`FV_DOC_IMAGE_RISK`, con puntos según la franja). Arranca en modo **SHADOW**: calcula y registra pero no bloquea; cuando se compruebe en producción que los que marca son fraude, pasa a ENFORCE. El analista, en el panel de revisión, ve la cédula con el mapa de calor y el riesgo, y su decisión final (fraude / legítimo) alimenta el reentrenamiento.

---

## 5. Datos

### 5.1 Dónde están
S3, bucket `sisteya-media`, una copia por entorno y tenant. La copia histórica (conserva todas las versiones): `{live}/{tenant}/clients/front_ids/front_id_image_{id}.png` y `back_ids/back_id_image_{id}.png`, referenciadas desde `users_client.front_id_image` / `back_id_image`. La copia cruda de la validación (`clients/id-validations/frontal-id-photos/{documento}.png`) se sobreescribe en cada intento y solo guarda la última. Los resultados de Rekognition están en `users_customeridvalidationresult.data_front_id` / `data_back_id`.

### 5.2 Reglas de etiquetado (acordadas)
- **Positivo (cédula falsa):** (a) `fraud_status = 'FRAUD'` y ningún crédito en estado `PAID` (quien pagó un crédito no fue fraude por foto); o (b) `fraud_status = 'POSSIBLE FRAUD'`, tenant `esmio`, registro entre **2026-09-13 y 2026-10-04**, sin crédito `PAID`: la red de la tienda 623 cuyos 65 documentos confirmó el equipo (`docs/fraud-score`). **No se cambia la marca en la base**; la ventana se pasa como parámetro al comando de exportación y se verifica con una consulta que el conteo coincida con los 65 (decisión 2026-11-09).
- **Negativo (cédula legítima):** sin `fraud_status` y al menos un crédito `PAID`.
- **Fuera del dataset:** `POSSIBLE FRAUD`, `VERIFICATION REQUIRED`, `VALIDATION REQUEST DENIED` (no se sabe si fue por foto) y clientes sin créditos pagados ni marca.
- Solo tenants `esmio` y `llevaloya`, registrados desde 2025-01-01.

### 5.3 Conteos (ejecutados el 2026-11-09)

| Tenant | Positivos (con ambas fotos) | Negativos (con ambas fotos) | Excluidos otro estado |
|---|---|---|---|
| esmio | 296 (**281**) | 16.014 (16.011) | 13.936 |
| llevaloya | 45 (**40**) | 12.077 (12.075) | 13.300 |

Positivos de ESMIO por mes: dos olas (abr–jun 2025 y feb–may 2026) más un goteo constante; sep–nov 2026 solo 8.

### 5.4 Decisiones pendientes sobre los datos
1. **DECIDIDO y VERIFICADO (2026-11-09): la red de sep–oct 2026 (ESMIO, tienda 623, `docs/fraud-score/01-problema.html`) entra como positivos por ventana de fechas — POSSIBLE FRAUD, esmio, registro 2026-09-13 a 2026-10-04, sin crédito pago — sin modificar la marca en la base.** La consulta por semana dio 14 + 16 + 49 = **79 clientes (78 con ambas fotos)**, consistente con los 65 confirmados + 16 marcados durante la investigación. Este bloque entra ya clasificado por modalidad: composición digital con reverso de plantilla reutilizada. Positivos totales con ambas fotos: 281 + 78 + 40 ≈ **399**.
   - **Reserva, no incluidos:** ~160 POSSIBLE FRAUD de la app entre el 5-oct y el 8-nov de 2026 (32/37/44/25/21 por semana) + 9 de comercio. Mezcla probable de red que siguió operando y de clientes retenidos por la política de fraude en ENFORCE desde el 20-oct (con ~1,6 % de legítimos esperados; la proporción con ambas fotos cae a la mitad en las últimas semanas). Se revisan en diciembre cuando venzan las primeras cuotas, o antes si el equipo entrega lista de confirmados; podrían sumar ~100 positivos más. Los 3 de finales de agosto / inicio de septiembre quedan fuera.
2. **Revisión visual de los ~399 positivos** (frente y reverso), clasificando cada uno en: composición digital / recaptura de pantalla o impresión / documento real con identidad robada / otro. Un par de horas de trabajo. Define qué aprende el modelo, permite reportar recall por modalidad, y los "documento real" pasan a negativos difíciles.

3. **Hallazgos en `mikro_back` pendientes de medir y decidir (2026-11-11).** No cambian ninguna decisión todavía; `scripts/10_check_database.py` entrega los números para decidir.
   - **Fotos copiadas entre marcas.** Cuando una persona ya validada en una marca se registra en otra, `validate_from_common_client` copia sus fotos desde el registro común sin volver a pasar por Rekognition. La misma foto puede estar en ESMIO (entrenamiento) y en Llévalo Ya (prueba externa), y ese cliente no tiene OCR en la segunda marca. El manifest incluye `person_id` (igual en todas las marcas) y el MD5 de cada imagen para detectarlo. Por decidir: qué hacer con esos clientes en la prueba externa.
   - **Marca de fraude en el registro común.** `Client.get_fraud_status()` lee primero `CommonClient.fraud_status` (esquema `public`) y después la del tenant. El etiquetado usa la del tenant. El manifest incluye `common_fraud_status`. Por decidir: si un negativo con marca en el registro común sigue siendo negativo.
   - **Créditos `NOVATED`.** Un crédito que se salda con un pago marcado como BNPL (`Credit.make_payment(..., is_bnpl=True)`) queda `NOVATED`, no `PAID`; varios reportes de `mikro_back` cuentan ambos estados como pagados. La regla vigente solo cuenta `PAID`. Por decidir si cambia, según cuántos clientes afecte.
   - **Línea base.** La regla de serial compartido ya está en producción (FR01, `users/utils/id_back_serial_utils.py`). Por decidir: incluirla en la línea base de §6 además de las reglas de Rekognition.
   - **Partición.** Separar por cliente no impide que la misma plantilla de reverso quede en entrenamiento y en validación. Por decidir: agrupar por plantilla (consecutivo o serie compartidos) al partir.
   - **Reverso repetido en el backtest.** `12_ocr_coverage.py` lo calcula sobre todo el dataset a la vez; en producción solo se puede comparar contra clientes anteriores. El backtest de la fase 5 debe calcularlo así.

### 5.5 Particiones
- Entrenamiento y validación: ESMIO, separado **por cliente y por tiempo** (entrenar con 2025, validar con 2026) para que el modelo no aprenda el celular o la fecha de una ola concreta.
- Prueba externa: **Llévalo Ya completo** (40 positivos, 12.075 negativos), nunca visto en entrenamiento. Demuestra que generaliza a otra marca, otros comercios y otros dispositivos.
- Evaluación de falsos positivos sobre **todos** los negativos, no sobre la muestra de entrenamiento.

### 5.6 Privacidad
Datos personales sensibles (Ley 1581 de 2012). El dataset se construye con identificadores anonimizados (hash del documento), se trabaja sobre los recortes de la cédula, en almacenamiento con acceso restringido, y nunca se suben fotos a servicios públicos (Colab abierto, Roboflow público). Queda documentado en la tesis.

---

## 6. Evaluación

- Métricas para clases desbalanceadas: precisión, recall y F1 de la clase "falsa", PR-AUC, ROC-AUC. **La métrica principal es el recall a una tasa de falsos positivos fija (≤2 %)**, porque eso es lo que determina cuántos clientes legítimos van a revisión manual. Accuracy no se reporta como métrica principal.
- Matriz de confusión desagregada por tenant, por tipo de cédula (amarilla con holograma vs. digital de policarbonato) y por modalidad de fraude (de la revisión visual).
- **Backtest histórico**: sobre los fraudes confirmados, cuántos habrían caído en revisión o rechazo y cuánto capital se habría evitado, con el mismo método que ya usa la empresa en `05-backtest-esmio.html`. Es el argumento de valor para la sustentación.
- Comparación con dos líneas base: las reglas actuales de Rekognition, y la red sobre la foto completa sin recorte.
- Explicabilidad: mapas Grad-CAM de aciertos y errores, revisados con un analista para confirmar que la red mira el documento y no el fondo.

---

## 6b. Repositorio del proyecto (creado 2026-11-09)

Carpeta `Proyecto de grado/id_fraud_detector/`, independiente de `mikro_back` (se convertirá en repositorio). Código en inglés, documentación en español. Incluye `CLAUDE.md` con el contexto para Claude Code y `docs/plan.md` (copia de este documento) para trabajar desde VS Code. Lee la base en solo lectura y S3 con boto3; no usa Django. No guarda credenciales: usa la sesión de SSO de AWS, toma el acceso a la base del secreto `sisteya_db_proxy_credentials` de Secrets Manager y elige el entorno con `STAGE` en `.env` (base `sisteya_<STAGE>`, salidas en `data/<STAGE>/`). Pendiente: dónde se entrena (máquina local, AWS o Colab Pro).

**Un solo repositorio para todo el proyecto (decisión 2026-11-11).** Datos, reglas, clasificador, evaluación, indicador y servicio de inferencia viven aquí, porque el recorte y las reglas deben ser exactamente el mismo código al entrenar, al evaluar y al servir. Lo único que queda fuera son los cambios dentro de `mikro_back` de la fase 6 (campo `data_doc_risk` y regla `FV_DOC_IMAGE_RISK`), que viven allá y consumen lo que este repositorio expone.

```
id_fraud_detector/
  pyproject.toml            nombre del paquete y dependencias por grupo
  config.yaml               reglas de etiquetado, tenants por entorno, muestreo
  docs/
    plan.md, context.md     qué se hace y por qué; historia de las decisiones
    results/                (por crear) un informe por fase, solo agregados: insumo de la tesis
  src/id_fraud_detector/    paquete instalable con toda la lógica reutilizable
    common.py               conexión de solo lectura, anonimización            [existe]
    dataset.py              fase 1: consultas, etiquetado, muestreo            [existe]
    rules/                  fase 1: registradores, parser del OCR, motor       [existe]
    crop.py                 fase 2: recorte y enderezado con OpenCV            [existe]
    model/                  fase 3: carga de datos, red, entrenamiento, Grad-CAM
    evaluation/             fases 3 y 5: particiones, métricas, backtest
    risk.py                 fase 5: combinación y calibración a 0–100
    pipeline.py             fase 5: una función, fotos + OCR -> riesgo
  scripts/                  puntos de entrada delgados; el primer dígito es la fase
    10_check_database.py    11_export_dataset.py    12_ocr_coverage.py    13_visual_review.py
    20_crop_test.py
    3x_*.py                 fase 3 (particiones, entrenamiento, evaluación)
    5x_*.py                 fase 5 (calibración, backtest)
  notebooks/                (por crear) solo exploración, sin lógica
  tests/                    pruebas con datos sintéticos
  service/                  fase 6: handler y Dockerfile del servicio de inferencia
  infra/                    fase 6: infraestructura como código del servicio
  data/<STAGE>/             nunca en git
  models/, runs/            nunca en git
```

Reglas de organización:
- **Paquete instalable.** `pip install -e ".[review]"`; todo se importa como `from id_fraud_detector...`. Los scripts no contienen lógica reutilizable.
- **Dependencias por grupo** en `pyproject.toml`: base (datos, reglas, recorte), `review` (Streamlit) y, cuando lleguen sus fases, `train` y `serve`. El contenedor de inferencia instala solo lo mínimo.
- **Una sola función de inferencia** (`pipeline.py`): es lo que se evalúa en el backtest y lo que se despliega.
- **Particiones como archivo**: se generan una vez (`splits.csv`) y todo lo demás las lee.
- **Cada entrenamiento deja rastro** en `runs/<id>/`: configuración, semilla, commit y métricas. No hace falta MLflow ni DVC para este tamaño (MLOps nivel 0, §7).
- **Las carpetas se crean cuando empieza su fase**, no antes. La de infraestructura depende de dónde se entrene y se sirva, que sigue pendiente.

## 7. Plan por fases

| Fase | Duración | Entregable | Apoyo de clase |
|---|---|---|---|
| 1. Dataset | 2–3 sem | Verificaciones previas (`10_check_database.py`); exportación con `11_export_dataset.py` (~399 pos, muestra de neg con imágenes para entrenar, todos los neg con OCR para evaluar las reglas); manifest anonimizado; medición de las reglas (`12_ocr_coverage.py`); **revisión visual por modalidad**; exploración con embeddings + t-SNE para ver si los fraudes se agrupan | Clase 4 (embeddings), Gatos/Perros (carga por carpetas) |
| 2. Recorte | 1 sem | Contorno + homografía con OpenCV; medir % de recorte correcto en 50 fotos por tenant; cola de "no recortadas" | OpenCV; Taller 2 (YOLO) solo si falla |
| 3. Modelo | 3–4 sem | Redes de frente y reverso por transferencia; partición por cliente y tiempo; métricas, Grad-CAM; línea base sin recorte | Taller 1 (MobileNetV3), Gatos/Perros |
| 4. Iteración | 2 sem | Según análisis de errores: resolución, zonas, residuo de ruido, sintéticos para modalidades escasas | — |
| 5. Indicador y backtest | 1 sem | Calibración 0–100, umbrales, backtest histórico, prueba externa en Llévalo Ya | `05-backtest-esmio.html` |
| 6. Integración (opcional) | 2 sem | Servicio de inferencia (Lambda container o SageMaker), campo `data_doc_risk`, regla `FV_DOC_IMAGE_RISK` en SHADOW, mapa de calor en el panel de revisión | `docs/fraud-score/03-plan.html` |

El proyecto de grado queda completo con las fases 1, 2, 3 y 5. Las fases 4 y 6 son mejora e integración; si no alcanzan, se presentan como trabajo futuro. Metodología: CRISP-DM (fases 1–5 corresponden a comprensión de datos, preparación, modelado y evaluación; la 6 a despliegue), con MLOps nivel 0 inicialmente.

---

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| El modelo aprende el dispositivo o la fecha de una ola, no la falsificación | Partición por cliente y tiempo; negativos del mismo período; prueba externa en otro tenant; Grad-CAM revisado |
| Parte de los positivos son documentos reales con identidad robada | Revisión visual previa; esos casos pasan a negativos difíciles |
| El recorte falla en fondos difíciles | Medir en muestra; cola de revisión manual; YOLO como respaldo |
| Dos tipos de cédula con apariencia distinta | Incluir ambos desde el inicio y reportar métricas por tipo |
| El estafador cambia de método | Modo SHADOW con monitoreo; reentrenamiento con lo que marque el analista; trabajo futuro en detección de anomalías |

---

## Anexo A — Hallazgos técnicos del repositorio

- `mikro_back`: Django 5.1 + DRF, multi-tenant por esquema (`django-tenants`), AWS Lambda vía Zappa, S3 con `django-s3-storage` (`AWS_S3_FILE_OVERWRITE=False`, por eso `front_ids/` conserva versiones). Endpoints de captura: `validate_id_labels_front/back` (app), `check_frontal_id_picture` / `check_back_id_picture` (comercio). Validación actual con Rekognition `detect_labels`, `detect_text`, `detect_faces` y umbrales en `IdValidationConfig`.
- `mikro_front`: `react-webcam` con guía visual de proporción 338/213 (app) o imagen `muckupFrontId.png` superpuesta (comercio), sin lógica de encuadre. `doCustomScreenShot` captura el cuadro completo del video a canvas y exporta PNG; muchos celulares ignoran `aspectRatio`, así que la foto trae más fondo del que el usuario ve. Sin EXIF.
- Etiquetas: `users_client.fraud_status` con historial en `users_customerhistoricalfraudstatus`; estados FRAUD, POSSIBLE FRAUD, VERIFICATION REQUIRED, VALIDATION REQUEST DENIED. `IdBackSerial` indexa seriales OCR del reverso para la regla FV_BACK_SERIAL_SHARED_DOCS.

## Anexo B — Consultas de conteo (ya ejecutadas; para repetir)

```sql
SET search_path TO esmio;   -- repetir con llevaloya
WITH pagos AS (
    SELECT customer_id, bool_or(status = 'PAID') AS pago FROM credits_credit GROUP BY customer_id
)
SELECT CASE
         WHEN c.fraud_status = 'FRAUD' AND NOT COALESCE(p.pago, false) THEN 'POSITIVO'
         WHEN c.fraud_status = 'FRAUD'                                 THEN 'EXCLUIDO fraud con pago'
         WHEN NULLIF(c.fraud_status, '') IS NOT NULL                   THEN 'EXCLUIDO otro estado'
         WHEN COALESCE(p.pago, false)                                  THEN 'NEGATIVO'
         ELSE 'SIN USO' END AS grupo,
       COUNT(*) AS clientes,
       COUNT(*) FILTER (WHERE c.front_id_image <> '' AND c.back_id_image <> '') AS con_ambas_fotos
FROM users_client c LEFT JOIN pagos p ON p.customer_id = c.id
WHERE c.terms_and_conditions_timestamp >= '2025-01-01'
GROUP BY 1 ORDER BY 1;
```
