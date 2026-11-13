# Cifras de los datos — id_fraud_detector

Resumen de los datos disponibles y usados en el proyecto, para el informe y la presentación.
Corte: **12 de noviembre de 2026**, entorno `live`, marcas ESMIO y Llévalo Ya.

Todo son conteos y tasas; no hay números de cédula, nombres ni texto de los documentos.

**Cómo regenerarlo.** `python scripts/17_data_summary.py` lee solo archivos locales y deja una
tabla por tema en `data/live/summary/*.csv` (lista para graficar). Si cambia el dataset, se
vuelve a correr y se actualizan las cifras de este documento. La tabla del universo (§1) sale
de `scripts/10_check_database.py`, que sí consulta la base.

---

## 1. De dónde sale el dataset

Clientes registrados desde el 1-ene-2025 (corrida del 11-nov-2026, antes de agregar las fuentes
confirmadas por el equipo de fraude):

| Grupo | ESMIO | Llévalo Ya | Total |
|---|---|---|---|
| Sin marca de fraude y sin crédito pagado (no se usan) | 113.056 | 110.167 | 223.223 |
| Otra marca (`VERIFICATION REQUIRED`, `POSSIBLE FRAUD` fuera de ventana, etc.) | 13.911 | 13.239 | 27.150 |
| Sin marca y con crédito pagado (negativos) | 16.086 | 12.150 | 28.236 |
| `FRAUD` sin crédito pagado (positivos) | 299 | 45 | 344 |
| `FRAUD` con crédito pagado (excluido) | 0 | 1 | 1 |
| **Total de clientes** | **143.352** | **135.602** | **278.954** |

De cada 1.000 clientes registrados, unos 101 quedan como negativos y poco más de 1 como
positivo. El resto no tiene una etiqueta confiable.

**Reglas de etiquetado**

- **Positivo (cédula falsa):** sin ningún crédito pagado, con ambas fotos, y además una de:
  `FRAUD`; `POSSIBLE FRAUD` de ESMIO registrado entre el 13-sep y el 4-oct de 2026 (red de la
  tienda 623); crédito en el comercio AMORETT entre el 14-oct y el 10-nov de 2026; crédito
  desembolsado a Banco Finandina en un comercio "Libre Destino"; o cédula confirmada como falsa
  por un analista en el verificador del equipo.
- **Negativo (cédula legítima):** sin marca de fraude y con al menos un crédito pagado.
- **Una fila por persona:** quien está en las dos marcas cuenta una sola vez.

## 2. El dataset

**28.293 personas: 410 positivas (1,45 %) y 27.883 negativas.**

| Etiqueta | Personas | En las dos marcas | Con fotos para el clasificador |
|---|---|---|---|
| Positiva | 410 | 22 | 410 |
| Negativa | 27.883 | 402 | 5.297 (muestra) |

Se quitaron 424 filas repetidas (personas etiquetadas en las dos marcas) y no hubo ninguna
persona positiva en una marca y negativa en la otra.

**Positivos por origen de la etiqueta**

| Origen | ESMIO | Llévalo Ya | Total |
|---|---|---|---|
| `FRAUD` | 283 | 19 | 302 |
| Red de la tienda 623 (`POSSIBLE FRAUD` en la ventana) | 79 | 0 | 79 |
| Comercio AMORETT | 17 | 0 | 17 |
| Verificador, confirmada por un analista | 7 | 1 | 8 |
| Finandina + Libre Destino (los demás ya eran positivos) | 4 | 0 | 4 |
| **Total** | **390** | **20** | **410** |

Negativos: 15.921 de ESMIO y 11.962 de Llévalo Ya.

**Negativos con fotos, para el clasificador (5.297).** No son una muestra proporcional: se
emparejaron con los positivos. Entran todos los negativos de app (980), todos los registrados
desde septiembre de 2026 (301) y una muestra por mes y canal del resto.

| Canal | Negativos con fotos | Positivos |
|---|---|---|
| Comercio | 4.197 | 273 |
| App | 980 | 128 |
| WhatsApp | 118 | 9 |
| Sin dato | 2 | 0 |

## 3. Canal de registro

| Canal | Negativos | % | Positivos | % |
|---|---|---|---|---|
| Comercio | 26.186 | 93,9 % | 273 | 66,6 % |
| App | 980 | 3,5 % | 128 | 31,2 % |
| WhatsApp | 715 | 2,6 % | 9 | 2,2 % |
| Sin dato | 2 | 0,0 % | 0 | 0,0 % |

El canal es un confusor: un positivo tiene casi nueve veces más probabilidad de venir de la app
que un negativo, y el canal decide la cámara con que se tomó la foto.

## 4. Personas por mes de registro

| Mes | Negativos | Positivos | | Mes | Negativos | Positivos |
|---|---|---|---|---|---|---|
| 2025-01 | 1.661 | 4 | | 2026-01 | 1.295 | 1 |
| 2025-02 | 1.646 | 7 | | 2026-02 | 1.454 | 10 |
| 2025-03 | 1.966 | 20 | | 2026-03 | 1.290 | 19 |
| 2025-04 | 1.646 | 26 | | 2026-04 | 1.020 | 27 |
| 2025-05 | 1.948 | 37 | | 2026-05 | 835 | 17 |
| 2025-06 | 1.747 | 33 | | 2026-06 | 524 | 1 |
| 2025-07 | 1.775 | 19 | | 2026-07 | 392 | 1 |
| 2025-08 | 1.571 | 16 | | 2026-08 | 306 | 2 |
| 2025-09 | 1.426 | 17 | | 2026-09 | 207 | 55 |
| 2025-10 | 1.699 | 7 | | 2026-10 | 87 | 53 |
| 2025-11 | 1.696 | 17 | | 2026-11 | 7 | 6 |
| 2025-12 | 1.685 | 15 | | | | |

- Los negativos caen en los meses recientes porque un cliente necesita tiempo para pagar un
  crédito completo; no es que haya menos clientes.
- 114 de los 410 positivos (28 %) se registraron entre septiembre y noviembre de 2026: son la
  red de la tienda 623 (79), AMORETT (17) y los demás casos recientes. El tiempo también es un
  confusor.

El detalle de positivos por mes y origen está en `data/live/summary/04_month_positive_sources.csv`.

## 5. Texto que ya estaba guardado en producción

Porcentaje de personas con el texto del reverso guardado por la validación actual:

| Canal | Negativos | Positivos |
|---|---|---|
| App | 85,7 % (840 de 980) | 96,1 % (123 de 128) |
| WhatsApp | 85,3 % (610 de 715) | 77,8 % (7 de 9) |
| Comercio | 9,9 % (2.593 de 26.186) | 12,1 % (33 de 273) |

El flujo de comercio casi nunca guarda el texto. Por eso se volvió a leer con Rekognition.

## 6. Fotos

**En disco:** 11.414 fotos de 5.707 personas, 22,1 GB.

| Etiqueta | Personas | Fotos | GB | Tamaño mediano de una foto |
|---|---|---|---|---|
| Positiva | 410 | 820 | 1,9 | 2,45 MB |
| Negativa | 5.297 | 10.594 | 20,2 | 1,43 MB |

**Tamaño de las fotos según el canal** (ancho × alto en píxeles, frente y reverso juntos):

| Canal | Tamaño más común | 2.º | 3.º |
|---|---|---|---|
| App | 1920×1210 (74 %) | 1280×720 (10 %) | 1280×800 (5 %) |
| Comercio | 1280×720 (54 %) | 1920×1210 (21 %) | 640×480 (11 %) |
| WhatsApp | 1280×720 (40 %) | 1920×1210 (24 %) | 1024×768 (10 %) |

| Canal | Hasta 700 px de ancho | 701 a 1400 px | Más de 1400 px |
|---|---|---|---|
| App | 65 (3 %) | 396 (18 %) | 1.755 (79 %) |
| Comercio | 1.207 (14 %) | 5.230 (59 %) | 2.503 (28 %) |
| WhatsApp | 24 (9 %) | 138 (54 %) | 92 (36 %) |

Los dos flujos piden a la cámara 1920 px de ancho; el navegador entrega lo que la cámara
soporta. En comercio la cámara es la del equipo de la tienda.

## 7. Lectura nueva con Rekognition

Leídas otra vez las dos caras de **5.707 personas**: los 410 positivos y 5.297 negativos.
11.414 fotos, más 200 de la prueba de ampliación: **11,61 USD**, pagados con la cuenta de AWS de
Alejandro, aparte de la corporativa.

**Qué se logra leer** (porcentaje de personas; solo líneas con confianza de 80 o más):

| Etiqueta | Canal | Personas | Número del frente | Código del reverso | Serie | Registrador |
|---|---|---|---|---|---|---|
| Negativa | App | 980 | 93 % | 38 % | 47 % | 69 % |
| Negativa | Comercio | 4.197 | 78 % | 14 % | 26 % | 26 % |
| Negativa | WhatsApp | 118 | 81 % | 30 % | 42 % | 35 % |
| Positiva | App | 128 | 98 % | 58 % | 44 % | 58 % |
| Positiva | Comercio | 273 | 82 % | 22 % | 36 % | 26 % |
| Positiva | WhatsApp | 9 | 89 % | 0 % | 0 % | 0 % |
| **Negativa** | **todos** | **5.297** | **81 %** | **19 %** | **30 %** | **34 %** |
| **Positiva** | **todos** | **410** | **88 %** | **32 %** | **37 %** | **36 %** |

**La lectura depende del ancho de la foto del reverso:**

| Ancho de la foto | Personas | Código del reverso | Serie | Registrador |
|---|---|---|---|---|
| Hasta 700 px | 646 | 6 % | 12 % | 7 % |
| 701 a 1400 px | 2.886 | 10 % | 25 % | 20 % |
| Más de 1400 px | 2.175 | 37 % | 44 % | 61 % |

**Pruebas que acompañaron la lectura**

- **Ampliar la cédula no ayuda.** En 200 personas, recortar la cédula y ampliarla al doble
  subió el código legible de 22,5 % a 23,5 % (6 ganados, 4 perdidos). El límite es la cámara.
- **El filtro de confianza está justificado.** Con confianza de 80 o más, el número del código
  coincidió con el del frente en 152 de 152 lecturas; por debajo de 80, en 77 de 102 (75 %).
  Medido sobre las primeras 681 personas.
- **El sexo impreso no se lee en nadie:** Rekognition entrega la etiqueta y el valor en líneas
  separadas y el patrón actual no encaja. Pendiente.

## 8. Reglas estructurales

Sobre las 5.707 personas leídas. Un fallo en un negativo es un falso positivo; en un positivo
es una detección. "Tope" es la tasa máxima compatible con cero fallos (95 % de confianza).

| Regla | Positivos: fallos / evaluables | Negativos: fallos / evaluables | Falsos positivos |
|---|---|---|---|
| Reverso repetido en otra cédula | 37 / 137 (27,0 %) | 0 / 990 | tope 0,30 % |
| Registrador que no corresponde a la fecha | 19 / 142 (13,4 %) | 1 / 1.330 | 0,08 % |
| Fecha de producción incoherente | 2 / 126 (1,6 %) | 0 / 929 | tope 0,32 % |
| Número del frente distinto al del código | 0 / 129 | 0 / 934 | tope 0,32 % |
| Formato del código | 0 / 133 | 0 / 990 | tope 0,30 % |
| Meses mal escritos | 0 / 374 | 0 / 3.568 | tope 0,08 % |
| Sexo del código distinto al impreso | 0 / 0 | 0 / 0 | sin medir |

**Alcance de las reglas**

| | Positivos (410) | Negativos (5.297) |
|---|---|---|
| Con alguna regla del código evaluable | 175 (43 %) | 1.500 (28 %) |
| Con al menos un fallo | 49 (12,0 %) | 1 (0,02 %) |

**Detección según el origen del positivo**

| Origen | Personas | Evaluables | Con algún fallo |
|---|---|---|---|
| Red de la tienda 623 | 79 | 62 | 26 (33 %) |
| Comercio AMORETT | 17 | 8 | 8 (47 %) |
| Finandina + Libre Destino (nuevos) | 4 | 4 | 1 (25 %) |
| `FRAUD` | 302 | 94 | 14 (5 %) |
| Verificador, confirmada por analista | 8 | 7 | 0 (0 %) |

**Detección según el canal**

| Canal | Positivos | Con algún fallo | Negativos | Con algún fallo |
|---|---|---|---|---|
| App | 128 | 37 (29 %) | 980 | 0 |
| Comercio | 273 | 12 (4 %) | 4.197 | 1 |
| WhatsApp | 9 | 0 | 118 | 0 |

Lectura: las reglas casi no se equivocan con clientes legítimos, pero alcanzan a pocos
positivos. Detectan sobre todo redes que reutilizan reversos (tienda 623, AMORETT) y casi nada
en comercio, donde la foto no deja leer el código. El clasificador de imágenes tiene que cubrir
el resto.

## 9. Otras fuentes revisadas

**Verificador de cédulas del equipo** (840 casos, 700 cédulas distintas, del 19-oct al 12-nov
de 2026):

| Veredicto | Lo puso un analista | Sin registro de quién | Lo puso la herramienta por puntaje |
|---|---|---|---|
| Falsa | 13 | 19 | 102 |
| Auténtica | 19 | 34 | 606 |
| Pendiente | 5 | 13 | 29 |

En cédulas distintas: 10 falsas confirmadas por un analista (las únicas usadas como etiqueta),
15 falsas sin registro de origen, 80 falsas solo por puntaje y 14 auténticas confirmadas.

**Fuentes confirmadas por el equipo de fraude** (ESMIO, 12-nov-2026):

| Fuente | Clientes | Ya eran positivos | Positivos nuevos |
|---|---|---|---|
| Comercio AMORETT, 14-oct a 10-nov de 2026 | 17 | 0 | 17 |
| Desembolso a Finandina en "Libre Destino" | 68 | 64 | 4 |

**Prueba del recorte de la cédula** (OpenCV, 188 fotos): encontró el contorno de la cédula en 4
fotos (2 %). El recorte, como está, no funciona con fotos reales.

## 10. Partición para la validación cruzada

Las 5.707 personas con fotos se repartieron en cinco partes (`data/live/splits.csv`,
`scripts/30_make_splits.py`). Cada persona está en una sola parte, y quienes comparten reverso
quedan juntos.

| Parte | Negativos | Positivos | Positivos de app | Positivos de comercio |
|---|---|---|---|---|
| 0 | 1.061 | 133 | 100 | 32 |
| 1 | 1.059 | 70 | 7 | 61 |
| 2 | 1.059 | 69 | 7 | 60 |
| 3 | 1.059 | 69 | 7 | 60 |
| 4 | 1.059 | 69 | 7 | 60 |

- **La tienda 623 y AMORETT resultaron ser una sola red:** comparten reversos entre sí. Los 79
  de la tienda 623, los 17 de AMORETT y los 4 nuevos de Finandina forman un grupo de 100
  personas, que quedó completo en la parte 0.
- Otros 11 positivos `FRAUD` comparten reverso en 5 grupos pequeños.
- 100 de los 128 positivos de app son de esa red. Fuera de ella hay 28 positivos de app.

## 11. Gráficas sugeridas para la presentación

| Gráfica | Tabla en `data/live/summary/` |
|---|---|
| Embudo: de 278.954 clientes a 28.293 personas etiquetadas | `01_universe.csv`, `02_people_totals.csv` |
| Barras: positivos por origen de la etiqueta | `02_people.csv` |
| Barras apiladas: canal en positivos y en negativos | `03_channel.csv` |
| Línea: personas por mes, positivos y negativos | `04_month.csv`, `04_month_positive_sources.csv` |
| Barras: texto guardado en producción por canal | `05_stored_ocr.csv` |
| Barras apiladas: ancho de la foto por canal | `07_photo_width_bands.csv` |
| Barras: código legible por canal y por ancho de foto | `09_legibility.csv`, `10_legibility_width.csv` |
| Barras: fallos de cada regla en positivos y negativos | `11_rules.csv` |
| Barras: detección por origen y por canal | `12_detection_source.csv`, `13_detection_channel.csv` |
| Tabla: personas por parte de la validación cruzada | `splits.csv` con el manifest |

## 12. Clasificador de imágenes: primeros resultados

MobileNetV3 afinada, una para el frente y otra para el reverso, con la foto completa a 512×320 y
validación cruzada en cinco partes. Cada persona recibe el puntaje de un modelo que no entrenó
con ella ni con su red. Corridas del 12-nov-2026 (`runs/front_...2041`, `runs/back_...1952` y
`runs/combined`). Todas las detecciones son con 2 % de falsos positivos sobre 410 positivos y
5.297 negativos.

### 12.1 Resumen

| Qué se usa | Positivos detectados | PR-AUC | ROC-AUC |
|---|---|---|---|
| Reglas solas | 49 (12 %) | — | — |
| Reverso solo | 251 (61 %) | 0,71 | 0,91 |
| Frente solo | 288 (70 %) | 0,78 | 0,93 |
| **Promedio de frente y reverso** | **322 (79 %)** | **0,84** | **0,95** |
| Promedio de las dos caras, o alguna regla en fallo | 332 (81 %) | — | — |

Con el promedio y las reglas juntos se marcan 106 de 5.297 negativos (2,0 %).

**Las dos caras se complementan** (cada una con su umbral del 2 %):

| Detectado por | Positivos | Negativos |
|---|---|---|
| Las dos caras | 204 | 11 |
| Solo el frente | 84 | 94 |
| Solo el reverso | 47 | 94 |
| Ninguna | 75 | 5.098 |

### 12.2 Promedio de las dos caras, por grupo

| Parte | Positivos | Detección | ROC-AUC |
|---|---|---|---|
| 0 (la red, nunca vista por su modelo) | 133 | 80 % | 0,97 |
| 1 | 70 | 86 % | 0,96 |
| 2 | 69 | 77 % | 0,96 |
| 3 | 69 | 83 % | 0,96 |
| 4 | 69 | 75 % | 0,91 |

| Canal | Positivos | Detectados | Negativos | Falsos positivos | ROC-AUC dentro del canal |
|---|---|---|---|---|---|
| App | 128 | 102 (80 %) | 980 | 18 (1,8 %) | 0,97 |
| Comercio | 273 | 211 (77 %) | 4.197 | 82 (2,0 %) | 0,94 |
| WhatsApp | 9 | 9 | 118 | 5 (4,2 %) | 0,99 |

| Origen del positivo | Positivos | Detectados |
|---|---|---|
| `FRAUD` | 302 | 238 (79 %) |
| Red de la tienda 623 | 79 | 63 (80 %) |
| Comercio AMORETT | 17 | 11 (65 %) |
| Verificador, confirmada por analista | 8 | 7 |
| Finandina + Libre Destino (nuevos) | 4 | 3 |

| Periodo de registro | Positivos | Detectados | Falsos positivos |
|---|---|---|---|
| Enero a junio de 2025 | 127 | 76 % | 2,3 % |
| Julio a diciembre de 2025 | 91 | 73 % | 2,2 % |
| Enero a agosto de 2026 | 78 | 91 % | 1,5 % |
| Septiembre a noviembre de 2026 | 114 | 78 % | 1,7 % |

**El modelo no aprendió la cámara ni la fecha** (`scripts/32_audit_run.py`):

| Qué se usa para puntuar | ROC-AUC | Detección al 2 % |
|---|---|---|
| Solo el canal | 0,35 | 0 % |
| Canal y tamaño de la foto | 0,43 | 0 % |
| Canal, tamaño de la foto y periodo de registro | 0,74 | 25 % |
| **Promedio de las dos caras (mira las fotos)** | **0,95** | **79 %** |
| El promedio, dentro de grupos con igual canal, tamaño y periodo | 0,96 | — |

**La resolución lo limita** (comercio, promedio de las dos caras):

| Ancho de la foto | Positivos | Detectados |
|---|---|---|
| Hasta 700 px | 29 | 41 % |
| 701 a 1400 px | 149 | 81 % |
| Más de 1400 px | 95 | 83 % |

### 12.3 Reverso solo

Corrida `runs/back_mobilenet_v3_large_512x320_20261112_1952`.

| Medida (410 positivos, 5.297 negativos) | Reverso |
|---|---|
| Detección con 2 % de falsos positivos | 61,2 % (251 de 410) |
| PR-AUC | 0,71 |
| ROC-AUC | 0,91 |
| Para comparar: positivos con algún fallo de las reglas | 12,0 % (49 de 410) |

| Parte | Positivos | Detección al 2 % | ROC-AUC |
|---|---|---|---|
| 0 (la red, nunca vista por su modelo) | 133 | 65 % | 0,92 |
| 1 | 70 | 83 % | 0,95 |
| 2 | 69 | 62 % | 0,92 |
| 3 | 69 | 67 % | 0,93 |
| 4 | 69 | 67 % | 0,88 |

| Canal | Positivos | Detectados | Negativos | Falsos positivos | ROC-AUC dentro del canal |
|---|---|---|---|---|---|
| App | 128 | 75 (59 %) | 980 | 19 (1,9 %) | 0,92 |
| Comercio | 273 | 169 (62 %) | 4.197 | 79 (1,9 %) | 0,90 |
| WhatsApp | 9 | 7 | 118 | 7 (5,9 %) | 0,96 |

| Origen del positivo | Positivos | Detectados |
|---|---|---|
| `FRAUD` | 302 | 192 (64 %) |
| Red de la tienda 623 | 79 | 42 (53 %) |
| Comercio AMORETT | 17 | 7 (41 %) |
| Verificador, confirmada por analista | 8 | 7 |
| Finandina + Libre Destino (nuevos) | 4 | 3 |

**El modelo no aprendió la cámara ni la fecha** (`scripts/32_audit_run.py`):

| Qué se usa para puntuar | ROC-AUC | Detección al 2 % |
|---|---|---|
| Solo el canal | 0,35 | 0 % |
| Canal y tamaño de la foto | 0,43 | 0 % |
| Canal, tamaño de la foto y periodo de registro | 0,74 | 25 % |
| **El clasificador (mira la foto)** | **0,91** | **61 %** |
| El clasificador, dentro de grupos con igual canal, tamaño y periodo | 0,92 | — |

**La resolución también lo limita** (comercio):

| Ancho de la foto | Positivos | Detectados |
|---|---|---|
| Hasta 700 px | 29 | 24 % |
| 701 a 1400 px | 149 | 61 % |
| Más de 1400 px | 95 | 75 % |

## 13. Clasificador de imágenes: segunda iteración

Cuatro variantes decididas de antemano, corridas una vez cada una para frente y reverso sobre la
misma partición (12 y 13-nov-2026). Detección con 2 % de falsos positivos, sobre 410 positivos
y 5.297 negativos.

### 13.1 Cada variante y el promedio

| Variante | Frente | Reverso | Promedio de las dos caras |
|---|---|---|---|
| Primera corrida (MobileNetV3, 512×320, semilla 42) | 70,2 % | 61,2 % | 322 (78,5 %) |
| La misma con otra semilla (7) | 77,3 % | 62,4 % | 326 (79,5 %) |
| Entrada de 768×480 | 73,2 % | 60,7 % | 320 (78,0 %) |
| EfficientNet-B0 | 78,8 % | 68,5 % | 355 (86,6 %) |
| **Promedio de las cuatro anteriores** | **81,2 %** | **71,7 %** | **356 (86,8 %)** |
| Sin transferencia (pesos al azar) | 11,7 % | 23,2 % | 68 (16,6 %) |

El promedio de las cuatro (`runs/ensemble`) tiene PR-AUC 0,89 y ROC-AUC 0,96. Su intervalo del
95 % (remuestreo de personas) va de 83,4 % a 90,0 %, y la mejora sobre la primera corrida es de
8,3 puntos (de 5,6 a 11,2). Con las reglas: 360 de 410 (87,8 %) y 106 de 5.297 negativos (2,0 %).

Cómo leer la tabla:

- **Cambiar solo la semilla movió el frente 7 puntos** (70,2 % a 77,3 %). Una diferencia de ese
  tamaño entre dos corridas sueltas no prueba que una variante sea mejor.
- **La entrada más grande no ayudó** y cuesta más del doble de cómputo.
- **EfficientNet-B0 salió mejor en las dos caras** (8,5 y 7,3 puntos), pero con una sola corrida
  por cara no se distingue del efecto de la semilla en el frente.
- **El promedio no elige:** usa las cuatro variantes preentrenadas tal como se corrieron. Por eso
  su cifra no está inflada por escoger la mejor sobre los mismos datos.
- **Sin transferencia el modelo casi no aprende** con 410 positivos, y es inestable: en el frente
  una parte llegó a 48 % y las otras cuatro quedaron por debajo de 5 %.

### 13.2 El promedio de las cuatro, por grupo

| Parte | Positivos | Detectados | Antes |
|---|---|---|---|
| 0 (la red, nunca vista por su modelo) | 133 | 119 (89 %) | 98 |
| 1 | 70 | 65 (93 %) | 60 |
| 2 | 69 | 56 (81 %) | 51 |
| 3 | 69 | 62 (90 %) | 60 |
| 4 | 69 | 54 (78 %) | 53 |

De las 100 personas de la red tienda 623 + AMORETT, detecta 93 (antes 77).

| Canal | Positivos | Detectados | Negativos | Falsos positivos |
|---|---|---|---|---|
| App | 128 | 119 (93 %) | 980 | 19 (1,9 %) |
| Comercio | 273 | 228 (84 %) | 4.197 | 80 (1,9 %) |
| WhatsApp | 9 | 9 | 118 | 6 (5,1 %) |

| Origen del positivo | Positivos | Detectados | Antes |
|---|---|---|---|
| Marcado `FRAUD` | 302 | 255 (84 %) | 238 |
| Red de la tienda 623 | 79 | 73 (92 %) | 63 |
| Comercio AMORETT | 17 | 16 | 11 |
| Verificador, confirmada por analista | 8 | 8 | 7 |
| Finandina + Libre Destino (nuevos) | 4 | 4 | 3 |

| Detectado por (cada cara con su umbral) | Positivos | Negativos |
|---|---|---|
| Las dos caras | 267 | 11 |
| Solo el frente | 66 | 94 |
| Solo el reverso | 27 | 94 |
| Ninguna | 50 | 5.098 |

### 13.3 Cámara, fecha y resolución

La revisión de cámara y fecha sigue limpia: dentro de grupos con igual canal, tamaño de foto y
periodo el ROC-AUC es 0,98 (337 de los 410 positivos están en esos grupos), contra 0,74 de un
puntaje que solo usa esos datos.

| Ancho de la foto (comercio) | Positivos | Detectados | Antes |
|---|---|---|---|
| Hasta 700 px | 29 | 17 (59 %) | 41 % |
| 701 a 1400 px | 149 | 130 (87 %) | — |
| Más de 1400 px | 95 | 81 (85 %) | — |

Quedan 54 positivos sin detectar (4 fallan alguna regla): `runs/ensemble/missed_positives.csv`.
De los 88 que no detectaba el primer modelo, el promedio recupera 35 y pierde 1.

## 14. Lo que falta medir

- Los mapas de calor (Grad-CAM) y la revisión a ojo de los 54 positivos que el promedio de las cuatro variantes no detecta.
- Repetir EfficientNet-B0 con otras semillas, si se quiere afirmar que es mejor que MobileNetV3.
- La calibración del puntaje a 0–100 y el backtest.
- El texto de los 22.586 negativos que no se han leído (unos 45 USD). Se dejó sin leer porque
  los ya leídos acotan los falsos positivos por debajo del 1 %.
- La modalidad de cada falsificación: la revisión visual se aplazó al análisis de errores.
