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

**28.292 personas: 410 positivas (1,45 %) y 27.882 negativas.**

| Etiqueta | Personas | En las dos marcas | Con fotos para el clasificador |
|---|---|---|---|
| Positiva | 410 | 22 | 410 |
| Negativa | 27.882 | 402 | 4.019 (muestra) |

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

Negativos: 15.921 de ESMIO y 11.961 de Llévalo Ya.

## 3. Canal de registro

| Canal | Negativos | % | Positivos | % |
|---|---|---|---|---|
| Comercio | 26.185 | 93,9 % | 273 | 66,6 % |
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
| 2025-03 | 1.966 | 20 | | 2026-03 | 1.289 | 19 |
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
| Comercio | 9,9 % (2.593 de 26.185) | 12,1 % (33 de 273) |

El flujo de comercio casi nunca guarda el texto. Por eso se volvió a leer con Rekognition.

## 6. Fotos

**En disco:** 9.362 fotos de 4.681 personas, 16,2 GB.

| Etiqueta | Personas | Fotos | GB | Tamaño mediano de una foto |
|---|---|---|---|---|
| Positiva | 410 | 820 | 1,9 | 2,45 MB |
| Negativa | 4.271 | 8.542 | 14,3 | 1,31 MB |

**Tamaño de las fotos según el canal** (ancho × alto en píxeles, frente y reverso juntos):

| Canal | Tamaño más común | 2.º | 3.º |
|---|---|---|---|
| App | 1920×1210 (79 %) | 1280×720 (8 %) | 1440×1210 (5 %) |
| Comercio | 1280×720 (54 %) | 1920×1210 (20 %) | 640×480 (11 %) |
| WhatsApp | 1280×720 (40 %) | 1920×1210 (24 %) | 1024×768 (10 %) |

| Canal | Hasta 700 px de ancho | 701 a 1400 px | Más de 1400 px |
|---|---|---|---|
| App | 8 (1 %) | 72 (13 %) | 488 (86 %) |
| Comercio | 1.178 (14 %) | 5.037 (59 %) | 2.323 (27 %) |
| WhatsApp | 24 (9 %) | 138 (54 %) | 92 (36 %) |

Los dos flujos piden a la cámara 1920 px de ancho; el navegador entrega lo que la cámara
soporta. En comercio la cámara es la del equipo de la tienda.

## 7. Lectura nueva con Rekognition

Leídas otra vez las dos caras de **4.667 personas**: los 410 positivos y 4.257 negativos.
9.334 fotos, más 200 de la prueba de ampliación: **9,53 USD**, pagados con la cuenta de AWS de
Alejandro, aparte de la corporativa.

**Qué se logra leer** (porcentaje de personas; solo líneas con confianza de 80 o más):

| Etiqueta | Canal | Personas | Número del frente | Código del reverso | Serie | Registrador |
|---|---|---|---|---|---|---|
| Negativa | App | 156 | 92 % | 35 % | 48 % | 71 % |
| Negativa | Comercio | 3.982 | 78 % | 14 % | 25 % | 25 % |
| Negativa | WhatsApp | 118 | 81 % | 30 % | 42 % | 35 % |
| Positiva | App | 128 | 98 % | 58 % | 44 % | 58 % |
| Positiva | Comercio | 273 | 82 % | 22 % | 36 % | 26 % |
| Positiva | WhatsApp | 9 | 89 % | 0 % | 0 % | 0 % |
| **Negativa** | **todos** | **4.257** | **78 %** | **15 %** | **26 %** | **27 %** |
| **Positiva** | **todos** | **410** | **88 %** | **32 %** | **37 %** | **36 %** |

**La lectura depende del ancho de la foto del reverso:**

| Ancho de la foto | Personas | Código del reverso | Serie | Registrador |
|---|---|---|---|---|
| Hasta 700 px | 602 | 6 % | 12 % | 8 % |
| 701 a 1400 px | 2.619 | 9 % | 24 % | 19 % |
| Más de 1400 px | 1.446 | 34 % | 40 % | 51 % |

**Pruebas que acompañaron la lectura**

- **Ampliar la cédula no ayuda.** En 200 personas, recortar la cédula y ampliarla al doble
  subió el código legible de 22,5 % a 23,5 % (6 ganados, 4 perdidos). El límite es la cámara.
- **El filtro de confianza está justificado.** Con confianza de 80 o más, el número del código
  coincidió con el del frente en 152 de 152 lecturas; por debajo de 80, en 77 de 102 (75 %).
  Medido sobre las primeras 681 personas.
- **El sexo impreso no se lee en nadie:** Rekognition entrega la etiqueta y el valor en líneas
  separadas y el patrón actual no encaja. Pendiente.

## 8. Reglas estructurales

Sobre las 4.667 personas leídas. Un fallo en un negativo es un falso positivo; en un positivo
es una detección. "Tope" es la tasa máxima compatible con cero fallos (95 % de confianza).

| Regla | Positivos: fallos / evaluables | Negativos: fallos / evaluables | Falsos positivos |
|---|---|---|---|
| Reverso repetido en otra cédula | 37 / 137 (27,0 %) | 0 / 630 | tope 0,48 % |
| Registrador que no corresponde a la fecha | 19 / 142 (13,4 %) | 1 / 883 | 0,11 % |
| Fecha de producción incoherente | 2 / 126 (1,6 %) | 0 / 593 | tope 0,51 % |
| Número del frente distinto al del código | 0 / 129 | 0 / 585 | tope 0,51 % |
| Formato del código | 0 / 133 | 0 / 630 | tope 0,48 % |
| Meses mal escritos | 0 / 374 | 0 / 2.873 | tope 0,10 % |
| Sexo del código distinto al impreso | 0 / 0 | 0 / 0 | sin medir |

**Alcance de las reglas**

| | Positivos (410) | Negativos (4.257) |
|---|---|---|
| Con alguna regla del código evaluable | 175 (43 %) | 1.033 (24 %) |
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
| App | 128 | 37 (29 %) | 156 | 0 |
| Comercio | 273 | 12 (4 %) | 3.982 | 1 |
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

## 10. Gráficas sugeridas para la presentación

| Gráfica | Tabla en `data/live/summary/` |
|---|---|
| Embudo: de 278.954 clientes a 28.292 personas etiquetadas | `01_universe.csv`, `02_people_totals.csv` |
| Barras: positivos por origen de la etiqueta | `02_people.csv` |
| Barras apiladas: canal en positivos y en negativos | `03_channel.csv` |
| Línea: personas por mes, positivos y negativos | `04_month.csv`, `04_month_positive_sources.csv` |
| Barras: texto guardado en producción por canal | `05_stored_ocr.csv` |
| Barras apiladas: ancho de la foto por canal | `07_photo_width_bands.csv` |
| Barras: código legible por canal y por ancho de foto | `09_legibility.csv`, `10_legibility_width.csv` |
| Barras: fallos de cada regla en positivos y negativos | `11_rules.csv` |
| Barras: detección por origen y por canal | `12_detection_source.csv`, `13_detection_channel.csv` |

## 11. Lo que falta medir

- El clasificador de imágenes: no existe todavía. No hay cifras de entrenamiento ni de prueba.
- El texto de los 23.625 negativos que no se han leído (unos 47 USD). Se dejó sin leer porque
  los ya leídos acotan los falsos positivos por debajo del 1 %.
- La modalidad de cada falsificación: la revisión visual se aplazó al análisis de errores.
