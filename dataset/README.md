# Dataset del proyecto

Conjuntos de imágenes utilizados en el proyecto **Evaluación de técnicas de mejora de imagen para reconocimiento facial en escenarios de baja resolución**, basados en [The ORL Database of Faces](https://www.kaggle.com/datasets/tavarez/the-orl-database-for-training-and-testing).

Esta carpeta contiene el dataset original limpio y organizado, las versiones degradadas a baja resolución y las versiones mejoradas con las técnicas de mejora de imagen evaluadas (CLAHE, Super Resolution y Lite-ESRGAN).

---

## Contenido

| Carpeta | Descripción |
|---|---|
| `ORL_Limpio/` | Dataset original validado y organizado por sujeto, sin imágenes corruptas. Es la base de los demás conjuntos. |
| `Degradacion/` | Imágenes degradadas a baja resolución en escalas x2 y x4 (*ORL_escala2* y *ORL_escala4*), generadas con el pipeline de degradación. |
| `Tecnica De mejora de imagen/` | Imágenes mejoradas con CLAHE, Super Resolution y Lite-ESRGAN, aplicadas sobre las imágenes degradadas en las escalas x2 y x4. |

---

## Estructura

Las tres carpetas comparten la misma organización interna:

```
<carpeta>/
├── Training/
│   ├── s1/    →  9 imágenes
│   ├── s2/    →  9 imágenes
│   ├── ...
│   └── s40/   →  9 imágenes
└── Testing/
    ├── s1/    →  1 imagen
    ├── s2/    →  1 imagen
    ├── ...
    └── s40/   →  1 imagen
```

- Cada carpeta `s1` a `s40` corresponde a un **sujeto** distinto (40 sujetos en total).
- `Training/` contiene **9 imágenes por sujeto** (360 imágenes en total).
- `Testing/` contiene **1 imagen por sujeto** (40 imágenes en total).

---

## Especificaciones de las imágenes

| Característica | Valor |
|---|---|
| Resolución | 112 x 92 píxeles (alto x ancho) |
| Formato | .png |
| Canales | Escala de grises |
| Sujetos | 40 (`s1` a `s40`) |
| Imágenes de entrenamiento | 360 (9 por sujeto) |
| Imágenes de prueba | 40 (1 por sujeto) |

---

## Tabla de variables

Al ser un dataset de imágenes, no cuenta con columnas tabulares. Las variables se derivan de cada imagen y de su organización en carpetas.

| Nombre de la variable | Rol | Tipo | Descripción | Unidades | Valores faltantes |
|---|---|---|---|---|---|
| imagen | Feature | Integer | Intensidad de gris de cada píxel de la imagen del rostro, con valores de 0 a 255. Cada imagen de 112 x 92 px tiene 10 304 valores. | píxeles (0 a 255) | no |
| sujeto | Target | Categorical | Identidad de la persona, dada por el nombre de la carpeta. Son 40 clases, de `s1` a `s40`. | n/a | no |
| partición | Other | Categorical | Conjunto al que pertenece la imagen: `Training` (9 imágenes por sujeto) o `Testing` (1 imagen por sujeto). | n/a | no |

---

## Ejemplos de imágenes

Reemplaza cada `URL_IMAGEN_...` por la dirección de la imagen correspondiente (por ejemplo, el enlace "raw" de GitHub o una imagen subida al repositorio).

### Dataset limpio

| ORL_Limpio |
|:---:|
| <img src="URL_IMAGEN_ORL_LIMPIO" width="120" alt="Imagen original del sujeto s1"> |

### Imágenes degradadas

| Escala x2 | Escala x4 |
|:---:|:---:|
| <img src="URL_IMAGEN_ESCALA_X2" width="120" alt="Imagen degradada x2"> | <img src="URL_IMAGEN_ESCALA_X4" width="120" alt="Imagen degradada x4"> |

### Imágenes mejoradas

| Técnica | Escala x2 | Escala x4 |
|---|:---:|:---:|
| CLAHE | <img src="URL_IMAGEN_CLAHE_X2" width="120" alt="CLAHE x2"> | <img src="URL_IMAGEN_CLAHE_X4" width="120" alt="CLAHE x4"> |
| Super Resolution | <img src="URL_IMAGEN_SR_X2" width="120" alt="Super Resolution x2"> | <img src="URL_IMAGEN_SR_X4" width="120" alt="Super Resolution x4"> |
| Lite-ESRGAN | <img src="URL_IMAGEN_LITEESRGAN_X2" width="120" alt="Lite-ESRGAN x2"> | <img src="URL_IMAGEN_LITEESRGAN_X4" width="120" alt="Lite-ESRGAN x4"> |

---

## Conjuntos generados

| Conjunto | Carpeta | Descripción |
|---|---|---|
| `ORL_limpio` | `ORL_Limpio/` | Dataset original validado y estructurado |
| `ORL_escala2` | `Degradacion/` | Degradación en escala x2 |
| `ORL_escala4` | `Degradacion/` | Degradación en escala x4 |
| `ORL_CLAHE_x2` | `Tecnica De mejora de imagen/` | CLAHE sobre imágenes en escala x2 |
| `ORL_CLAHE_x4` | `Tecnica De mejora de imagen/` | CLAHE sobre imágenes en escala x4 |
| `ORL_SuperResolution_x2` | `Tecnica De mejora de imagen/` | Super Resolution (SRCNN) sobre imágenes en escala x2 |
| `ORL_SuperResolution_x4` | `Tecnica De mejora de imagen/` | Super Resolution (SRCNN) sobre imágenes en escala x4 |
| `ORL_LiteESRGAN_x2` | `Tecnica De mejora de imagen/` | Lite-ESRGAN sobre imágenes en escala x2 |
| `ORL_LiteESRGAN_x4` | `Tecnica De mejora de imagen/` | Lite-ESRGAN sobre imágenes en escala x4 |

---

## Dataset original

The ORL Database of Faces: [Kaggle, The ORL database for training and testing](https://www.kaggle.com/datasets/tavarez/the-orl-database-for-training-and-testing)

---

## Autor

Abel Alejandro Mora López, Universidad Nacional de Loja, Carrera de Computación.
