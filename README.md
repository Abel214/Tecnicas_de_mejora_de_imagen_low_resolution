# Reconocimiento facial con imágenes degradadas

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23343973.svg)](https://doi.org/10.5281/zenodo.23343973)

Trabajo de Integración Curricular de la **Universidad Nacional de Loja** (Carrera de Computación): *Evaluación de técnicas de mejora de imagen para reconocimiento facial en escenarios de baja resolución*.


## Objetivo

Evaluar el desempeño (**Rank-1**, **EER** y **TAR**) de los modelos ligeros **MobileFaceNet**, **ShuffleFaceNet** y **GhostFaceNetV2** al aplicar las técnicas de mejora de imagen **CLAHE**, **Super Resolution** (SRCNN) y **Lite-ESRGAN** sobre imágenes de baja resolución de *The ORL Database of Faces*.

## Resumen del método

| Etapa | Qué se hace |
|---|---|
| 1. Dataset | ORL (40 sujetos, 400 imágenes en escala de grises de 112 × 92 px), con 9 imágenes de entrenamiento y 1 de prueba por sujeto (`ORL_limpio`) |
| 2. Degradación | Reducción bilineal en escalas ×2 y ×4 con ruido gaussiano (`ORL_escala2`, `ORL_escala4`) |
| 3. Mejora de imagen | CLAHE, Super Resolution (SRCNN) y Lite-ESRGAN (versión reducida, sin discriminador), sobre ambas escalas: 6 conjuntos mejorados |
| 4. Entrenamiento | Modelo base con las imágenes originales y cuatro experimentos por arquitectura (ver abajo) |
| 5. Evaluación | Rank-1, EER y TAR por modelo, técnica y escala |

### Experimentos

| Experimento | Descripción |
|---|---|
| Modelo base | Entrenamiento con las 360 imágenes originales de *Training* |
| 1 | Evaluación del modelo base sobre las imágenes degradadas y mejoradas, sin reentrenar |
| 2 | Reentrenamiento con Cross-Entropy sobre imágenes originales y degradadas o mejoradas (720 de entrenamiento y 80 de prueba por condición) |
| 3 | Reentrenamiento con activación PReLU y pérdida ArcFace |
| 4 | Solo GhostFaceNetV2: ablación sin la rama DFC |

## Ejemplos de imágenes

### Dataset limpio

| ORL_Limpio |
|:---:|
| <img src="https://i.postimg.cc/pVq5Bf0Y/1.jpg" width="120" alt="Imagen original del sujeto s1"> |

### Imágenes degradadas

| Escala x2 | Escala x4 |
|:---:|:---:|
| <img src="https://i.postimg.cc/8PcJX79s/1.png" width="120" alt="Imagen degradada x2"> | <img src="https://i.postimg.cc/KvWK6Zh7/3.png" width="120" alt="Imagen degradada x4"> |

### Imágenes mejoradas

| Técnica | Escala x2 | Escala x4 |
|---|:---:|:---:|
| CLAHE | <img src="https://i.postimg.cc/zGRL59J1/2.png" width="120" alt="CLAHE x2"> | <img src="https://i.postimg.cc/3w5kwnRx/2.png" width="120" alt="CLAHE x4"> |
| Super Resolution | <img src="https://i.postimg.cc/25QVLhbB/6.png" width="120" alt="Super Resolution x2"> | <img src="https://i.postimg.cc/xC9qv8MB/6.png" width="120" alt="Super Resolution x4"> |
| Lite-ESRGAN | <img src="https://i.postimg.cc/B6NbM3C8/8.png" width="120" alt="Lite-ESRGAN x2"> | <img src="https://i.postimg.cc/2yK6NxMN/8.png" width="120" alt="Lite-ESRGAN x4"> |

## Estructura del proyecto

```text
.
├── Organización del dataset y degradación de imágenes/
│   └── Semana 1 y 2.ipynb
├── Experimentos Técnicas de Mejora de Imagen/
│   ├── CLAHE/
│   ├── Lite-Ersgan/
│   └── Super resolution/
├── Entrenamiento de GhostFaceNetv2/
├── Entrenamiento de MobileFaceNet/
├── Entrenamiento de ShuffleFaceNet/
├── Evaluación de métricas/
│   ├── GhostFacenetV2/
│   ├── MobilFacenet/
│   └── ShuffleFacenet/
├── dataset/
│   ├── Dataset Original de Kaggle/
│   ├── Degradacion/
│   ├── ORL_limpio/
│   └── Tecnica De mejora de imagen/
└── requirements.txt
```

Las carpetas de entrenamiento contienen los notebooks, los historiales (`.json`) y, en algunos casos, los pesos (`.pth`) de cada variante por técnica y escala.

## Resultados

Los resultados de EER y TAR se exportan junto a cada notebook de evaluación (`Evaluación de métricas/`):

- `GhostFacenetV2/resultados_eer_tar_ghostfacenetv2.xlsx`
- `MobilFacenet/resultados_eer_tar_mobilefacenet.xlsx`
- `ShuffleFacenet/resultados_eer_tar_shufflefacenet.xlsx`

No se incluye una tabla consolidada: compara valores solo si corresponden a la misma partición y configuración experimental.

## Entorno y ejecución

Probado en Windows con Python 3.13.3, PyTorch 2.6.0 (CUDA 12.4) y torchvision 0.21.0. Los modelos de reconocimiento se entrenaron en CPU; la GPU se usó solo para Lite-ESRGAN.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Abre el notebook de la etapa que quieras ejecutar (por ejemplo, en VS Code con el entorno `.venv` como kernel) y sigue sus celdas en orden. Las rutas son relativas a la carpeta del notebook. Para CPU u otra versión de CUDA, adapta las líneas de PyTorch en `requirements.txt`.

## Fuente del dataset y alcance

- Dataset: *The ORL Database of Faces* (AT&T Laboratories Cambridge), descargado de Kaggle. Consulta sus condiciones de uso antes de redistribuirlo.
- EER y TAR miden verificación y no equivalen a la exactitud de clasificación del entrenamiento.
- El estudio no incluye despliegue, sistemas de vigilancia ni validación en entornos reales.

## Cómo citar

A. A. Mora López, *Evaluación de técnicas de mejora de imagen para reconocimiento facial en escenarios de baja resolución*, Universidad Nacional de Loja, 2026. DOI: [10.5281/zenodo.23343973](https://doi.org/10.5281/zenodo.23343973).

## Autor

**Abel Alejandro Mora López** — Universidad Nacional de Loja, Carrera de Computación
