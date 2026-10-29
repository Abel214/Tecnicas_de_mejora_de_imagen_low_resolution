# Reconocimiento facial con imágenes degradadas

Proyecto académico de la **Universidad Nacional de Loja**, Carrera de Computación. Reúne experimentos para estudiar cómo técnicas de mejora de imagen y distintos modelos de reconocimiento facial se comportan con imágenes de la base ORL en diferentes condiciones de calidad.

El trabajo está organizado en notebooks de preparación del dataset, mejora de imágenes, entrenamiento y evaluación. **No incluye una aplicación web**: los experimentos se ejecutan desde los notebooks de cada carpeta.

## Objetivo

Evaluar el nivel de precisión de Rank-1, EER y TAR alcanzado por los modelos de reconocimiento facial **ShuffleFaceNet**, **GhostFaceNetv2** y **MobileFaceNet** al aplicar las técnicas de mejora de imagen Super Resolution, CLAHE y Lite-ESRGAN sobre imágenes de baja resolución del dataset The ORL Database of Faces.

## Flujo de trabajo

1. **Organización y preparación:** los notebooks de organización preparan el dataset ORL y sus conjuntos para los experimentos.
2. **Degradación y mejora:** se generan o procesan variantes de las imágenes con CLAHE, super-resolución (SR) y Lite-ESRGAN, según la escala y el experimento.
3. **Entrenamiento:** los notebooks de cada arquitectura entrenan y registran experimentos para las variantes disponibles. Las carpetas contienen notebooks, historiales de entrenamiento y, en algunos casos, checkpoints.
4. **Evaluación:** notebooks independientes calculan métricas de verificación, entre ellas **EER** y **TAR**, y exportan resultados a hojas de cálculo.

Los detalles de configuración, rutas, variantes de entrenamiento y resultados de cada ejecución están en el notebook correspondiente. Las métricas se deben consultar en los archivos de resultados; no se presenta aquí una tabla consolidada porque los experimentos no necesariamente comparten exactamente la misma configuración.

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
│   └── GhostFaceNetv2/
├── Entrenamiento de MobileFaceNet/
│   └── MobileFacenet/
├── Entrenamiento de ShuffleFaceNet/
│   └── Shufflefacenet/
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

Las carpetas de entrenamiento incluyen subcarpetas para las variantes por arquitectura y técnica (CLAHE, SR y Lite-ESRGAN, entre otras). También pueden contener archivos `.json` con historiales y archivos `.pth` con pesos. Los nombres exactos y la disponibilidad de cada variante dependen de la carpeta.

## Evaluación y resultados

Cada arquitectura tiene un notebook de evaluación en `Evaluación de métricas/`. Los resultados exportados se encuentran junto a esos notebooks:

- `GhostFacenetV2/resultados_eer_tar_ghostfacenetv2.xlsx`
- `MobilFacenet/resultados_eer_tar_mobilefacenet.xlsx`
- `ShuffleFacenet/resultados_eer_tar_shufflefacenet.xlsx`

Consulta también los notebooks para revisar cómo se cargan los modelos y los datos, y cómo se calculan las métricas. Los valores no deben compararse sin comprobar antes que correspondan a la misma partición y configuración experimental.

## Requisitos y ejecución

Se recomienda utilizar un entorno virtual. Desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

Abre el notebook de la etapa que deseas ejecutar y sigue sus celdas en orden. Los notebooks utilizan rutas relativas al dataset; ejecútalos desde su ubicación o ajusta las rutas si cambias la estructura de carpetas. Algunos experimentos requieren los pesos `.pth` y los datos de entrada esperados por el notebook.

`requirements.txt` fija versiones de PyTorch y torchvision para CUDA 12.4, además de dependencias de procesamiento de imágenes, análisis y Jupyter. Comprueba la compatibilidad con el controlador y el hardware disponibles antes de instalar. Para utilizar CPU u otra versión de CUDA, adapta esas dependencias al entorno y a las instrucciones del notebook; el archivo actual no garantiza compatibilidad con todas las plataformas.

## Alcance y consideraciones

- Los resultados dependen del dataset, sus particiones, los checkpoints y la configuración específica de cada notebook.
- EER y TAR corresponden a evaluación de verificación y no son equivalentes al accuracy de clasificación durante el entrenamiento.
- Los notebooks pueden requerir una ejecución prolongada y recursos de memoria, especialmente durante entrenamiento y super-resolución.
- Las carpetas documentadas aquí describen el flujo experimental principal; otros directorios del repositorio pueden contener pruebas auxiliares o código de terceros.

## Autor

**Abel Alejandro Mora López**  
Universidad Nacional de Loja — Carrera de Computación
