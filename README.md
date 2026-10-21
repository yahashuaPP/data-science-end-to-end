# Fundamentos de Ciencia de Datos

Repositorio base del curso.

Aquí viven los laboratorios y los PSets, construidos en torno a un único proyecto de ciencia de datos que cada estudiante define y lleva a cabo de principio a fin durante la clase. Es la fuente de la verdad: todo tu trabajo del semestre va aquí, no en otro lado.

## Antes de empezar

No clones ni hagas push a este repositorio directamente. Cada estudiante trabaja en su propio fork:

1. Arriba a la derecha, haz clic en **"Fork"** y créalo en tu propia cuenta de GitHub.
2. Todo tu trabajo del semestre (laboratorios, PSets) va en tu fork, no aquí.

## Sobre el ambiente de desarrollo

Las primeras dos semanas del curso son de formulación y diseño del proyecto: no requieren código ni un entorno de desarrollo todavía. Desde la Sesión "Contenedores para pipelines de proyectos", cada laboratorio produce código: empaquetas el entorno de tu proyecto con Docker y aprovisionas su almacenamiento en Google Cloud con Terraform (Semana 3, Sesión 2). Cada carpeta de `laboratorios/` que lo requiera incluye sus propios requisitos e instrucciones de instalación en su `README.md`; no hay un ambiente único para todo el curso, porque cada estudiante empaqueta el ambiente de su propio proyecto, no uno compartido.

## Estructura del repositorio

```
laboratorios/           Laboratorio de cada clase, uno por carpeta
  semana-01-sesion-02/
  semana-02-sesion-01/
  semana-02-sesion-02/
  semana-03-sesion-01/
  semana-03-sesion-02/
  ...
psets/                  Tus entregas de PSet, una carpeta por PSet
  pset-01/
  ...
```

Cada carpeta de `laboratorios/` y `psets/` tiene su propio `README.md` con las instrucciones de esa entrega.

## Cómo actualizar tu fork cada semana

Cuando se agregue una semana nueva, sincroniza tu fork:

1. En la página de tu fork en GitHub, haz clic en el botón **"Sync fork" → "Update branch"**.
2. En tu copia local o en tu Codespace, ejecuta `git pull` para obtener esos cambios.

Por terminal, en vez del botón:
```
git fetch upstream
git merge upstream/master
```
(la primera vez, agrega el remoto: `git remote add upstream https://github.com/ErickNaunay/fundamentos-ciencia-datos.git`)

## Cómo trabajar aquí

- Haz commits de tu avance de forma continua, con mensajes que indiquen qué hiciste, no `"cambios"` ni `"fix"`.
- Cada entrega (laboratorio, PSet) tiene su propia carpeta con su propio `README.md`: revísalo antes de empezar.
