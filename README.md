### Sabiendo Programar un Poco 01
# Descripción de imágenes con Florence-2

| <img src="https://github.com/user-attachments/assets/9cadc830-9e69-48bb-90a0-a3deef514224" width="50%"/> | 
|---|
|*The image shows a man in a black suit and tie standing on a stage with a large screen in the background. He is holding a microphone and appears to be giving a speech or presentation. The man is in a crouching position with his left leg extended and his right leg bent at the knee. He has a serious expression on his face and his mouth is open as if he is shouting or gesturing with his hands. The stage is lit up with bright lights and there is a logo on the top left corner of the screen.*|

Florence-2 es un modelo desarrollado por Microsoft, que analiza imágenes de distintas maneras: puede describirlas, sugerir áreas de interés, detectar objetos, vincular palabras o frases a objetos de la imagen, segmentar a nivel de píxeles o extraer el texto de una imagen. Esto nos puede servir para crear motores de búsqueda de imágenes, software de accesibilidad o analizar datos, por ejemplo.

Es un modelo bastante potente y versátil, y lo mejor es que lo podemos ejecutar en nuestro ordenador, incluso sin GPU. En este tutorial vamos a usar el modelo para obtener las descripciones del contenido (captions) de las imágenes que tenemos en una carpeta.

El código y una carpeta con imágenes de ejemplo están disponibles en [nuestro Github](https://github.com/BothRocks/SPUP-01).

### ¿Qué vamos a aprender?

- Cómo configurar y usar un modelo de visión por ordenador de última generación.
- Cómo procesar múltiples imágenes automáticamente y obtener su descripción.
- Cómo guardar los resultados en varios formatos, y aprovecharlos en futuros proyectos.

### Requisitos previos

- **Editor de código**: Un editor de texto o IDE para Python. Por aquí nos gusta Visual Studio Code, pero puedes usar cualquier editor con el que te sientas cómoda.
- **Familiaridad con entornos virtuales**: Necesitaremos crear y activar un entorno virtual en Python (sea con `venv` o `conda`).
- **Conocimientos básicos de Python**: Variables, funciones, bucles y manejo de archivos en Python.
- **Espacio en disco**: Necesitaremos unos 2GB de espacio en disco, para instalar las dependencias y el modelo.
- **Conexión a Internet**: Necesitamos una conexión a Internet estable para descargar las librerías y el modelo.

### Paso 0: Entorno virtual y librerías

Os recomiendo crear un entorno virtual (con `conda` o `venv`) para no tener conflictos entre las librerías. En cuanto a la versión de Python, la 3.11 va de maravilla, que es la que nos sugiere Visual Studio Code.

A continuación, hay que instalar las siguientes librerías

- PIL (Python Imaging Library): para manipular y procesar imágenes.
- Transformers: para acceder a modelos de lenguaje y visión avanzados.
- Torch (PyTorch): para todos los temas de ML.
- Einops: para simplificar las operaciones con tensores.
- Timm (PyTorch Image Models): una colección de modelos preentrenados de visión por ordenador.

```bash
pip install pillow 'transformers[torch]' einops timm
```

### Paso 1: Inicialización de cosas

Cargamos las librerías, detectamos si el ordenador tiene GPU o no, y según eso usamos un tipo de datos u otro (16 o 32 bits), definimos el nombre del modelo, la carpeta donde están las imágenes, el nombre del fichero de salida y la tarea que queremos que haga el modelo, en este caso `<CAPTION>`, vamos, que nos describa la imagen. Dejo comentadas dos opciones más, que nos devolverían descripciones cada vez más detalladas. Aquí ya es cuestión de probar y usar la que mejor nos sirva.

```python
import glob
import json
import os
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
```

```python
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "microsoft/Florence-2-large"

image_folder = "demo/"
output_file = "demo.jsonl"

task = "<CAPTION>"
# task = "<DETAILED_CAPTION>"
# task = "<MORE_DETAILED_CAPTION>"
```

### Paso 2: Carga del modelo

Bueno, este paso es un poco movida: resulta que el modelo se va a quejar porque no encuentra una librería, pero resulta que esa librería no la usa para nada y además solo funciona con GPU, así que hay que hacer ~~una chapuza~~ un truco para ignorar esa librería.

Básicamente, con el siguiente bloque de código, estamos diciendo que  cuando se cargue `modeling_florence2.py`, pase totalmente del requisito de `flash_attn` y que, en otro caso, respete los imports.

```python
def fixed_get_imports(filename):
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

def get_model_processor(model_id, device, torch_dtype):
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor
```

### Paso 3: Obtenemos la lista de imágenes para analizar

Esta función devuelve la lista de ficheros de imagen que haya en la carpeta que le indiquemos. Eso sí, comprobamos antes que la carpeta existe.

```python
def get_image_files(folder):
    if not os.path.exists(folder):
        print(f"Error: La carpeta '{folder}' no existe.")
        return []
    image_files = (
        glob.glob(os.path.join(folder, "*.jpg"))
        + glob.glob(os.path.join(folder, "*.jpeg"))
        + glob.glob(os.path.join(folder, "*.png"))
    )
    return image_files
```

### Paso 4: El bucle principal

Aquí está el meollo del programa. Antes de nada, cargamos el modelo. La primera vez tardará un ratillo, ya que se lo baja, y ocupa un giga y pico. El siguiente paso es sacar la lista de imágenes y finalmente, para cada imagen, pasarla por el modelo. El modelo toma la imagen (convertida en un tensor) y un prompt (convertido en tokens) como entrada y devuelve una secuencia de tokens de salida que se convierten en texto, obteniéndose así la descripción (caption) de la imagen.

Entrando en detalle, estos son los pasos que se dan en el bucle principal:

1. Se carga la imagen.
2. La imagen y el prompt se pasan a un formato que la red neuronal es capaz de interpretar. El prompt lo convierte en la siguiente lista de tokens (valores numéricos que representan a palabras o partes de palabras) `[0, 2264, 473, 5, 2274, 6190, 116, 2]` , y la imagen, en un tensor (array multidimensional) de 3 canales (RGB) y 768 x 768 valores.
3. La imagen y el prompt, transformados en el paso anterior, se presentan a la red neuronal y obtenemos una serie de tokens de vuelta, algo de este palo: `[[2, 0, 250, 3034, 5271, 2828, 15, 5, 1255, 220, 7, 10, 8875, 64, 4, 2]]`
4. Los tokens se convierten en texto: `</s><s>A computer monitor sitting on the ground next to a trash can.</s>`
5. Y el texto finalmente se convierte en una linea de JSON: `{'<CAPTION>': 'A computer monitor sitting on the ground next to a trash can.'}`
6. Dejamos listo el JSON para grabarlo posteriormente: `{'text': 'A computer monitor sitting on the ground next to a trash can.', 'file_name': '753534820828741632_0.jpg'}`

```python
model, processor = get_model_processor(model_id, device, torch_dtype)
image_files = get_image_files(image_folder)

results = []
for index, image_file in enumerate(image_files, 1):

    # 1
    image = Image.open(image_file)

    # 2
    inputs = processor(text=task, images=image, return_tensors="pt").to(
        device, torch_dtype
    )

    # 3
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
        early_stopping=False,
    )

    # 4
    gen_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # 5
    parsed_answer = processor.post_process_generation(
        gen_text, task=f"{task}", image_size=(image.width, image.height)
    )

    # 6
    value = parsed_answer.pop(task)
    parsed_answer["text"] = value
    parsed_answer["file_name"] = os.path.basename(image_file)

    print(
        f"{index}/{len(image_files)} {parsed_answer['file_name']}: {parsed_answer['text']}"
    )
    results.append(parsed_answer)
```

### Paso 5: Lo grabamos todo en disco

Vamos a utilizar el formato JSONL (JSON Lines), que es un formato de texto para almacenar datos estructurados donde cada línea es un objeto JSON válido. Combina la estructuración del JSON con la facilidad de procesar un fichero línea por línea, de manera independiente.

```python
with open(output_file, "w") as jsonl_file:
    for line in results:
        jsonl_file.write(json.dumps(line) + "\n")
```

### Resultado

Tras ejecutar el código con las imágenes de ejemplo, habremos obtenido el siguiente fichero JSONL.

```json
{"text": "A computer monitor sitting on the ground next to a trash can.", "file_name": "753534820828741632_0.jpg"}
{"text": "A trash can sitting on the side of a street.", "file_name": "752079462820200448_0.jpg"}
{"text": "A television sitting on the ground next to a green trash can.", "file_name": "754066664700641280_0.jpg"}
{"text": "A portable air conditioner sitting on the ground next to a wall.", "file_name": "756467765369520128_0.jpg"}
{"text": "A metal box sitting on the ground in the woods.", "file_name": "754617998299512832_0.jpg"}
```


| <img src="https://github.com/user-attachments/assets/8bebdcd7-74c9-48be-b620-3bf21abbf542" alt="A portable air conditioner sitting on the ground next to a wall." width="50%"/>   | <img src="https://github.com/user-attachments/assets/2658b01b-0d47-4cdd-91bc-721ea29dc0d4" alt="A computer monitor sitting on the ground next to a trash can." width="50%"/>   |
|:---:|:---:|
| *A portable air conditioner sitting on the ground next to a wall.*  |*A computer monitor sitting on the ground next to a trash can.* |



### ¡Bola extra!

La idea que tengo es usar estas descripciones en otros sitios y, como cada modelo o workflow te las pide en un formato distinto, he desarrollado un conversor la mar de sencillo que a partir del JSONL genera un JSON, un CSV y una lista de ficheros de texto con el mismo nombre que las imágenes pero con extensión txt.

```python
import csv
import json
import os

input_file = "spup01.jsonl"

base_name = os.path.splitext(input_file)[0]
json_output_file = f"{base_name}.json"
csv_output_file = f"{base_name}.csv"

data = []
with open(input_file, "r") as f:
    for line in f:
        data.append(json.loads(line))

with open(json_output_file, "w") as f:
    json.dump(data, f, indent=4)

with open(csv_output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "file_name"])
    writer.writeheader()
    writer.writerows(data)

for entry in data:
    text_content = entry["text"]
    file_name = entry["file_name"].replace(".jpg", ".txt")
    with open(file_name, "w") as f:
        f.write(text_content)
```

### Próximos pasos

Algunas ideas para continuar el proyecto:

- Probar con los prompts `<DETAILED_CAPTION>` y `<MORE_DETAILED_CAPTION>` para ver cómo varia el detalle de la descripción. Estos prompts le indican al modelo que genere descripciones más detalladas de las imágenes, lo que puede ser útil para un análisis más profundo.
- Tratar las imágenes por grupos, en vez de de una en una como ahora. En determinado casos, esto puede hacer que el procesado sea más rápido.
- Modificar el código para que el prompt, la carpeta de entrada y el fichero de salida se puedan configurar desde la línea de comando. Esto nos permite usar el script sin tener que modificar el código fuente.
- Modificar el código para hacerlo más robusto: por ejemplo, comprobando que la carpeta de entrada exista, que las imágenes se puedan leer...
- Adaptar el código para que funcione también en una GPU.

### Referencias

Paper: [https://arxiv.org/abs/2311.06242](https://arxiv.org/abs/2311.06242)

Página oficial: [https://huggingface.co/microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)

Código del tutorial: [https://github.com/BothRocks/SPUP-01](https://github.com/BothRocks/SPUP-01)
