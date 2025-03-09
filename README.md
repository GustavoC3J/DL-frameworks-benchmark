# DL Frameworks Benchmark

Este proyecto evalúa el rendimiento y eficiencia de diferentes frameworks y librerías de entrenamiento para **Deep Learning**.

## Características
- Comparación de múltiples frameworks como TensorFlow, PyTorch y JAX, con y sin Keras.
- Evaluación de modelos MLP, CNN y LSTM.
- Evaluación de métricas clave como tiempo de entrenamiento y consumo energético.

## Preparación del entorno
Para ejecutar este proyecto, es recomendable utilizar **Miniconda** para gestionar los entornos. Sigue estos pasos:

### Instalar Miniconda
Descarga e instala Miniconda desde la página oficial:  
🔗 [Descargar Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)

### Crear los entornos de trabajo
Este proyecto usa tres entornos de Conda, cada uno definido en archivos `.yml` dentro de la carpeta `environments/`. Para crearlos, ejecuta los siguientes comandos desde la ruta base del proyecto:

```sh
conda env create -f environments/tf_env.yml
conda env create -f environments/torch_env.yml
conda env create -f environments/jax_env.yml
```

### Descargar los conjuntos de datos
En la carpeta `datasets/`se encuentra un script de Python para descargar los conjuntos de datos. Simplemente hay que ejecutar:

```sh
python datasets/download.py
```


## Ejecutar un experimento

En la carpeta `bash/`se encuentra el script `run.sh`. Este script recibe los parámetros del experimento, activa el entorno correspondiente al framework, y ejecuta el script python `experiment.run`.
Los parámetros son el nombre del framework, el tipo de modelo, el id de la GPU y la semilla:
  1. Framework: tf-keras, torch, torch-keras, jax, jax-keras.
  2. Modelo: mlp, cnn, lstm.
  3. GPU: Número de GPU (0, 1, 2...). Ejecutar el comando `nvidia-smi`para ver el id de cada GPU.
  4. Semilla: Cualquier número.

Ejemplo 1: Para ejecutar un experimento con PyTorch y un modelo CNN en la GPU 0 con la semilla 42, usa:

```sh
bash/run.sh torch cnn 0 42
```
Ejemplo 2: Para ejecutar un experimento con JAX y Keras, y un modelo MLP en la GPU 2 con la semilla 123456, usa:

```sh
bash/run.sh jax-keras mlp 2 123456
```

Al terminar la ejecución, los resultados del experimento se almacenarán en su carpeta correspondiente dentro de `results/`.
