# Guía Completa: Crear una IA Mínima pero Divertida con TensorFlow

## Resumen Ejecutivo

Esta guía te permitirá construir tu primera IA funcional con TensorFlow en menos de 2 horas. Implementaremos tres proyectos progresivos: un clasificador básico, una red neuronal simple, y un generador de texto estilo Shakespeare. Cada proyecto está diseñado para ser inmediatamente ejecutable y visualmente satisfactorio, priorizando el aprendizaje práctico sobre la teoría exhaustiva.

**Perfil ideal**: Desarrolladores con conocimientos básicos de Python que buscan una introducción práctica a TensorFlow sin prerequisitos de machine learning.

**Tiempo total estimado**: 90-120 minutos para completar los tres proyectos.

## Proyecto 1: Tu Primer Grafo Computacional (15 minutos)

### Objetivo
Entender el paradigma fundamental de TensorFlow: los grafos computacionales. Este primer proyecto te dará confianza inmediata con la sintaxis básica.

### Código Completo

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Crear un grafo simple pero visual
def primer_grafo_divertido():
    # Definir constantes con nombres descriptivos
    numero_magico = tf.constant(42, name="respuesta_universal")
    multiplicador = tf.constant(2, name="duplicador")
    
    # Operaciones básicas
    resultado = tf.multiply(numero_magico, multiplicador, name="resultado_final")
    
    # Crear datos aleatorios para visualización
    datos_random = tf.random.normal([100, 2], mean=0, stddev=1, name="datos_aleatorios")
    
    # Crear una sesión y ejecutar
    with tf.Session() as sess:
        # Ejecutar operaciones simples
        valor_final = sess.run(resultado)
        print(f"🎯 El resultado de la vida × 2 = {valor_final}")
        
        # Generar y visualizar datos
        puntos = sess.run(datos_random)
        
        # Guardar el grafo para TensorBoard
        writer = tf.summary.FileWriter('./mi_primer_grafo', sess.graph)
        writer.close()
    
    # Visualización divertida
    plt.figure(figsize=(8, 6))
    plt.scatter(puntos[:, 0], puntos[:, 1], c=np.random.rand(100), 
                alpha=0.6, s=50, cmap='viridis')
    plt.title('Tu Primera Nube de Datos Generada con TensorFlow')
    plt.xlabel('Dimensión X')
    plt.ylabel('Dimensión Y')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return puntos

# Ejecutar
if __name__ == "__main__":
    datos = primer_grafo_divertido()
    print(f"📊 Generaste {len(datos)} puntos de datos!")
```

### Conceptos Clave Implementados
- **Grafos computacionales**: Definición antes de ejecución
- **Sesiones**: Contexto de ejecución del grafo
- **Tensores**: Arrays multidimensionales como unidad básica
- **TensorBoard**: Visualización del grafo (ejecuta `tensorboard --logdir=./mi_primer_grafo`)

## Proyecto 2: Red Neuronal para Clasificación Visual (35 minutos)

### Objetivo
Construir una red neuronal que clasifique patrones espirales en 2D, creando visualizaciones atractivas del proceso de aprendizaje.

### Código Completo

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

class RedNeuronalVisual:
    def __init__(self):
        tf.reset_default_graph()
        self.construir_modelo()
        self.historia = {'perdida': [], 'precision': []}
        
    def generar_datos_espiral(self, n_puntos=300):
        """Genera datos en forma de espiral para clasificación binaria"""
        np.random.seed(42)
        theta = np.sqrt(np.random.rand(n_puntos, 1)) * 2 * np.pi
        
        # Clase 0: espiral interior
        r_a = 2 * theta + np.pi
        data_a = np.concatenate([
            r_a * np.cos(theta), 
            r_a * np.sin(theta)
        ], axis=1)
        
        # Clase 1: espiral exterior
        r_b = -2 * theta - np.pi
        data_b = np.concatenate([
            r_b * np.cos(theta), 
            r_b * np.sin(theta)
        ], axis=1)
        
        # Combinar y normalizar
        X = np.concatenate([data_a, data_b])
        X += np.random.randn(*X.shape) * 0.5
        y = np.concatenate([
            np.zeros((n_puntos, 1)), 
            np.ones((n_puntos, 1))
        ])
        
        # Mezclar datos
        indices = np.random.permutation(len(X))
        return X[indices].astype(np.float32), y[indices].astype(np.float32)
    
    def construir_modelo(self):
        """Red neuronal con 2 capas ocultas"""
        # Placeholders
        self.X = tf.placeholder(tf.float32, [None, 2], name="entrada")
        self.y = tf.placeholder(tf.float32, [None, 1], name="etiquetas")
        
        # Arquitectura: 2 -> 16 -> 8 -> 1
        with tf.variable_scope("red_neuronal"):
            # Capa 1
            capa1 = tf.layers.dense(
                self.X, 16, 
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.1),
                name="capa_oculta_1"
            )
            
            # Dropout para regularización
            capa1_dropout = tf.layers.dropout(capa1, rate=0.2)
            
            # Capa 2
            capa2 = tf.layers.dense(
                capa1_dropout, 8,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.1),
                name="capa_oculta_2"
            )
            
            # Capa de salida
            self.logits = tf.layers.dense(
                capa2, 1,
                kernel_initializer=tf.random_normal_initializer(0, 0.1),
                name="salida"
            )
            
            self.prediccion = tf.nn.sigmoid(self.logits)
        
        # Función de pérdida y optimizador
        self.perdida = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.y, 
                logits=self.logits
            )
        )
        
        # Métricas
        predicciones_binarias = tf.cast(self.prediccion > 0.5, tf.float32)
        self.precision = tf.reduce_mean(
            tf.cast(tf.equal(predicciones_binarias, self.y), tf.float32)
        )
        
        # Optimizador con learning rate adaptativo
        self.global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            0.01, self.global_step, 100, 0.96
        )
        self.optimizador = tf.train.AdamOptimizer(learning_rate).minimize(
            self.perdida, global_step=self.global_step
        )
        
    def entrenar(self, X_train, y_train, epocas=500):
        """Entrena el modelo y guarda métricas"""
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for epoca in range(epocas):
                _, perdida_actual, precision_actual = sess.run(
                    [self.optimizador, self.perdida, self.precision],
                    feed_dict={self.X: X_train, self.y: y_train}
                )
                
                self.historia['perdida'].append(perdida_actual)
                self.historia['precision'].append(precision_actual)
                
                if epoca % 50 == 0:
                    print(f"📈 Época {epoca}: Pérdida={perdida_actual:.4f}, "
                          f"Precisión={precision_actual:.2%}")
            
            # Guardar predicciones finales para visualización
            self.predicciones_finales = sess.run(
                self.prediccion, 
                feed_dict={self.X: X_train}
            )
    
    def visualizar_resultados(self, X, y):
        """Crea visualización interactiva del entrenamiento"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Datos originales
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y.ravel(), 
                              cmap='coolwarm', alpha=0.6, s=30)
        ax1.set_title('Datos Originales: Problema de las Espirales')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1)
        
        # 2. Predicciones del modelo
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(X[:, 0], X[:, 1], 
                              c=self.predicciones_finales.ravel(),
                              cmap='coolwarm', alpha=0.6, s=30)
        ax2.set_title('Predicciones del Modelo')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2)
        
        # 3. Curva de pérdida
        ax3 = axes[1, 0]
        ax3.plot(self.historia['perdida'], 'b-', linewidth=2)
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Pérdida')
        ax3.set_title('Evolución de la Pérdida')
        ax3.grid(True, alpha=0.3)
        
        # 4. Curva de precisión
        ax4 = axes[1, 1]
        ax4.plot(self.historia['precision'], 'g-', linewidth=2)
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Precisión')
        ax4.set_title('Evolución de la Precisión')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.show()

# Ejecutar el proyecto completo
if __name__ == "__main__":
    print("🤖 Iniciando Red Neuronal Visual...")
    
    # Crear y entrenar modelo
    modelo = RedNeuronalVisual()
    X, y = modelo.generar_datos_espiral(300)
    
    print(f"📊 Datos generados: {X.shape[0]} muestras")
    print("🎯 Iniciando entrenamiento...")
    
    modelo.entrenar(X, y, epocas=500)
    modelo.visualizar_resultados(X, y)
    
    print("✅ ¡Entrenamiento completado!")
```

### Conceptos Implementados
- **Arquitectura multicapa**: Input → Hidden(16) → Hidden(8) → Output(1)
- **Activaciones no lineales**: ReLU para capas ocultas, Sigmoid para salida
- **Regularización**: Dropout para prevenir overfitting
- **Learning rate adaptativo**: Decaimiento exponencial
- **Visualización en tiempo real**: Matplotlib para tracking de métricas

## Proyecto 3: Generador de Texto Mini-Shakespeare (40 minutos)

### Objetivo
Crear un modelo LSTM que aprenda patrones de texto y genere frases con estilo shakesperiano.

### Código Completo Simplificado

```python
import tensorflow as tf
import numpy as np
import random
import textwrap

class GeneradorTextoMinimo:
    def __init__(self, texto_ejemplo=None):
        """Inicializa con texto de ejemplo o usa Shakespeare simplificado"""
        if texto_ejemplo is None:
            self.texto = """
            To be or not to be that is the question
            Whether tis nobler in the mind to suffer
            The slings and arrows of outrageous fortune
            Or to take arms against a sea of troubles
            Love looks not with the eyes but with the mind
            And therefore is winged Cupid painted blind
            All the worlds a stage and all the men and women merely players
            They have their exits and their entrances
            """
        else:
            self.texto = texto_ejemplo
            
        self.preparar_datos()
        self.construir_modelo()
        
    def preparar_datos(self):
        """Convierte texto a secuencias numéricas"""
        # Crear vocabulario
        self.chars = sorted(list(set(self.texto)))
        self.char_a_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_a_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        print(f"📚 Vocabulario: {self.vocab_size} caracteres únicos")
        
        # Convertir texto a índices
        self.texto_indices = [self.char_a_idx[ch] for ch in self.texto]
        
        # Crear secuencias de entrenamiento
        self.seq_length = 40
        self.crear_secuencias_entrenamiento()
        
    def crear_secuencias_entrenamiento(self):
        """Genera pares entrada-salida para entrenamiento"""
        self.X = []
        self.y = []
        
        for i in range(len(self.texto_indices) - self.seq_length):
            secuencia = self.texto_indices[i:i + self.seq_length]
            siguiente = self.texto_indices[i + self.seq_length]
            
            self.X.append(secuencia)
            self.y.append(siguiente)
            
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        print(f"📝 Secuencias de entrenamiento: {len(self.X)}")
        
    def construir_modelo(self):
        """Red LSTM simple pero efectiva"""
        tf.reset_default_graph()
        
        # Placeholders
        self.input_ph = tf.placeholder(tf.int32, [None, self.seq_length])
        self.target_ph = tf.placeholder(tf.int32, [None])
        
        # Embedding layer
        embedding = tf.Variable(
            tf.random_uniform([self.vocab_size, 128], -1.0, 1.0)
        )
        inputs = tf.nn.embedding_lookup(embedding, self.input_ph)
        
        # LSTM cell
        lstm = tf.nn.rnn_cell.BasicLSTMCell(256)
        
        # Desenrollar LSTM
        outputs, states = tf.nn.dynamic_rnn(
            lstm, inputs, dtype=tf.float32
        )
        
        # Usar solo la última salida
        output = outputs[:, -1, :]
        
        # Capa de salida
        W = tf.Variable(tf.truncated_normal([256, self.vocab_size], stddev=0.1))
        b = tf.Variable(tf.zeros([self.vocab_size]))
        
        self.logits = tf.matmul(output, W) + b
        self.prediccion = tf.nn.softmax(self.logits)
        
        # Pérdida y optimización
        self.perdida = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, 
                labels=self.target_ph
            )
        )
        
        self.optimizador = tf.train.AdamOptimizer(0.001).minimize(self.perdida)
        
    def entrenar(self, epocas=100, batch_size=32):
        """Entrena el modelo mostrando progreso"""
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.sess = sess  # Guardar sesión para generación
            
            n_batches = len(self.X) // batch_size
            
            print("🎭 Entrenando generador estilo Shakespeare...")
            print("-" * 50)
            
            for epoca in range(epocas):
                perdida_total = 0
                
                # Mezclar datos
                indices = np.random.permutation(len(self.X))
                X_mezclado = self.X[indices]
                y_mezclado = self.y[indices]
                
                for i in range(n_batches):
                    inicio = i * batch_size
                    fin = inicio + batch_size
                    
                    X_batch = X_mezclado[inicio:fin]
                    y_batch = y_mezclado[inicio:fin]
                    
                    _, perdida = sess.run(
                        [self.optimizador, self.perdida],
                        feed_dict={
                            self.input_ph: X_batch,
                            self.target_ph: y_batch
                        }
                    )
                    
                    perdida_total += perdida
                
                if epoca % 10 == 0:
                    perdida_promedio = perdida_total / n_batches
                    print(f"Época {epoca:3d}: Pérdida = {perdida_promedio:.4f}")
                    
                    # Generar texto de muestra
                    if epoca > 0:
                        muestra = self.generar_texto(sess, longitud=100)
                        print(f"  Muestra: '{muestra}'")
                        print("-" * 50)
            
            # Generar texto final
            print("\n🎨 TEXTO GENERADO FINAL:")
            print("=" * 50)
            texto_final = self.generar_texto(sess, longitud=200)
            
            # Formatear bonito
            for linea in textwrap.wrap(texto_final, width=50):
                print(f"  {linea}")
            print("=" * 50)
            
    def generar_texto(self, sess, semilla=None, longitud=100, temperatura=1.0):
        """Genera texto nuevo basado en el modelo entrenado"""
        if semilla is None:
            # Usar secuencia aleatoria del texto original
            inicio = random.randint(0, len(self.texto_indices) - self.seq_length)
            secuencia = self.texto_indices[inicio:inicio + self.seq_length]
        else:
            # Convertir semilla a secuencia
            secuencia = [self.char_a_idx.get(ch, 0) for ch in semilla[-self.seq_length:]]
            secuencia = secuencia + [0] * (self.seq_length - len(secuencia))
        
        texto_generado = []
        
        for _ in range(longitud):
            # Preparar entrada
            x = np.array([secuencia])
            
            # Predecir siguiente carácter
            predicciones = sess.run(
                self.prediccion,
                feed_dict={self.input_ph: x}
            )[0]
            
            # Aplicar temperatura para controlar aleatoriedad
            predicciones = np.log(predicciones + 1e-10) / temperatura
            predicciones = np.exp(predicciones)
            predicciones = predicciones / np.sum(predicciones)
            
            # Muestrear siguiente carácter
            siguiente_idx = np.random.choice(len(predicciones), p=predicciones)
            
            # Agregar a texto generado
            texto_generado.append(self.idx_a_char[siguiente_idx])
            
            # Actualizar secuencia
            secuencia = secuencia[1:] + [siguiente_idx]
        
        return ''.join(texto_generado)

# Función principal divertida
def main():
    print("🎭 GENERADOR DE TEXTO SHAKESPERIANO MÍNIMO")
    print("=" * 50)
    
    # Opción de texto personalizado
    print("\n¿Quieres usar tu propio texto? (s/n): ", end="")
    respuesta = input().lower()
    
    if respuesta == 's':
        print("Ingresa tu texto (mínimo 200 caracteres):")
        texto_usuario = input()
        if len(texto_usuario) < 200:
            print("⚠️ Texto muy corto, usando Shakespeare por defecto")
            generador = GeneradorTextoMinimo()
        else:
            generador = GeneradorTextoMinimo(texto_usuario)
    else:
        generador = GeneradorTextoMinimo()
    
    # Entrenar
    generador.entrenar(epocas=100, batch_size=32)
    
    print("\n✨ ¡Modelo entrenado! Tu IA ahora puede escribir.")

if __name__ == "__main__":
    main()
```

### Conceptos Avanzados Implementados
- **LSTM (Long Short-Term Memory)**: Memoria a largo plazo para secuencias
- **Embeddings**: Representación densa de caracteres
- **Muestreo con temperatura**: Control de creatividad vs coherencia
- **Generación autoregresiva**: Cada output alimenta el siguiente input

## Pacing y Estrategias de Aprendizaje

### Ruta Recomendada por Nivel

**Principiante Total (3-4 horas)**
1. Ejecuta cada proyecto tal cual está
2. Modifica parámetros visuales (colores, tamaños)
3. Cambia textos de entrada y observa resultados
4. Experimenta con hiperparámetros básicos

**Intermedio (2-3 horas)**
1. Ejecuta y comprende el flujo de datos
2. Modifica arquitecturas (añade capas, cambia neuronas)
3. Implementa diferentes funciones de activación
4. Añade métricas adicionales de evaluación

**Avanzado (1-2 horas)**
1. Enfócate en optimizaciones de rendimiento
2. Implementa regularización adicional (L1/L2, batch norm)
3. Experimenta con arquitecturas alternativas
4. Integra con APIs o aplicaciones web

### Puntos de Extensión

**Proyecto 1 → Aplicación**
- Crea una API REST que ejecute grafos TensorFlow
- Visualización interactiva con D3.js del grafo computacional

**Proyecto 2 → Portfolio**
- Adapta para clasificar imágenes simples (MNIST)
- Crea visualización web interactiva del proceso de aprendizaje
- Implementa early stopping y validación cruzada

**Proyecto 3 → Producto**
- Entrena con corpus más grandes (libros completos)
- Crea bot de Twitter con personalidad única
- Implementa modelo bidireccional para mejor contexto

## Troubleshooting Común

### Error: "No module named tensorflow"
```bash
pip install tensorflow==1.15  # Para compatibilidad con código legacy
# O para TensorFlow 2.x:
pip install tensorflow
# Luego añade al inicio del código:
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

### Error: Memoria insuficiente
- Reduce batch_size a la mitad
- Disminuye el número de neuronas en capas ocultas
- Usa tf.data.Dataset para procesamiento por lotes

### Modelo no converge
- Normaliza datos de entrada: `(X - X.mean()) / X.std()`
- Reduce learning rate: `0.001 → 0.0001`
- Aumenta épocas de entrenamiento
- Verifica que los datos estén correctamente formateados

## Recursos para Profundizar

**Documentación Esencial**
- TensorFlow Playground: playground.tensorflow.org (visualización interactiva)
- Guías oficiales TF 2.x: tensorflow.org/tutorials
- Colab notebooks gratuitos: colab.research.google.com

**Siguientes Pasos Recomendados**
1. **Computer Vision**: Implementa CNN para CIFAR-10
2. **NLP Avanzado**: Transformers y atención
3. **Reinforcement Learning**: Agente que juega juegos simples
4. **GANs**: Generación de imágenes sintéticas

## Conclusión y Llamada a la Acción

Has construido tres modelos de IA funcionales en menos de 2 horas. Cada uno demuestra conceptos fundamentales que se escalan a aplicaciones de producción. El Proyecto 1 es la base de sistemas de procesamiento distribuido, el Proyecto 2 escala a reconocimiento de imágenes médicas, y el Proyecto 3 es el fundamento de GPT y modelos de lenguaje modernos.

**Tu siguiente desafío**: Toma uno de estos proyectos y conviértelo en una aplicación web desplegable. Usa Flask/FastAPI para el backend y crea una interfaz que permita a usuarios no técnicos interactuar con tu IA.

Recuerda: La mejor forma de aprender ML es construyendo. Cada línea de código que modificas te acerca más a la maestría.