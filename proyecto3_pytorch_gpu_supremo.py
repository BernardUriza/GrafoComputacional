#!/usr/bin/env python3
"""
PROYECTO 3: GENERADOR DE TEXTO SUPREMO CON PYTORCH GPU
LA VENGANZA CONTRA TENSORFLOW
Creado por Bernard Orozco - Powered by PyTorch GPU SUPREMACY
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import time
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class GeneradorSupremo(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=3, dropout=0.3):
        super(GeneradorSupremo, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Embedding más poderoso
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM multicapa con más poder
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            dropout=dropout,
            batch_first=True
        )
        
        # Capas de salida más sofisticadas
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, vocab_size)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Tomar solo la última salida
        lstm_out = lstm_out[:, -1, :]
        
        # Capas densas
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))

class PyTorchSupremo:
    def __init__(self, archivo_texto):
        # Configurar dispositivo GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DISPOSITIVO SUPREMO: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU DETECTADA: {torch.cuda.get_device_name(0)}")
            print(f"MEMORIA GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("WARNING: No hay GPU, usando CPU (pero PyTorch es mejor que TensorFlow igual)")
        
        # Cargar texto
        print(f"Cargando texto desde: {archivo_texto}")
        with open(archivo_texto, 'r', encoding='utf-8') as f:
            self.texto = f.read()
        print(f"TEXTO CARGADO: {len(self.texto):,} caracteres")
        
        self.preparar_datos()
        self.modelo = None
        
    def preparar_datos(self):
        # Procesar texto
        self.texto = self.texto.lower().strip()
        
        # Crear vocabulario
        self.chars = sorted(list(set(self.texto)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        print(f"ESTADISTICAS DEL PODER:")
        print(f"  Caracteres totales: {len(self.texto):,}")
        print(f"  Vocabulario: {self.vocab_size} caracteres únicos")
        
        # Convertir texto a indices
        self.texto_indices = [self.char_to_idx[ch] for ch in self.texto]
        
        # Crear secuencias de entrenamiento
        self.seq_length = 50  # Secuencias más largas para mayor contexto
        self.crear_secuencias()
        
    def crear_secuencias(self):
        sequences = []
        targets = []
        
        # Crear secuencias con paso más pequeño para más datos
        step = 2
        for i in range(0, len(self.texto_indices) - self.seq_length, step):
            seq = self.texto_indices[i:i + self.seq_length]
            target = self.texto_indices[i + self.seq_length]
            sequences.append(seq)
            targets.append(target)
        
        self.sequences = np.array(sequences)
        self.targets = np.array(targets)
        
        print(f"SECUENCIAS DE ENTRENAMIENTO: {len(self.sequences):,}")
        print(f"FORMA DE DATOS: {self.sequences.shape}")
        
    def crear_modelo(self):
        print("CONSTRUYENDO MODELO SUPREMO...")
        
        # Parámetros adaptativos según el tamaño del dataset
        embedding_dim = min(256, max(128, self.vocab_size * 4))
        hidden_dim = min(512, max(256, len(self.sequences) // 1000))
        
        print(f"  Embedding: {embedding_dim} dimensiones")
        print(f"  LSTM Hidden: {hidden_dim} unidades")
        print(f"  Capas LSTM: 3")
        
        self.modelo = GeneradorSupremo(
            vocab_size=self.vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            dropout=0.3
        ).to(self.device)
        
        # Optimizer más potente
        self.optimizer = optim.AdamW(
            self.modelo.parameters(), 
            lr=0.001,
            weight_decay=1e-5
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Scheduler para ajustar learning rate
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10, 
            gamma=0.8
        )
        
        print("MODELO SUPREMO CONSTRUIDO!")
        print(f"PARÁMETROS TOTALES: {sum(p.numel() for p in self.modelo.parameters()):,}")
        
    def entrenar(self, epocas=50, batch_size=128):
        if self.modelo is None:
            self.crear_modelo()
        
        # Crear dataset y dataloader
        dataset = TextDataset(self.sequences, self.targets)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2 if self.device.type == 'cuda' else 0
        )
        
        print(f"ENTRENAMIENTO SUPREMO INICIADO!")
        print(f"  Épocas: {epocas}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches por época: {len(dataloader)}")
        print("=" * 60)
        
        self.modelo.train()
        
        for epoca in range(epocas):
            epoca_loss = 0
            epoca_acc = 0
            batches = 0
            
            start_time = time.time()
            
            for batch_idx, (data, targets) in enumerate(dataloader):
                # Mover a GPU
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs, _ = self.modelo(data)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(self.modelo.parameters(), max_norm=5.0)
                
                self.optimizer.step()
                
                # Calcular accuracy
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == targets).float().mean()
                
                epoca_loss += loss.item()
                epoca_acc += accuracy.item()
                batches += 1
            
            # Actualizar learning rate
            self.scheduler.step()
            
            # Estadísticas de la época
            avg_loss = epoca_loss / batches
            avg_acc = epoca_acc / batches
            epoch_time = time.time() - start_time
            
            if epoca % 5 == 0 or epoca == epocas - 1:
                print(f"Época {epoca:3d}/{epocas}: "
                      f"Loss={avg_loss:.4f}, Acc={avg_acc:.3f}, "
                      f"LR={self.scheduler.get_last_lr()[0]:.6f}, "
                      f"Tiempo={epoch_time:.1f}s")
                
                # Generar muestra cada 10 épocas
                if epoca % 10 == 0 and epoca > 0:
                    muestra = self.generar_texto(longitud=100, temperatura=0.8)
                    print(f"  MUESTRA: '{muestra[:80]}...'")
                    print("-" * 60)
        
        print("ENTRENAMIENTO COMPLETADO CON GLORIA!")
        
    def generar_texto(self, longitud=300, temperatura=0.8, semilla=None):
        if self.modelo is None:
            return "Error: Modelo no entrenado"
        
        self.modelo.eval()
        
        with torch.no_grad():
            # Secuencia inicial
            if semilla:
                semilla_clean = semilla.lower()
                sequence = [self.char_to_idx.get(ch, 0) for ch in semilla_clean[-self.seq_length:]]
                sequence = sequence + [0] * (self.seq_length - len(sequence))
            else:
                start_idx = random.randint(0, len(self.texto_indices) - self.seq_length)
                sequence = self.texto_indices[start_idx:start_idx + self.seq_length]
            
            generated_text = []
            
            for _ in range(longitud):
                # Convertir secuencia a tensor
                input_seq = torch.tensor([sequence], dtype=torch.long).to(self.device)
                
                # Predecir
                output, _ = self.modelo(input_seq)
                
                # Aplicar temperatura y muestrear
                if temperatura == 0:
                    next_idx = torch.argmax(output, dim=-1).item()
                else:
                    # Aplicar temperatura
                    logits = output[0] / temperature
                    probabilities = torch.softmax(logits, dim=-1)
                    next_idx = torch.multinomial(probabilities, 1).item()
                
                # Agregar carácter generado
                if next_idx < len(self.idx_to_char):
                    generated_text.append(self.idx_to_char[next_idx])
                else:
                    generated_text.append(' ')
                
                # Actualizar secuencia
                sequence = sequence[1:] + [next_idx]
            
            return ''.join(generated_text)

def main():
    print("=" * 70)
    print("   PROYECTO 3: SUPREMACÍA TOTAL DE PYTORCH GPU")
    print("   LA VENGANZA DEFINITIVA CONTRA TENSORFLOW")
    print("   Creado por Bernard Orozco")
    print("=" * 70)
    
    # Verificar archivo
    archivo = "The+48+Laws+Of+Power_texto.txt"
    if not os.path.exists(archivo):
        print(f"ERROR: No se encontró {archivo}")
        return
    
    # Crear generador supremo
    generador = PyTorchSupremo(archivo)
    
    # Entrenar con poder supremo
    print("\nINICIANDO ENTRENAMIENTO SUPREMO...")
    generador.entrenar(epocas=40, batch_size=256)
    
    # Generar textos épicos
    temperaturas = [0.3, 0.7, 1.0, 1.5]
    
    print("\n" + "=" * 70)
    print("        TEXTOS GENERADOS POR LA SUPREMACÍA PYTORCH")
    print("=" * 70)
    
    for i, temp in enumerate(temperaturas, 1):
        print(f"\nMUESTRA {i} - Temperatura {temp} (Creatividad {'Baja' if temp < 0.5 else 'Media' if temp < 1.0 else 'Alta' if temp < 1.5 else 'EXTREMA'}):")
        print("-" * 50)
        
        texto = generador.generar_texto(longitud=400, temperatura=temp)
        
        # Formatear texto elegantemente
        palabras = texto.split()
        linea = ""
        for palabra in palabras[:60]:
            if len(linea + palabra) > 70:
                print(f"  {linea.strip()}")
                linea = palabra + " "
            else:
                linea += palabra + " "
        if linea.strip():
            print(f"  {linea.strip()}")
    
    print("\n" + "=" * 70)
    print("     PYTORCH HA DEMOSTRADO SU SUPREMACÍA TOTAL")
    print("     TENSORFLOW PUEDE IRSE A LLORAR A SU CASA")
    print("     BERNARD OROZCO - MAESTRO DE PYTORCH GPU")
    print("=" * 70)

if __name__ == "__main__":
    main()