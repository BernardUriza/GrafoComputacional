#!/usr/bin/env python3
"""
Script de prueba épica de GPU con PyTorch
Creado por Bernard Orozco
"""

import torch
import time
import sys

def test_gpu_power():
    print("=" * 70)
    print("🚀 PRUEBA ÉPICA DE GPU - by Gandalf the GPU Master")
    print("=" * 70)
    
    # Información básica
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible. Usando CPU.")
        return False
    
    # Información de GPU
    gpu_count = torch.cuda.device_count()
    print(f"GPUs disponibles: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Prueba de rendimiento
    print("\n🔥 INICIANDO PRUEBA DE RENDIMIENTO...")
    
    # Configurar dispositivo
    device = torch.device('cuda:0')
    print(f"Usando dispositivo: {device}")
    
    # Crear tensores grandes
    size = 4096
    print(f"Creando matrices {size}x{size}...")
    
    # En GPU
    start_time = time.time()
    a_gpu = torch.randn(size, size, device=device)
    b_gpu = torch.randn(size, size, device=device)
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()  # Esperar a que termine
    gpu_time = time.time() - start_time
    
    # En CPU para comparar
    start_time = time.time()
    a_cpu = torch.randn(size, size, device='cpu')
    b_cpu = torch.randn(size, size, device='cpu')
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    
    # Resultados
    print(f"\n📊 RESULTADOS:")
    print(f"GPU Time: {gpu_time:.4f} segundos")
    print(f"CPU Time: {cpu_time:.4f} segundos")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x más rápido en GPU!")
    
    # Uso de memoria GPU
    memory_used = torch.cuda.memory_allocated(device) / 1024**3
    memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"Memoria GPU usada: {memory_used:.2f} GB / {memory_total:.1f} GB")
    
    # Prueba de gradientes (para redes neuronales)
    print(f"\n🧠 PROBANDO GRADIENTES...")
    x = torch.randn(1000, 1000, device=device, requires_grad=True)
    y = x.sum()
    y.backward()
    print(f"✅ Gradientes calculados correctamente en GPU!")
    
    print(f"\n🎉 ¡GPU FUNCIONANDO PERFECTAMENTE!")
    print(f"🔥 Tu RTX 2000 Ada está lista para dominar grafos computacionales!")
    
    return True

if __name__ == "__main__":
    success = test_gpu_power()
    if success:
        print("\n🧙‍♂️ Gandalf approves: GPU setup complete!")
    else:
        print("\n😞 Necesitamos seguir trabajando...")