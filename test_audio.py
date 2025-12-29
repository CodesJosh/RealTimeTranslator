import pyaudio
import numpy as np

p = pyaudio.PyAudio()

# Buscar Cable
device_index = None
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if "CABLE Output" in info["name"]:
        device_index = i
        print(f"✅ Probando dispositivo: {info['name']}")
        break

if device_index is None:
    print("❌ No encontré el Cable. Usando defecto.")
    device_index = p.get_default_input_device_info()["index"]

stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, input_device_index=device_index, frames_per_buffer=1024)

print("\n--- MEDIDOR DE VOLUMEN (Cierra con Ctrl+C) ---")
print("Si ves todo en 0, es que no entra audio.")

try:
    while True:
        data = stream.read(1024, exception_on_overflow=False)
        # Calcular volumen
        volumen = int(np.abs(np.frombuffer(data, dtype=np.int16)).mean())
        
        # Dibujar barrita
        barra = "█" * (volumen // 10)
        print(f"Nivel: {volumen:03d} {barra}")
except KeyboardInterrupt:
    pass