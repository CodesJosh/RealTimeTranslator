import os
import sys
import site
import pyaudio
import numpy as np
import threading
import tkinter as tk
from tkinter import scrolledtext
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

# --- CONFIGURACIÓN VISUAL ---
OPACIDAD = 0.8  
TAMANO_LETRA = 11
FONDO = "black"
# ----------------------------

class TraductorOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Traductor CS2")
        self.root.geometry("500x250")
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', OPACIDAD)
        self.root.configure(bg=FONDO)

        self.text_area = scrolledtext.ScrolledText(
            self.root, bg=FONDO, fg="white", font=("Consolas", TAMANO_LETRA), bd=0
        )
        self.text_area.pack(expand=True, fill='both', padx=10, pady=10)
        
        self.text_area.tag_config('verde', foreground='#00FF00')
        self.text_area.tag_config('amarillo', foreground='#FFFF00')
        self.text_area.tag_config('cyan', foreground='#00FFFF')
        self.text_area.tag_config('rojo', foreground='#FF5555')

        self.log("🚀 Iniciando sistema (Modo Amplificado)...", 'verde')
        self.thread = threading.Thread(target=self.correr_traductor, daemon=True)
        self.thread.start()

    def log(self, mensaje, tag=None):
        self.text_area.insert(tk.END, mensaje + "\n", tag)
        self.text_area.see(tk.END)

    def buscar_dlls_nvidia(self):
        try:
            rutas_site = site.getsitepackages()
            encontrado = False
            for ruta in rutas_site:
                ruta_nvidia = os.path.join(ruta, "nvidia")
                if os.path.exists(ruta_nvidia):
                    for root, dirs, files in os.walk(ruta_nvidia):
                        if 'bin' in dirs:
                            dll_path = os.path.join(root, 'bin')
                            os.add_dll_directory(dll_path)
                            os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']
                            encontrado = True
            if encontrado: self.log("✅ Drivers NVIDIA vinculados.", 'verde')
        except: pass

    def correr_traductor(self):
        self.buscar_dlls_nvidia()
        try:
            self.log("🧠 Cargando IA...", 'verde')
            model = WhisperModel("small", device="cuda", compute_type="int8")
            self.log("✅ ¡IA LISTA!", 'verde')
        except Exception as e:
            self.log(f"❌ Error IA: {e}", 'rojo')
            return

        CHUNK = 1024 
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000 
        p = pyaudio.PyAudio()

        device_index = None
        try:
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if "CABLE Output" in info["name"]:
                    device_index = i
                    self.log(f"🎤 Escuchando: {info['name']}", 'verde')
                    break
            if device_index is None:
                default = p.get_default_input_device_info()
                device_index = default["index"]
                self.log(f"⚠️ Usando micrófono default: {default['name']}", 'rojo')

            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                            input_device_index=device_index, frames_per_buffer=CHUNK)
        except Exception as e:
            self.log(f"❌ Error Audio: {e}", 'rojo')
            return

        banderas = {'pt': '🇧🇷', 'en': '🇺🇸', 'ru': '🇷🇺', 'fr': '🇫🇷', 'de': '🇩🇪', 'it': '🇮🇹', 'es': '🇨🇱'}
        
        frames = []
        silence_counter = 0
        is_recording = False
        
        SILENCE_THRESHOLD = 20 
        SILENCE_LIMIT = 8        
        AMPLIFICACION = 15.0   
        # ------------------------------------

        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                
                audio_boosted = np.clip(audio_chunk * AMPLIFICACION, -32768, 32767).astype(np.int16)
                
                volumen = np.abs(audio_boosted).mean() 

                if volumen > SILENCE_THRESHOLD:
                    is_recording = True
                    silence_counter = 0
                    frames.append(audio_boosted.tobytes())
                elif is_recording:
                    silence_counter += 1
                    frames.append(audio_boosted.tobytes())
                    if silence_counter > SILENCE_LIMIT:
                        is_recording = False
                        silence_counter = 0
                        
                        if len(frames) > 10:
                            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).flatten().astype(np.float32) / 32768.0
                            
                            segments, info = model.transcribe(audio_data, beam_size=1)
                            idioma = info.language
                            prob = info.language_probability
                            
                            if prob > 0.4 and idioma != 'es':
                                texto_full = ""
                                for s in segments: texto_full += s.text + " "
                                texto_orig = texto_full.strip()
                                
                                if len(texto_orig) > 1:
                                    try:
                                        trad = GoogleTranslator(source=idioma, target='es').translate(texto_orig)
                                        flag = banderas.get(idioma, '🌍')
                                        self.log(f"{flag} ({idioma}): {texto_orig}", 'amarillo')
                                        self.log(f"🇨🇱: {trad}", 'cyan')
                                        self.log("-" * 30)
                                    except: pass
                        frames = []
            except: continue

if __name__ == "__main__":
    app = TraductorOverlay()
    app.root.mainloop()
