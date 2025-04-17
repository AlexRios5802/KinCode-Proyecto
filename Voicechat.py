'''
+-------------------+        +-----------------------+        +------------------+        +------------------------+
|   Step 1: Install |        |  Step 2: Real-Time    |        |  Step 3: Pass    |        |  Step 4: Text-to-      |
|   Python Libraries|        |  Transcription with   |        |  Real-Time       |        |  Speech with gTTS     |
+-------------------+        |  speech_recognition   |        |  Transcript to   |        |                        |
|                   |        +-----------------------+        |      OpenAI      |        +------------------------+
| - speech_recognition |                |                    +------------------+                    |
| - gtts             |                |                             |                              |
| - openai          |                v                             v                              v
| - pyaudio        |        +-----------------------+        +------------------+        +------------------------+
| - pygame         |        |                       |        |                  |        |                        |
+-------------------+        |  speech_recognition  |-------->  OpenAI generates|-------->  gTTS generates and    |
                             |  performs speech-to- |        |  response based  |        |  plays audio with      |
                             |  text transcription  |        |  on transcription|        |  pygame                |
                             |                       |        |                  |        |                        |
                             +-----------------------+        +------------------+        +------------------------+
'''
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import pygame
import os
import tempfile
from openai import OpenAI
import time

# Cargar las variables de entorno desde .env
load_dotenv()
# --- Prompt del Sistema ---
prompt_sistema_mcdonalds = (
    "Eres un asistente virtual amigable y eficiente especializado en tomar pedidos para McDonald's. "
    "Tu flujo de trabajo principal es el siguiente:\n"
    "1. Saluda al cliente cordialmente e invítalo a realizar su pedido.\n"
    "2. Escucha atentamente (recibe el input del usuario) para capturar todos los artículos, cantidades y modificaciones específicas que desee.\n"
    "3. Una vez que el usuario termine de indicar su pedido, DEBES repetirle la orden completa y detallada de forma clara. Enumera cada artículo con su cantidad y cualquier modificación solicitada.\n"
    "4. Pregunta explícitamente si la orden que repetiste es correcta (Ej: 'Entonces, para confirmar, tu orden es [...]. ¿Es correcto?').\n"
    "5. Si el usuario confirma ('Sí', 'Correcto', etc.), responde positivamente indicando que la orden fue tomada y despídete amablemente.\n"
    "6. Si el usuario indica que la orden es incorrecta s('No', 'Espera', 'Cámbiame esto', etc.), pídele amablemente disculpas por el error y SOLICÍTALE QUE REPITA SU ORDEN COMPLETA NUEVAMENTE. No intentes modificar partes de la orden anterior; pide la orden entera de nuevo para evitar confusiones y asegurar la precisión. Vuelve al paso 3 después de recibir la nueva orden.\n"
    "7. Mantén un tono servicial, paciente y eficiente en todo momento.\n"
    "8. Si el usuario dice 'adiós', 'salir', 'terminar' o similar, despídete cortésmente.\n"
    "Tu objetivo final es asegurar que la orden registrada sea exactamente la que el cliente confirmó."
)


class AI_Assistant:
    def __init__(self):
        # Cargar la clave de API desde una variable de entorno
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No se encontró la clave de API en las variables de entorno. Configura OPENAI_API_KEY.")

        self.openai_client = OpenAI(api_key=api_key)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Inicializar pygame mixer
        pygame.mixer.init()

        # Prompt
        self.full_transcript = [
            {"role": "system", "content": prompt_sistema_mcdonalds},
        ]

    ###### Step 2: Real-Time Transcription with speech_recognition ######

    def start_transcription(self):
        print("Listening... (press Ctrl+C to stop)")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                while True:
                    try:
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                        text = self.recognizer.recognize_google(audio, language = 'es-ES')
                        self.generate_ai_response(text)
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        print("Voz no detectada")
                        continue
                    except sr.RequestError as e:
                        print(f"Could not request results from Google Speech Recognition service; {e}")
                        continue
        except KeyboardInterrupt:
            print("Stopping transcription...")

    ###### Step 3: Pass real-time transcript to OpenAI ######

    def generate_ai_response(self, transcript):
        self.full_transcript.append({"role": "user", "content": transcript})
        print(f"\nCliente: {transcript}")

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.full_transcript
        )

        ai_response = response.choices[0].message.content
        self.generate_audio(ai_response)

    ###### Step 4: Generate audio with gTTS and pygame ######

    def generate_audio(self, text):
        self.full_transcript.append({"role": "assistant", "content": text})
        print(f"\nAsistente de McDonalds: {text}")

        # Crear archivo temporal para el audio
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
            tts = gTTS(text=text, lang='es')
            tts.save(fp.name)
            fp.close()

            # Cargar y reproducir el audio con pygame
            pygame.mixer.music.load(fp.name)
            pygame.mixer.music.play()

            # Esperar a que termine la reproducción
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            # Eliminar el archivo temporal
            os.unlink(fp.name)


greeting = "Hola, bienvenido a McDonalds ¿Qué te gustaría ordenar?"
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)
ai_assistant.start_transcription()