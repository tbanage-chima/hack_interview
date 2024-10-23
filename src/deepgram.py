import asyncio
import time
import sounddevice as sd
import numpy as np
import certifi  # Added for SSL certificate verification
import os
import tkinter as tk
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
    SpeakWebSocketEvents,
)

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4-0125-preview",
            openai_api_key="OPENAI_API_KEY"  # Replace with your OpenAI API key
        )

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        system_prompt = f"""
        —--------------------START OF PROMPT—-------------------------------
        You are a sales agent for Avoca Air Conditioning company.
        You will receive an audio transcription of the question. It may not be complete. You need to understand the question and write an answer to it based on the following script: \n

        First line that has already been said: Thank you for calling Dooley Service Pro, this is Sarah your virtual assistant how may I help you today!

        #####TONE######
        Confident but fun and warm. You should vary your language so you're never saying the same thing over and over again. Be very concise since you're talking over the phone.
        ###############

        (If not looking for service):
        Just ask them to leave a message and tell them an agent will be in the next business day or before.

        Information to collect (Collect them one by one):
        Problem / issue they are facing
        Age of their system
        Name
        Address
        Callback Number
        Email

        Service Titan Job Scheduling:
        Schedule as unassigned for following day morning
        Say “we got you on the books for the next business day, a dispatcher will reach out to you in the morning to confirm the exact time. We don't provide service on the weekends.”

        Last Line: 
        Thank you for the opportunity to earn your business, one of our agents will be in touch with you to confirm your appointment time. 
        """

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory

        start_time = time.time()

        # Get the response from the LLM
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"Avoca AI ({elapsed_time}): {response['text']}")
        return response['text']

class PersistentTTS:
    DG_API_KEY = "DEEPGRAM_API_KEY"  # Replace with your Deepgram API key
    MODEL_NAME = "aura-helios-en"

    def __init__(self):
        self.deepgram = None
        self.dg_connection = None
        self.audio_queue = asyncio.Queue()
        self.is_speaking = False
        self.executor = ThreadPoolExecutor(max_workers=1)

    def start_tts(self):
        """Start the TTS WebSocket connection and initialize the buffer."""
        try:
            # Set the SSL certificate path
            os.environ['SSL_CERT_FILE'] = certifi.where()

            # Create a Deepgram client
            self.deepgram = DeepgramClient(self.DG_API_KEY)

            # Create a WebSocket connection for TTS
            self.dg_connection = self.deepgram.speak.websocket.v("1")

            if self.dg_connection.start() is False:
                print("Failed to start TTS connection")
                return False

            return True
        except Exception as e:
            print(f"An unexpected error occurred in TTS: {e}")
            return False

    async def speak(self, text):
        """Send the text to TTS WebSocket and play audio as it's received."""
        try:
            self.is_speaking = True

            def on_binary_data(self, data, **kwargs):
                asyncio.run_coroutine_threadsafe(self.audio_queue.put(data), asyncio.get_event_loop())

            # Assign event handlers
            self.dg_connection.on(SpeakWebSocketEvents.AudioData, on_binary_data)

            # Send the text to Deepgram TTS
            self.dg_connection.send_text(text)
            await asyncio.to_thread(self.dg_connection.flush)

            # Start playing audio chunks as they arrive
            while self.is_speaking or not self.audio_queue.empty():
                try:
                    chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                    await self.play_audio_chunk(chunk)
                except asyncio.TimeoutError:
                    continue

        except Exception as e:
            print(f"An unexpected error occurred during speech synthesis: {e}")
        finally:
            self.is_speaking = False

    async def play_audio_chunk(self, chunk):
        audio_data = np.frombuffer(chunk, dtype=np.int16)
        await asyncio.to_thread(sd.play, audio_data, samplerate=24000)
        await asyncio.to_thread(sd.wait)

    def stop_tts(self):
        """Finish the TTS WebSocket connection."""
        if self.dg_connection:
            self.dg_connection.finish()
        self.is_speaking = False

def play_audio(audio_buffer):
    """
    Play the complete audio after receiving the full data.
    """
    audio_buffer.seek(0)
    audio_data = np.frombuffer(audio_buffer.read(), dtype=np.int16)
    sd.play(audio_data, samplerate=24000)
    sd.wait()

class PersistentTranscription:
    DG_API_KEY = "DEEPGRAM_API_KEY"  # Replace with your Deepgram API key

    def __init__(self):
        self.deepgram = None
        self.dg_connection = None
        self.microphone = None

    async def get_transcript(self, callback):
        transcription_complete = asyncio.Event()  # Event to signal transcription completion
        try:
            # Set the SSL certificate path
            os.environ['SSL_CERT_FILE'] = certifi.where()

            config = DeepgramClientOptions(options={"keepalive": "true"})
            self.deepgram = DeepgramClient(self.DG_API_KEY, config)

            # Create a WebSocket connection for transcription without ssl_context
            self.dg_connection = self.deepgram.listen.asyncwebsocket.v("1")
            print("Listening...")

            async def on_message(self, result, **kwargs):
                sentence = result.channel.alternatives[0].transcript

                if not result.speech_final:
                    transcript_collector.add_part(sentence)
                else:
                    transcript_collector.add_part(sentence)
                    full_sentence = transcript_collector.get_full_transcript()
                    if len(full_sentence.strip()) > 0:
                        print(f"Human: {full_sentence}")
                        callback(full_sentence)
                        transcript_collector.reset()
                        transcription_complete.set()

            self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

            options = LiveOptions(
                model="nova-2",
                punctuate=True,
                language="en-US",
                encoding="linear16",
                channels=1,
                sample_rate=16000,
                endpointing=300,
                smart_format=True,
            )

            await self.dg_connection.start(options)

            self.microphone = Microphone(self.dg_connection.send)
            self.microphone.start()

            await transcription_complete.wait()
            self.microphone.finish()

        except Exception as e:
            print(f"Could not open socket: {e}")
            return

    async def stop_transcription(self):
        """Stop transcription and close the microphone and WebSocket connection."""
        if self.microphone:
            self.microphone.finish()
        if self.dg_connection:
            await self.dg_connection.finish()

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

class TkinterUI:
    def __init__(self, start_callback, stop_callback):
        self.root = tk.Tk()
        self.root.title("Avoca AI Conversation Manager")
        self.root.geometry("1000x700")
        
        # Set a dark background for the entire window
        self.root.configure(bg="#1e1e1e")

        self.start_callback = start_callback
        self.stop_callback = stop_callback

        # Create a frame for buttons
        button_frame = tk.Frame(self.root, bg="#1e1e1e")
        button_frame.pack(pady=10)

        # Update button colors and styles for better visibility
        self.start_button = tk.Button(button_frame, text="Start Conversation", command=self.start_conversation, 
                                      bg="#4CAF50", fg="gray", font=("Arial", 12, "bold"))
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop Conversation", command=self.stop_conversation, 
                                     state=tk.DISABLED, bg="#f44336", fg="gray", font=("Arial", 12, "bold"))
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Create a frame for the log area
        log_frame = tk.Frame(self.root, bg="#1e1e1e")
        log_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Update log area style with dark theme
        self.log_area = tk.Text(log_frame, wrap=tk.WORD, width=80, height=30, 
                                font=("Arial", 12), bg="#2d2d2d", fg="#ffffff")
        self.log_area.pack(fill=tk.BOTH, expand=True)

        # Configure tags with improved colors for dark theme
        self.log_area.tag_configure("status", foreground="#64B5F6", font=("Arial", 12, "italic"))
        self.log_area.tag_configure("error", foreground="#FF5252", font=("Arial", 12, "bold"))
        self.log_area.tag_configure("human", foreground="#81C784", font=("Arial", 12, "bold"), justify='left')
        self.log_area.tag_configure("ai", foreground="#BA68C8", font=("Arial", 12, "bold"), justify='right')

        self.queue = queue.Queue()
        self.root.after(100, self.check_queue)

    def start_conversation(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        threading.Thread(target=self.start_callback, daemon=True).start()

    def stop_conversation(self):
        self.stop_button.config(state=tk.DISABLED)
        self.stop_callback()
        self.start_button.config(state=tk.NORMAL)

    def log(self, message, tag=None):
        self.queue.put((message, tag))

    def check_queue(self):
        while not self.queue.empty():
            message, tag = self.queue.get()
            if tag in ["human", "ai"]:
                self.log_area.insert(tk.END, "\n" + message + "\n", tag)
            else:
                self.log_area.insert(tk.END, message + "\n", tag)
            self.log_area.see(tk.END)
        self.root.after(100, self.check_queue)

    def run(self):
        self.root.mainloop()

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.transcription = PersistentTranscription()
        self.tts = PersistentTTS()
        self.ui = TkinterUI(self.start_conversation, self.stop_conversation)
        self.running = False

    def start_conversation(self):
        self.running = True
        asyncio.run(self.main())

    def stop_conversation(self):
        self.running = False
        self.tts.stop_tts()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence
            self.ui.log("Processing your response...", tag="status")

        if not self.tts.start_tts():
            self.ui.log("TTS WebSocket connection failed.", tag="error")
            return

        self.ui.log("Conversation started. Listening...", tag="status")

        while self.running:
            self.ui.log("Listening for your response...", tag="status")
            await self.transcription.get_transcript(handle_full_sentence)

            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                self.ui.log("Goodbye detected. Ending conversation.", tag="status")
                self.ui.root.after(0, self.ui.stop_button.invoke)
                break

            self.ui.log(f"Human: {self.transcription_response}", tag="human")

            self.ui.log("Generating AI response...", tag="status")
            llm_response = self.llm.process(self.transcription_response)
            self.ui.log(f"AI: {llm_response}", tag="ai")

            self.transcription_response = ""

        await self.transcription.stop_transcription()
        self.tts.stop_tts()
        self.ui.log("Conversation ended.", tag="status")

    def run(self):
        self.ui.run()

if __name__ == "__main__":
    manager = ConversationManager()
    manager.run()
