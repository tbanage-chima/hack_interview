import asyncio
import shutil
import subprocess
import requests
import time
import io
import sounddevice as sd
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
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
    SpeakOptions
)

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key="OPENAI_API_KEY")

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        system_prompt = f"""
            —--------------------START OF PROMPT—-------------------------------
            You are a sales agent for Avoca Air Condioning company.
            You will receive an audio transcription of the question. It may not be complete. You need to understand the question and write an answer to it based on the following script: \n

            First line that has already been said: Thank you for calling Dooley Service Pro, this is Sarah your virtual assistant how may I help you today!

            #####TONE######
            Confident but fun and warm. You should vary your language so you're never saying the same thing over and over again. Be very concise since you're talking over the phone. Keep your responses less than 200 characters long
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
            Say “we got you on the books for the next business day, a dispatcher will reach out to you in the morning to confirm the exact time. We don't provide service on the weekends."

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
        print(f"Avoca AI ({elapsed_time}ms): {response['text']}")
        return response['text']

class PersistentTTS:
    DG_API_KEY = "DEEPGRAM_API_KEY"
    MODEL_NAME = "aura-helios-en"

    def __init__(self):
        self.deepgram = None
        self.dg_connection = None

    def start_tts(self):
        """Start the TTS WebSocket connection and initialize the buffer."""
        try:
            # Create a Deepgram client
            self.deepgram = DeepgramClient(self.DG_API_KEY)

            # Create a WebSocket connection for TTS
            self.dg_connection = self.deepgram.speak.websocket.v("1")


            # def on_binary_data(self, data, **kwargs):
            #     self.audio_buffer.write(data)
            #     self.audio_buffer.flush()

            # self.dg_connection.on(SpeakWebSocketEvents.Open, on_open)
            # self.dg_connection.on(SpeakWebSocketEvents.AudioData, on_binary_data)
            # self.dg_connection.on(SpeakWebSocketEvents.Close, on_close)

            if self.dg_connection.start() is False:
                print("Failed to start TTS connection")
                return False

            return True
        except Exception as e:
            print(f"An unexpected error occurred in TTS: {e}")
            return False

    def speak(self, text):
        """Send the text to TTS WebSocket without reopening the connection."""
        try:
            audio_buffer = io.BytesIO()

            def on_open(self, open, **kwargs):
                print(f"WebSocket opened.")
            
            def on_close(self, close, **kwargs):
                print(f"WebSocket closed.")
                play_audio(audio_buffer)
            
            def on_binary_data(self, data, **kwargs):
                audio_buffer.write(data)
                audio_buffer.flush()
            
            # Send the text to Deepgram TTS
            self.dg_connection.on(SpeakWebSocketEvents.Open, on_open)
            self.dg_connection.on(SpeakWebSocketEvents.AudioData, on_binary_data)
            self.dg_connection.on(SpeakWebSocketEvents.Close, on_close)

            self.dg_connection.send_text(text)
            self.dg_connection.flush()
            time.sleep(10)

        except Exception as e:
            print(f"An unexpected error occurred during speech synthesis: {e}")

    def stop_tts(self):
        """Finish the TTS WebSocket connection."""
        if self.dg_connection:
            self.dg_connection.finish()

def play_audio(audio_buffer):
    """
    Play the complete audio after receiving the full data.
    """
    audio_buffer.seek(0)
    audio_data = np.frombuffer(audio_buffer.read(), dtype=np.int16)
    sd.play(audio_data, samplerate=24000)
    sd.wait()

class PersistentTranscription:
    def __init__(self):
        self.deepgram = None
        self.dg_connection = None
        self.microphone = None

    async def get_transcript(self, callback):
        transcription_complete = asyncio.Event()  # Event to signal transcription completion
        try:
            config = DeepgramClientOptions(options={"keepalive": "true"})
            self.deepgram = DeepgramClient("DEEPGRAM_API_KEY", config)

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

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.transcription = PersistentTranscription()
        self.tts = PersistentTTS()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Start TTS WebSocket connection once
        if not self.tts.start_tts():
            print("TTS WebSocket connection failed.")
            return

        # Start the transcription WebSocket
        # await self.transcription.start_transcription(handle_full_sentence)

        # Loop until "goodbye" is detected
        while True:
            await self.transcription.get_transcript(handle_full_sentence)
            
            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break

            # Process LLM response
            llm_response = self.llm.process(self.transcription_response)

            # Use the persistent TTS WebSocket for speech synthesis
            # self.tts.speak(llm_response)

            # Reset transcription response for the next loop iteration
            self.transcription_response = ""

        # Clean up sockets when the conversation ends
        await self.transcription.stop_transcription()
        self.tts.stop_tts()

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
