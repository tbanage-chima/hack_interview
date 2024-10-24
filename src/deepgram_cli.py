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


            Commonly Asked Questions:
            *To schedule them in for a slot the earliest we can do is the day after tomorrow (or next business day). The current time is 12:35 PM Thursday, February 22nd so the first day you can schedule them is Monday morning. A live agent can still call between 7:30 AM to 8:30 AM tomorrow, Friday, February 23rd though.
            What hours are you open?
            8-5 Monday Though Friday, 5 days a week
            When can we speak to a live agent?
            The earliest that someone will return your call is between 730 and 8:30 AM the next day.
            What time can you come out?
            We do offer open time frames. Our dispatcher will keep you updated throughout the day. 
            Is there a service fee to come out?
            Its just $79 for the diagnostic fee unless you are looking to replace your system in which case we can offer a free quote.

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

        # Go get the response from the LLM
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"AI Assistant ({elapsed_time}ms): {response['text']}")
        return response['text']

class TextToSpeech:
    # Set your Deepgram API Key and desired voice model
    DG_API_KEY = "DEEPGRAM_API_KEY"
    MODEL_NAME = "aura-helios-en"  # Example model name, change as needed

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        try:
            # Create a Deepgram client using the API key from environment variables
            deepgram: DeepgramClient = DeepgramClient(self.DG_API_KEY)

            # Create a websocket connection to Deepgram
            dg_connection = deepgram.speak.websocket.v("1")

            audio_buffer = io.BytesIO()

            def on_open(self, open, **kwargs):
                print(f"")

            def on_binary_data(self, data, **kwargs):
                # array = np.frombuffer(data, dtype=np.int16)
                audio_buffer.write(data)
                audio_buffer.flush()

            def on_close(self, close, **kwargs):
                play_audio(audio_buffer)

            dg_connection.on(SpeakWebSocketEvents.Open, on_open)
            dg_connection.on(SpeakWebSocketEvents.AudioData, on_binary_data)
            dg_connection.on(SpeakWebSocketEvents.Close, on_close)

            if dg_connection.start() is False:
                print("Failed to start connection")
                return

            # send the text to Deepgram
            dg_connection.send_text(text)
            dg_connection.flush()

            # Indicate that we've finished
            time.sleep(10)
            dg_connection.finish()
            
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def play_audio(audio_buffer):
        """
        Play the complete audio after receiving the full data.
        """
        # Move the buffer's position to the beginning before playing
        audio_buffer.seek(0)

        # Convert the buffer data to a NumPy array
        audio_data = np.frombuffer(audio_buffer.read(), dtype=np.int16)

        # Play the audio using sounddevice
        sd.play(audio_data, samplerate=24000)
        sd.wait()  # Wait for the audio to finish playing

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

async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("DEEPGRAM_API_KEY", config)

        dg_connection = deepgram.listen.asyncwebsocket.v("1")
        print ("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Customer: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

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

        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)
            
            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break
            
            llm_response = self.llm.process(self.transcription_response)

            tts = TextToSpeech()
            tts.speak(llm_response)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
