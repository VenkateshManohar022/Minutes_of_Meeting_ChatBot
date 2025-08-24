import os
import warnings
from dotenv import load_dotenv
import gradio as gr
from huggingface_hub import login
from transformers import pipeline
import ollama
from openai import OpenAI

# -----------------------------
# Suppress warnings for cleaner output
# -----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Environment & HuggingFace Login
# -----------------------------
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

# -----------------------------
# Audio Model Setup (Whisper)
# -----------------------------
AUDIO_MODEL = "openai/whisper-small"
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=AUDIO_MODEL,
    # task="transcribe",
)

# -----------------------------
# Ollama / LLaMA Chat Client
# -----------------------------
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
LLAMA_MODEL = "llama3.2"
MAX_TOKENS = 2000

# -----------------------------
# Functions
# -----------------------------
def chat_with_ai(message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=messages,
        stream=True
    )

    partial_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            partial_response += delta.content
            yield partial_response
    yield partial_response



# The wrapper function is not needed. The function can be called directly
def audio_transcribe_and_summarize(audio_file):
    try:
        # First yield: Indicate that transcription is starting
        yield "Starting transcription..."

        # 1. Transcribe the audio
        transcription = asr_pipe(audio_file, return_timestamps=True)
        transcript_text = transcription["text"]
        
        # Second yield: Indicate that summarization is starting
        yield f"Transcription complete. Concluding the Content:\n\n Sharing Few Lines from the Meeting : {gr.Markdown(transcript_text)} \n\n ### Starting summarization on Meeting Points..."
        # yield f"Transcription complete. Transcribing to English:\n\n{transcript_text}\n\nStarting summarization..."

        # 2. Prepare the prompt for summarization
        system_message = "You are an assistant that produces minutes of meetings..."
        user_prompt = f"Below is a meeting transcript. Please write minutes in markdown...\n\n{transcript_text}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]

        # 3. Stream the summary from the LLM
        partial_response = ""
        stream = ollama.chat(
            model=LLAMA_MODEL,
            messages=messages,
            options={"num_ctx": 8192, "num_predict": MAX_TOKENS},
            stream=True
        )
        
        for chunk in stream:
            delta = chunk['message']['content']
            if delta:
                partial_response += delta
                yield partial_response
        
    except Exception as e:
        yield f"Error: {e}"




# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🗣️ Chat with AI & Meeting Minutes Generator")

    # Chat tab
    with gr.Tab("Chat"):
        chat = gr.ChatInterface(fn=chat_with_ai, type="messages")

    # Audio → Transcript & Summary tab
    with gr.Tab("Audio → Transcript & Summary"):
        audio_input = gr.Audio(label="Upload English Audio", type="filepath")
        summary_output = gr.Markdown(value=" ")
        generate_button = gr.Button("Transcribe & Summarize")

        generate_button.click(
            fn=audio_transcribe_and_summarize,
            inputs=audio_input,
            outputs=summary_output
        )

# Launch
demo.launch(share=True)