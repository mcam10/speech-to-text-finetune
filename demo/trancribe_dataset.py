from pathlib import Path
from typing import Tuple
import gradio as gr
import soundfile as sf
import pandas as pd
from transformers import pipeline, Pipeline
from speech_to_text_finetune.hf_utils import get_available_languages_in_cv
import shutil
import os

# Constants and configurations
parent_dir = "local_data"
custom_dir = f"{parent_dir}/custom"
transcribed_dir = f"{parent_dir}/transcribed"
languages = get_available_languages_in_cv("mozilla-foundation/common_voice_17_0").keys()
model_ids = [
    "kostissz/whisper-tiny-gl",
    "kostissz/whisper-tiny-el",
    "openai/whisper-tiny",
    "openai/whisper-small",
    "openai/whisper-medium",
]

def save_text_audio_to_file(
    audio_input: gr.Audio,
    sentence: str,
    dataset_dir: str,
) -> Tuple[str, None]:
    """
    Save the audio recording and associated text to files.
    """
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    text_data_path = Path(f"{dataset_dir}/text.csv")
    
    if text_data_path.is_file():
        text_df = pd.read_csv(text_data_path)
    else:
        text_df = pd.DataFrame(columns=["index", "sentence"])
    
    index = len(text_df)
    text_df = pd.concat(
        [text_df, pd.DataFrame([{"index": index, "sentence": sentence}])],
        ignore_index=True,
    )
    text_df = text_df.sort_values(by="index")
    text_df.to_csv(f"{dataset_dir}/text.csv", index=False)
    
    audio_filepath = f"{dataset_dir}/rec_{index}.wav"
    sr, data = audio_input
    sf.write(file=audio_filepath, data=data, samplerate=sr)
    
    return (
        f"""✅ Updated {dataset_dir}/text.csv \n✅ Saved recording to {audio_filepath}""",
        None,
    )

def save_transcribed_pair(
    audio_path: str,
    transcribed_text: str,
    dataset_dir: str = transcribed_dir
) -> str:
    """
    Save the uploaded audio file and its transcription to the dataset.
    """
    if not audio_path or not transcribed_text:
        return "⚠️ Both audio and transcription are required"
    
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    text_data_path = Path(f"{dataset_dir}/text.csv")
    
    if text_data_path.is_file():
        text_df = pd.read_csv(text_data_path)
    else:
        text_df = pd.DataFrame(columns=["index", "sentence"])
    
    index = len(text_df)
    text_df = pd.concat(
        [text_df, pd.DataFrame([{"index": index, "sentence": transcribed_text.strip()}])],
        ignore_index=True,
    )
    text_df = text_df.sort_values(by="index")
    text_df.to_csv(f"{dataset_dir}/text.csv", index=False)
    
    # Copy the audio file to the dataset directory
    new_audio_path = f"{dataset_dir}/rec_{index}.wav"
    shutil.copy2(audio_path, new_audio_path)
    
    return f"""✅ Updated {dataset_dir}/text.csv \n✅ Saved audio to {new_audio_path}"""

def load_model(model_id: str, language: str) -> Tuple[Pipeline, str]:
    """
    Load the selected speech recognition model.
    """
    if model_id and language:
        yield None, f"Loading {model_id}..."
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            generate_kwargs={"language": language},
        )
        yield pipe, f"✅ Model {model_id} has been loaded."
    else:
        yield None, "⚠️ Please select a model and a language from the dropdown"

def transcribe(pipe: Pipeline, audio: str) -> str:
    """
    Transcribe audio using the loaded model.
    """
    if pipe is None:
        return "⚠️ Please load a model first"
    if audio is None:
        return "⚠️ Please provide an audio input"
    
    text = pipe(audio)["text"]
    return text

def setup_gradio_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            """ # 🎤 Speech Processing Tool
            This tool combines dataset recording and speech transcription capabilities.
            """
        )

        with gr.Tabs():
            # Dataset Recording Tab
            with gr.Tab("Dataset Recorder"):
                gr.Markdown("## Dataset Recording")
                local_sentence_textbox = gr.Text(label="Write your text here")
                local_audio_input = gr.Audio(
                    sources=["microphone"], 
                    label="Record your voice"
                )
                gr.Markdown("_Note: Make sure the recording is not longer than 30 seconds._")
                local_save_button = gr.Button("Save text-recording pair to file")
                local_save_result = gr.Markdown()
                custom_dir_gr = gr.Text(custom_dir, visible=False)
                
                local_save_button.click(
                    fn=save_text_audio_to_file,
                    inputs=[local_audio_input, local_sentence_textbox, custom_dir_gr],
                    outputs=[local_save_result, local_audio_input],
                )

            # Transcription Tab
            with gr.Tab("Speech Transcription"):
                gr.Markdown("## Speech-to-Text Transcription")
                with gr.Row():
                    dropdown_model = gr.Dropdown(
                        choices=model_ids, 
                        value=None, 
                        label="Select a model"
                    )
                    selected_lang = gr.Dropdown(
                        choices=list(languages), 
                        value=None, 
                        label="Select a language"
                    )
                
                load_model_button = gr.Button("Load model")
                model_loaded = gr.Markdown()
                
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record a message or upload audio (WAV/MP3)"
                )
                transcribe_button = gr.Button("Transcribe")
                transcribe_output = gr.Text(label="Output")
                
                # Add dataset creation for transcribed files
                gr.Markdown("### Save to Dataset")
                gr.Markdown("_Review the transcription and click Save to add to the dataset_")
                save_transcribed_button = gr.Button("Save to Dataset")
                save_transcribed_result = gr.Markdown()
                
                model = gr.State()
                load_model_button.click(
                    fn=load_model,
                    inputs=[dropdown_model, selected_lang],
                    outputs=[model, model_loaded],
                )
                transcribe_button.click(
                    fn=transcribe, 
                    inputs=[model, audio_input], 
                    outputs=transcribe_output
                )
                save_transcribed_button.click(
                    fn=save_transcribed_pair,
                    inputs=[audio_input, transcribe_output],
                    outputs=save_transcribed_result
                )

    demo.launch()

if __name__ == "__main__":
    setup_gradio_demo()
