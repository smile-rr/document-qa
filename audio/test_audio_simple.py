import whisper
import time
import warnings
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Suppress the FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def load_and_transcribe(model_name: str, audio_file: str) -> str:
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_file)
    return result["text"]

def run_task(task_callable):
    # Start the timer
    start_time = time.time()

    # Show spinner and timer using rich
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Transcribing...", total=None)
        result = task_callable()
        progress.update(task, advance=1)

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Transcription:")
    print(result)
    
    print("")
    print(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    model_name = "turbo"
    audio_file = "/Users/pc-rn/Music/ErasTour/02 - Cruel Summer.flac"
    
    # Define the task as a lambda function
    task = lambda: load_and_transcribe(model_name, audio_file)
    
    # Run the task with performance and progress tracking
    run_task(task)