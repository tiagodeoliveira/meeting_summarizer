import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from typing import List

import boto3
import numpy as np
import py_cui
import pyaudio
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream
from dotenv import load_dotenv

SUMMARIZE_SYSTEM_PROMPT = """You are an executive assistant, responsible for listening into meetings and discussions and summarize that.
This is an ongoing conversation, not all ideas might be formed, if that is the case, try to summarize the best you can.
Messages will be provided in json format, with the speaker identification, timestamp, and the message itself.
Try to identify the names of each interlocutor, since they will be only identified by the speaker id.
"""


load_dotenv()


class CUILogHandler(logging.Handler):
    def __init__(self, cui_app):
        super().__init__()
        self.cui_app = cui_app

    def emit(self, record):
        log_entry = self.format(record)
        self.cui_app.add_log(log_entry)


class TUIApp:
    def __init__(self, master):
        self.master = master

        self.left_column = self.master.add_scroll_menu(
            "Transcription", 0, 0, row_span=6, column_span=1
        )
        self.right_column = self.master.add_scroll_menu(
            "Summary", 0, 1, row_span=6, column_span=1
        )

        self.log_box = self.master.add_text_block(
            "Logs", 5, 0, row_span=2, column_span=2
        )
        self.log_box.set_selectable(False)
        self.master.set_status_bar_text("RMS: 0")

    def add_log(self, message):
        current_logs = self.log_box.get()
        new_logs = current_logs + message + "\n"
        self.log_box.set_text(new_logs)

    def update_level(self, level):
        self.master.set_status_bar_text(level)

    def update_screen(self):
        logging.info("Updating screen")
        self.left_column.clear()
        self.left_column.add_item_list(
            [f"{t.speaker}: {t.text}" for t in transcription_manager.transcriptions]
        )
        self.right_column.clear()
        self.right_column.add_item_list([transcription_manager.summary])


@dataclass
class Config:
    CHUNK: int = 1024 * 2
    FORMAT: int = pyaudio.paInt16
    CHANNELS: int = 1
    RATE: int = 16000
    REGION: str = "us-west-2"
    MODEL_ID: str = "anthropic.claude-3-sonnet-20240229-v1:0"


config = Config()

bedrock = boto3.client("bedrock-runtime", region_name=config.REGION)
transcribe_client = TranscribeStreamingClient(region=config.REGION)

audio_queue: asyncio.Queue = asyncio.Queue()
transcription_queue: asyncio.Queue = asyncio.Queue()


@dataclass
class Transcription:
    ts: float
    speaker: str
    text: str


class TranscriptionManager:
    def __init__(self):
        self.transcriptions: List[Transcription] = []
        self.summary: str = ""

    def add_transcription(self, transcription: Transcription):
        self.transcriptions.append(transcription)

    def update_summary(self, new_summary: str):
        self.summary = new_summary

    def get_transcriptions_json(self) -> str:
        return json.dumps([asdict(t) for t in self.transcriptions])


transcription_manager = TranscriptionManager()


async def capture_audio(app):
    logging.info("Starting audio capture")

    def callback(in_data, _frame_count, _time_info, _status):
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data**2))
        app.update_level(f"RMS: {rms:.2f}")
        audio_queue.put_nowait(in_data)
        return (in_data, pyaudio.paContinue)

    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=config.FORMAT,
            channels=config.CHANNELS,
            rate=config.RATE,
            input=True,
            stream_callback=callback,
            frames_per_buffer=config.CHUNK,
        )

        while stream.is_active():
            await asyncio.sleep(0.1)
    except Exception as e:
        logging.error(f"Error in audio capture: {e}")
    finally:
        if "stream" in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()


class TranscriptionHandler(TranscriptResultStreamHandler):
    def __init__(self, transcript_result_stream: TranscriptResultStream, app):
        super().__init__(transcript_result_stream)
        self.buffer: List[Transcription] = []
        self.app = app

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        try:
            results = transcript_event.transcript.results
            for result in results:
                for alt in result.alternatives:
                    self._process_items(alt.items)

            if len(self.buffer) >= 5:
                await self._flush_buffer()
        except Exception as e:
            logging.error(f"Error in transcript event handling: {e}")

    def _process_items(self, items):
        current_item = None
        for item in items:
            if item.stable and item.speaker:
                if current_item and (
                    current_item.speaker != item.speaker or item.content.startswith(" ")
                ):
                    self._add_to_buffer(current_item)
                    current_item = None

                if current_item is None:
                    current_item = Transcription(
                        ts=item.start_time, speaker=item.speaker, text=""
                    )

                current_item.text += item.content + " "

        if current_item:
            self._add_to_buffer(current_item)

    def _add_to_buffer(self, item: Transcription):
        self.buffer.append(item)
        transcription_manager.add_transcription(item)
        self.app.update_screen()

    async def _flush_buffer(self):
        await transcription_queue.put(self.buffer.copy())
        self.buffer.clear()


async def summarize(app):
    logging.info("Starting summarizer")

    while True:
        try:
            transcription = await transcription_queue.get()
            if transcription is None:
                break

            logging.info("Summarizing transcription")
            prompt = transcription_manager.get_transcriptions_json()

            response = bedrock.converse(
                modelId=config.MODEL_ID,
                system=[{"text": SUMMARIZE_SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
            )
            new_summary = response["output"]["message"]["content"][0]["text"]
            transcription_manager.update_summary(new_summary)
            app.update_screen()
        except Exception as e:
            logging.error(f"Error in summarization: {e}")


async def process_audio_chunks(app):
    logging.info("Starting audio processing")
    try:
        stream = await transcribe_client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=config.RATE,
            media_encoding="pcm",
            show_speaker_label=True,
            enable_partial_results_stabilization=True,
            partial_results_stability="medium",
        )

        async def write_chunks():
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
            await stream.input_stream.end_stream()

        handler = TranscriptionHandler(stream.output_stream, app)
        await asyncio.gather(write_chunks(), handler.handle_events())
        await transcription_queue.put(None)
    except Exception as e:
        logging.error(f"Error in audio processing: {e}")


async def main():
    tasks = []
    try:
        root = py_cui.PyCUI(7, 2, auto_focus_buttons=False)
        root.set_title("Meeting Summarizer")
        root.set_refresh_timeout(0.5)

        app = TUIApp(root)

        cui_handler = CUILogHandler(app)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        cui_handler.setFormatter(formatter)
        cui_handler.setLevel(logging.INFO)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[cui_handler],
        )

        logging.info("Starting main execution")

        tasks = [
            asyncio.create_task(process_audio_chunks(app)),
            asyncio.create_task(summarize(app)),
            asyncio.create_task(capture_audio(app)),
            asyncio.to_thread(root.start),
        ]
        await asyncio.gather(*tasks)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Program interrupted by user. Shutting down.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        pass
