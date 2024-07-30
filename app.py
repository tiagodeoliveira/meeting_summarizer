import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from typing import List

import boto3
import numpy as np
import pyaudio
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream
from dotenv import load_dotenv
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import ProgressBar, Static

load_dotenv()

SUMMARIZE_SYSTEM_PROMPT = """You are an executive assistant, responsible for listening into meetings and discussions and summarize that.
This is an ongoing conversation, not all ideas might be formed, if that is the case, try to summarize the best you can.
Messages will be provided in json format, with the speaker identification, timestamp, and the message itself.
Try to identify the names of each interlocutor, since they will be only identified by the speaker id.
"""


@dataclass
class Config:
    CHUNK: int = 1024 * 2
    FORMAT: int = pyaudio.paInt16
    CHANNELS: int = 1
    RATE: int = 16000
    REGION: str = "us-west-2"
    MODEL_ID: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    SUMMARIZATION_BUFFER_LENGTH: int = 5


config = Config()

bedrock = boto3.client("bedrock-runtime", region_name=config.REGION)
transcribe_client = TranscribeStreamingClient(region=config.REGION)

level_queue: asyncio.Queue = asyncio.Queue()
audio_queue: asyncio.Queue = asyncio.Queue()
transcription_queue: asyncio.Queue = asyncio.Queue()


@dataclass
class Transcription:
    ts: float
    speaker: str
    text: str


class TranscriptionBox(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Static("Transcription", classes="title")

    def add_transcription(self, transcription: Transcription) -> None:
        new_transcription = Static(
            f"{transcription.speaker}: {transcription.text}",
            classes="transcription-message",
        )
        self.mount(new_transcription)
        self.scroll_end(animate=False)


class SummaryBox(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Static("Summary", classes="title")
        yield Static("", id="summary_text")

    def update_summary(self, text: str) -> None:
        self.query_one("#summary_text").update(text)


class LogBox(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Static("Application Logs", classes="log-title")

    def add_log(self, message: str) -> None:
        new_log = Static(message, classes="log-message")
        self.mount(new_log)
        self.scroll_end(animate=False)


class LogMessage(Message):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__()


class TextualHandler(logging.Handler):
    def __init__(self, app: App) -> None:
        super().__init__()
        self.app = app

    def emit(self, record: logging.LogRecord) -> None:
        log_entry = self.format(record)
        self.app.post_message(LogMessage(log_entry))


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

            if len(self.buffer) >= config.SUMMARIZATION_BUFFER_LENGTH:
                await self._flush_buffer()
        except Exception as e:
            logging.error(f"Error in transcript event handling: {e}")

    def _process_items(self, items):
        current_item = None
        for item in items:
            if item.stable and item.speaker:
                if current_item and (
                    current_item.speaker != item.speaker or item.content.startswith(
                        " ")
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
        self.app.add_transcription(item)

    async def _flush_buffer(self):
        await transcription_queue.put(self.buffer.copy())
        self.buffer.clear()


class AudioApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    Horizontal {
        height: 2fr;
        margin-bottom: 1;
    }

    TranscriptionBox, SummaryBox {
        width: 1fr;
        height: 100%;
        border: solid green;
    }

    LogBox {
        height: 1fr;
        border: solid yellow;
    }

    Footer {
        background: $boost;
        color: $text;
        height: 1;
    }

    VolumeGauge {
        content-align: center middle;
        width: 100%;
        height: 40;
    }

    .title, .log-title {
        background: blue;
        color: white;
        padding: 1;
        text-align: center;
        text-style: bold;
    }

    .log-message {
        color: yellow;
    }

    .transcription-message {
        margin: 1 0;
        padding: 1;
        border: solid white;
    }
    """

    def __init__(self):
        super().__init__()
        self.transcriptions: List[Transcription] = []
        self.summary: str = ""

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal():
                yield TranscriptionBox()
                yield SummaryBox()
            yield LogBox()
            yield ProgressBar(total=100, show_eta=False)

    def on_mount(self) -> None:
        self.setup_logging()
        self.transcription_box = self.query_one(TranscriptionBox)
        self.summary_box = self.query_one(SummaryBox)
        self.log_box = self.query_one(LogBox)
        self.volume_level = self.query_one(ProgressBar)

        self.start_background_tasks()

    def setup_logging(self) -> None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = TextualHandler(self)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def start_background_tasks(self) -> None:
        asyncio.create_task(self.capture_audio())
        asyncio.create_task(self.process_audio_chunks())
        asyncio.create_task(self.summarize())
        asyncio.create_task(self.update_volume())

    async def update_volume(self) -> None:
        while True:
            in_data = await level_queue.get()

            if in_data is None:
                break

            audio_data = np.frombuffer(in_data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data**2))
            if not np.isnan(rms):
                self.volume_level.update(progress=rms)

    async def capture_audio(self) -> None:
        logging.info("Starting audio capture")

        def callback(in_data, _frame_count, _time_info, _status):
            try:
                level_queue.put_nowait(in_data)
                audio_queue.put_nowait(in_data)
            except Exception as e:
                logging.error(f"Error in audio queue: {e}")

            return (in_data, pyaudio.paContinue)

        try:
            p = pyaudio.PyAudio()

            stream = p.open(
                format=config.FORMAT,
                channels=config.CHANNELS,
                rate=config.RATE,
                input=True,
                stream_callback=callback,
                frames_per_buffer=config.CHUNK,
            )

            logging.info("Audio stream opened")
            while stream.is_active():
                await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Error in audio capture: {e}")
        finally:
            if "stream" in locals():
                stream.stop_stream()
                stream.close()
            p.terminate()

    def get_transcriptions_json(self) -> str:
        return json.dumps([asdict(t) for t in self.transcriptions])

    def add_transcription(self, transcription: Transcription) -> None:
        self.transcriptions.append(transcription)
        self.transcription_box.add_transcription(transcription)

    async def process_audio_chunks(self) -> None:
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

            handler = TranscriptionHandler(stream.output_stream, self)
            asyncio.create_task(handler.handle_events())

            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                await stream.input_stream.send_audio_event(audio_chunk=chunk)

            await stream.input_stream.end_stream()
            await transcription_queue.put(None)
        except Exception as e:
            logging.error(f"Error in audio processing: {e}")

    async def summarize(self) -> None:
        logging.info("Starting summarizer")

        while True:
            try:
                transcription = await transcription_queue.get()
                if transcription is None:
                    break

                logging.info("Summarizing transcription")
                prompt = self.get_transcriptions_json()

                response = bedrock.converse(
                    modelId=config.MODEL_ID,
                    system=[{"text": SUMMARIZE_SYSTEM_PROMPT}],
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                )
                new_summary = response["output"]["message"]["content"][0]["text"]
                self.summary_box.update_summary(new_summary)
            except Exception as e:
                logging.error(f"Error in summarization: {e}")

    def on_log_message(self, message: LogMessage) -> None:
        self.log_box.add_log(message.message)


if __name__ == "__main__":
    app = AudioApp()
    app.run()