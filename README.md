# Meeting Summarizer

Meeting Summarizer is a Python application that provides real-time transcription, summarization, and chat functionality for audio meetings. It uses Amazon Transcribe for speech-to-text conversion and Claude 3 Sonnet for summarization and chat interactions.

[![asciicast](https://asciinema.org/a/670310.svg)](https://asciinema.org/a/670310)

## Features

- Real-time audio capture and transcription
- Speaker identification
- Continuous summarization of the meeting content
- Text-based User Interface (TUI) for easy monitoring
- Volume level visualization
- Chat functionality to ask questions about the meeting
- Audio recording and meeting data persistence

## Prerequisites

- Python 3.7+
- AWS account with access to Amazon Transcribe and Bedrock (for Claude 3 Sonnet)
- PyAudio and its dependencies

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/tiagodeoliveira/meeting-summarizer.git
   cd meeting-summarizer
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your AWS credentials and region in a `.env` file:
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-west-2
   ```

## Usage

1. Run the application with:
   ```
   python app.py
   ```

2. In the TUI that appears, you'll see:
   - Top left: Real-time transcription
   - Top right: Ongoing summary
   - Bottom left: Application logs
   - Bottom right: Chat interface
   - Bottom: Audio input level

3. Use the following key bindings:
   - `s`: Start/Stop the meeting recording and processing
   - `q`: Quit the application

4. When the meeting is active, the application will automatically capture audio, transcribe it, generate summaries, and allow for chat interactions.

5. Use the chat interface to ask questions about the ongoing meeting.

## Configuration

You can modify the `Config` class in the script to adjust audio settings, AWS region, or the Claude model version. The current configuration includes:

- CHUNK: 2048
- FORMAT: pyaudio.paInt16
- CHANNELS: 1
- RATE: 16000
- REGION: "us-west-2"
- MODEL_ID: "anthropic.claude-3-sonnet-20240229-v1:0"
- SUMMARIZATION_BUFFER_LENGTH: 5
- PERSISTING_DIR: "meetings"

## Components

- `AudioApp`: Main application class managing the TUI and background tasks
- `TranscriptionBox`: Displays real-time transcriptions
- `SummaryBox`: Shows the ongoing summary
- `LogBox`: Displays application logs
- `ChatBox`: Provides chat functionality for interacting with the meeting content
- `TranscriptionHandler`: Processes transcription events from Amazon Transcribe

## Data Persistence

The application saves meeting data in the `meetings` directory:
- Audio recordings are saved as WAV files
- Meeting transcriptions, summaries, and chat logs are saved as Markdown files

## Troubleshooting

- Check the logs at the bottom-left of the TUI for any error messages.
- Ensure your AWS credentials are correctly set up in the `.env` file.
- If you're having issues with audio input, verify that your microphone is properly configured and accessible to the application.

## Dependencies

- asyncio
- boto3
- numpy
- pyaudio
- amazon-transcribe
- python-dotenv
- textual
- wave

## Note

This application uses the Claude 3 Sonnet model for summarization and chat interactions. Ensure you have the necessary permissions and credits to use this model in your AWS account.