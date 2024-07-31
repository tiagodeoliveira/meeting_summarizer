# Meeting Summarizer

Meeting Summarizer is a Python application that provides real-time transcription, summarization, and interactive chat functionality for audio meetings. It uses Amazon Transcribe for speech-to-text conversion and Claude 3 Sonnet for summarization and chat interactions.

[![asciicast](https://asciinema.org/a/670292.svg)](https://asciinema.org/a/670292)

## Features

- Real-time audio capture and transcription
- Speaker identification
- Continuous summarization of the meeting content
- Interactive chat interface to ask questions about the meeting
- Text-based User Interface (TUI) for easy monitoring
- Volume level visualization

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
   - Left column: Real-time transcription and application logs
   - Right column: Ongoing summary and chat interface
   - Bottom: Audio input level

3. The application will automatically start capturing audio, transcribing it, and generating summaries.

4. To interact with the chat:
   - Type your question in the chat input box at the bottom of the right column
   - Press Enter to submit your question
   - The AI will respond based on the meeting's transcription and context

5. Additional commands:
   - Press 'q' to quit the application
   - Press 's' to save the transcription and summary to a file named 'meeting.md'

## Configuration

You can modify the `Config` class in the script to adjust audio settings, AWS region, or the Claude model version. The current configuration includes:

- CHUNK: 2048
- FORMAT: pyaudio.paInt16
- CHANNELS: 1
- RATE: 16000
- REGION: "us-west-2"
- MODEL_ID: "anthropic.claude-3-sonnet-20240229-v1:0"
- SUMMARIZATION_BUFFER_LENGTH: 5

## Components

- `AudioApp`: Main application class managing the TUI and background tasks
- `TranscriptionBox`: Displays real-time transcriptions
- `SummaryBox`: Shows the ongoing summary
- `ChatBox`: Provides an interactive chat interface
- `LogBox`: Displays application logs
- `TranscriptionHandler`: Processes transcription events from Amazon Transcribe

## Troubleshooting

- Check the logs in the bottom-left of the TUI for any error messages.
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

## Note

This application uses the Claude 3 Sonnet model for summarization and chat interactions. Ensure you have the necessary permissions and credits to use this model in your AWS account.
