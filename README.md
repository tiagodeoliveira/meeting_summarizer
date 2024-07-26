# Meeting Summarizer

Meeting Summarizer is a Python application that provides real-time transcription and summarization of audio meetings. It uses Amazon Transcribe for speech-to-text conversion and Claude 3 Sonnet for summarization.

## Features

- Real-time audio capture and transcription
- Speaker identification
- Continuous summarization of the meeting content
- Text-based User Interface (TUI) for easy monitoring

## Prerequisites

- Python 3.7+
- AWS account with access to Amazon Transcribe and Bedrock (for Claude 3 Sonnet)
- PyAudio and its dependencies
- [Optional] BlackHole 2ch (for audio routing)

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

4. [Optional] Install BlackHole:
   - Download BlackHole from [the official GitHub repository](https://github.com/ExistentialAudio/BlackHole)
   - Follow the installation instructions provided in the BlackHole repository

## Usage

2. Run the application with:
   ```
   python app.py
   ```

3. In the TUI that appears, you'll see:
   - Left column: Real-time transcription
   - Right column: Ongoing summary
   - Bottom: Audio input level and logs

4. The application will automatically start capturing audio from BlackHole, transcribing it, and generating summaries.

## Configuration

You can modify the `Config` class in the script to adjust audio settings, AWS region, or the Claude model version. If you're using a different virtual audio driver or want to change the input device, you may need to update the `CHANNELS` and `RATE` in the `Config` class.

## Components

- `TUIApp`: Manages the text-based user interface
- `TranscriptionHandler`: Processes transcription events from Amazon Transcribe
- `TranscriptionManager`: Manages transcriptions and summaries

## Troubleshooting

- If you're not seeing any audio input, make sure BlackHole is properly set up and your meeting software is outputting to BlackHole.
- Check the logs at the bottom of the TUI for any error messages.
- Ensure your AWS credentials are correctly set up in the `.env` file.
