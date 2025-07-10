# AI-INTERVIEWER

## Project Description

AI-INTERVIEWER is an interactive AI-powered tool that simulates a job interview process. It analyzes a candidate's resume and a job description, identifies key skills and overlaps, and conducts a structured, voice-enabled interview using advanced language models and speech recognition. The app provides both a web interface (via Streamlit) and a command-line/voice interface.

## Features
- Upload and parse résumé (PDF/TXT) and job description (PDF/TXT/DOCX)
- Extract and summarize key skills from both documents
- Rank and match top skills between candidate and job
- Conduct a live, voice-based interview with AI-generated questions
- Transcribe and analyze candidate responses
- Download summaries and skill-match results
- Modern, interactive UI with Streamlit

## Installation

### Prerequisites
- Python 3.8+
- [ffmpeg](https://ffmpeg.org/) (required for Whisper speech recognition)

### Python Dependencies
Install all required packages with pip:

```bash
pip install streamlit openai openai-whisper elevenlabs pdfminer.six spacy wordfreq python-docx sounddevice scipy numpy pydantic python-dotenv
```

- You may also need to install the spaCy English model:
  ```bash
  python -m spacy download en_core_web_sm
  ```
- For Whisper, ensure ffmpeg is installed and available in your PATH.
- For ElevenLabs TTS, you need an ElevenLabs API key.

### Environment Variables
Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
ELEVEN_API_KEY=your_elevenlabs_api_key
```

## Usage

### Web App (Streamlit)
Run the Streamlit app:
```bash
streamlit run app.py
```
- Upload your résumé and job description files as prompted.
- Use the sidebar to start/stop a live voice interview.
- View parsed skills, summaries, and skill overlap.
- Download results as JSON.

### Command-Line/Voice Interview
Run the voice-interview agent directly:
```bash
python agent_avatar.py
```
- Follow the prompts to record and transcribe your answers.

## Contribution
Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License
This project is licensed under the MIT License. See the LICENSE file for details.