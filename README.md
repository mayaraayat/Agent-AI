# Agent - AI-Powered Multimodal Assistant

## Overview

This project is an AI-powered assistant built using the SmolAgents framework and Hugging Face models. The agent integrates various tools, including text analysis, object detection, named entity recognition (NER), file reading, summarization, image captioning, and web search. The interface is powered by Gradio to provide a user-friendly experience.

## Features

- Text Processing

    Named Entity Recognition (NER)

    Text Summarization

    File Reading (PDF, DOCX, TXT)

- Multimodal Capabilities

    Image Captioning

    Object Detection

    Image Generation

- Utilities

    Timezone Conversion

    Web Search (via DuckDuckGo)

    AI Code Agent Execution

- Interactive Chat Interface using Gradio

## Installation

### Prerequisites

Ensure you have Python 3.11+ installed.

### Setup
Clone the repository and install dependencies:

```bash
# Clone repository
git clone https://github.com/mayaraayat/Agent-AI.git
cd Agent-AI

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows, use: myenv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### Running the Agent
Run the agent using the following command:
    
```bash
python app.py
```

This will start a Gradio interface where users can interact with the agent through text and file uploads.

## Acknowledgements

This project was developed based on the Hugging Face Agents Course tutorial: [https://huggingface.co/learn/agents-course/unit1/tutorial](tutorial).

## Future Enhancements

- Multimodal Interaction (Better handling of multiple input types simultaneously)

- Speech-to-Text & Text-to-Speech capabilities

- Fine-tuned AI Models for Improved Accuracy