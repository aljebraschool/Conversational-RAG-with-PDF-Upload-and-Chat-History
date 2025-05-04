# Conversational RAG with PDF Upload and Chat History

This project is a **Conversational Retrieval-Augmented Generation (RAG) chatbot** built with [Streamlit](https://streamlit.io/). It allows you to upload PDF files and chat with their content using a powerful LLM (Groq's Gemma2-9b-it) and HuggingFace embeddings. The app maintains chat history for context-aware conversations.

## Features

- **PDF Upload:** Upload one or more PDF files to use as the knowledge base.
- **Conversational RAG:** Ask questions about your PDFs and get concise, context-aware answers.
- **Chat History:** Maintains session-based chat history for more natural, contextual conversations.
- **Groq LLM Integration:** Uses Groq's Gemma2-9b-it model for high-quality responses.
- **HuggingFace Embeddings:** Uses `all-MiniLM-L6-v2` for semantic search over your documents.

## Demo

![image](https://github.com/user-attachments/assets/681c5b0a-df25-4ff9-a6a0-c9c8d66a1a58)



## Getting Started

### Prerequisites

- Python 3.9+
- [Streamlit](https://streamlit.io/)
- [Groq API Key](https://console.groq.com/)
- [HuggingFace API Key](https://huggingface.co/settings/tokens)
- The following Python packages (see below)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   - Create a `.env` file in the project root with your HuggingFace token:

     ```
     HF_TOKEN=your_huggingface_token_here
     ```

### Running the App

```bash
streamlit run app.py
```

### Usage

1. **Enter your Groq API key** in the input box.
2. **(Optional) Enter a session ID** to keep your chat history separate.
3. **Upload one or more PDF files** using the file uploader.
4. **Ask questions** about the content of your PDFs in the chat box.
5. The assistant will answer using only the information from your uploaded PDFs, maintaining context from your previous questions.

## File Structure

```
.
├── app.py
├── requirements.txt
├── .env
└── README.md
```

## Requirements

Example `requirements.txt`:

```
streamlit
langchain
langchain-community
langchain-chroma
langchain-groq
langchain-huggingface
langchain-text-splitters
python-dotenv
```

## Notes

- Your data and chat history are stored in memory and are not persisted after the app stops.
- Make sure your API keys are kept secure and **never commit your `.env` file** to public repositories.

## License

MIT License

---

*Built with ❤️ using Streamlit and LangChain.*
```

