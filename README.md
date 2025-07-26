# Multilingual RAG Web App 🤖

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that can answer questions in both Bengali and English. This application processes PDF documents and provides accurate answers based on the document content using Google's Gemini AI model.

## Features

- **Multilingual Support**: Handles questions and answers in both Bengali and English
- **PDF Document Processing**: Loads and processes PDF files for question-answering
- **Intelligent Text Splitting**: Uses recursive character text splitter optimized for Bengali text
- **Vector Search**: Employs FAISS vector store for efficient similarity search
- **Context-Aware Responses**: Shows retrieved context alongside answers for transparency
- **Real-time Processing**: Interactive web interface with real-time document processing

## Prerequisites

- Python 3.8+
- Google Gemini API key
- Required Python packages (see Installation)

## Installation

1. **Clone the repository** (or create a new directory for your project):
```bash
mkdir multilingual-rag-app
cd multilingual-rag-app
```

2. **Install required packages**:
```bash
pip install streamlit python-dotenv langchain-community langchain-text-splitters langchain-google-genai langchain-core sentence-transformers faiss-cpu pypdf
```

3. **Set up environment variables**:
Create a `.env` file in your project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

4. **Prepare your document**:
   - Create a `document` folder in the parent directory (`../document/`)
   - Place your PDF file named `HSC26-Bangla1st-Paper.pdf` in that folder
   - Or modify the `file_path` variable in the code to point to your PDF location

## Project Structure

```
multilingual-rag-app/
├── app.py                          # Main Streamlit application
├── .env                           # Environment variables
├── README.md                      # This file
└── ../document/
    └── HSC26-Bangla1st-Paper.pdf  # Your PDF document
```

## Usage

1. **Start the application**:
```bash
streamlit run app.py
```

2. **Initialize the RAG System**:
   - Click the "Initialize RAG System" button
   - Wait for the system to process your PDF document
   - The system will split the document into chunks and create embeddings

3. **Ask Questions**:
   - Enter your question in Bengali or English
   - Click "Get Answer" to receive a response
   - View the retrieved context in the expandable section

## How It Works

### Document Processing
1. **PDF Loading**: Uses PyPDFLoader to extract text from PDF files
2. **Text Splitting**: Employs RecursiveCharacterTextSplitter with Bengali-specific separators (`।`, `।`)
3. **Embeddings**: Uses multilingual sentence transformer model (`paraphrase-multilingual-MiniLM-L12-v2`)
4. **Vector Storage**: Stores embeddings in FAISS vector database

### Question Answering
1. **Query Processing**: Accepts questions in Bengali or English
2. **Similarity Search**: Retrieves top 5 most relevant document chunks
3. **Context Assembly**: Combines retrieved chunks as context
4. **AI Response**: Uses Google Gemini 2.0 Flash model to generate answers
5. **Transparency**: Shows retrieved context for verification

## Configuration Options

### Text Splitter Settings
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,              # Size of each text chunk
    chunk_overlap=150,           # Overlap between chunks
    separators=["\n\n", "\n", "।", "।", " ", ""]  # Bengali-specific separators
)
```

### Retrieval Settings
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}       # Number of chunks to retrieve
)
```

## Customization

### Change PDF Document
Modify the `file_path` variable:
```python
file_path = "path/to/your/document.pdf"
```

### Adjust Model Settings
Change the embedding model:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="your-preferred-multilingual-model"
)
```

Change the LLM model:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",  # or other available models
    google_api_key=api_key
)
```

## Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY is not set"**
   - Ensure your `.env` file contains the correct API key
   - Verify the `.env` file is in the same directory as your script

2. **"PDF file not found"**
   - Check if the PDF file exists at the specified path
   - Ensure the file path is correct relative to your script location

3. **Memory Issues**
   - Reduce `chunk_size` for large documents
   - Consider using CPU-optimized models if running on limited hardware

4. **Poor Answer Quality**
   - Increase the number of retrieved chunks (`k` parameter)
   - Adjust chunk size and overlap settings
   - Try different embedding models

## API Keys Setup

### Getting Google Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

## Dependencies

- `streamlit`: Web app framework
- `python-dotenv`: Environment variable management
- `langchain-community`: Document loaders and embeddings
- `langchain-text-splitters`: Text processing utilities
- `langchain-google-genai`: Google Gemini integration
- `langchain-core`: Core LangChain functionality
- `sentence-transformers`: Multilingual embeddings
- `faiss-cpu`: Vector similarity search
- `pypdf`: PDF processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request


## Support

For issues and questions:
- Check the troubleshooting section above
- Review the LangChain documentation
- Ensure all dependencies are properly installed
- Verify your API keys are correctly configured

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [LangChain](https://langchain.com/)
- Uses [Google Gemini](https://deepmind.google/technologies/gemini/) AI
- Multilingual embeddings from [Sentence Transformers](https://www.sbert.net/)
