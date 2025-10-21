# HUDCO Document Management System

A comprehensive RAG (Retrieval-Augmented Generation) system with role-based access control for document management and intelligent Q&A.

## 🚀 Key Features
- **Role-Based Access Control**: Admin and User roles with different permissions
- **Document Management**: Upload, index, and manage PDF/TXT/MD files
- **Intelligent Q&A**: Chat with documents using Gemini AI
- **Vector Search**: Semantic search using Chroma DB and embeddings
- **User-Friendly Interface**: Clean Streamlit web interface
- **Document Tracking**: Track upload status, indexing, and metadata

## 🏗️ System Architecture

### User Roles
- **Admin**: Upload documents, manage document library, chat with documents
- **User**: View available documents, chat with indexed documents

### Core Components
- **Authentication System**: Simple login with SQLite user management
- **Document Manager**: File upload, storage, and indexing tracking
- **RAG Pipeline**: Document chunking, embedding, and retrieval
- **AI Integration**: Gemini AI for intelligent responses

## 📁 Project Structure
```
HUDCO-Document-Management/
├─ app.py                      # Main Streamlit application
├─ auth.py                     # Authentication system
├─ document_manager.py         # Document management logic
├─ demo_setup.py              # Demo data setup script
├─ requirements.txt           # Python dependencies
├─ config.yaml               # System configuration
├─ rag/                      # RAG implementation
│  ├─ ingest.py             # Document processing
│  ├─ embeddings.py         # Embedding generation
│  ├─ vectorstore.py        # Vector database interface
│  ├─ retriever.py          # Semantic search
│  └─ generator.py          # AI response generation
├─ data/sample/content/      # Sample documents
├─ uploaded_docs/           # User uploaded files
└─ .chroma/                # Vector database storage
```

## 🛠️ Quick Start

### 1. Environment Setup
```bash
# Clone and navigate to project
cd HUDCO-Document-Management

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Initialize Demo Data
```bash
python demo_setup.py
```

### 4. Run the Application
```bash
streamlit run app.py
```

### 5. Login Credentials
- **Admin**: `admin` / `admin123`
- **Demo User**: `demo_user` / `password123`

## 💼 Usage Workflows

### Admin Workflow
1. **Login** as admin
2. **Upload Documents**: Use "Upload Documents" tab to add PDF/TXT/MD files
3. **Manage Library**: View, re-index, or delete documents in "Manage Documents"
4. **Chat**: Ask questions about any uploaded document

### User Workflow
1. **Login** as regular user
2. **Browse Documents**: View available indexed documents
3. **Select & Chat**: Choose specific documents to chat with
4. **Get Answers**: Ask questions and receive AI-powered responses

## 🔧 Configuration

### System Settings (`config.yaml`)
```yaml
defaults:
  top_k: 4                    # Number of chunks to retrieve
  chunk_size: 800            # Document chunk size
  chunk_overlap: 120         # Overlap between chunks
  vector_persist_dir: ".chroma"  # Vector database location
```

### Supported File Types
- **PDF**: Automatically extracted using PyPDF2
- **TXT**: Plain text files
- **MD**: Markdown files

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
1. Set up environment variables
2. Configure proper database paths
3. Use production WSGI server
4. Set up proper authentication

## 🔐 Security Features
- Password hashing using SHA-256
- Role-based access control
- File upload validation
- Secure file storage
