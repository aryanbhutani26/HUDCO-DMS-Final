"""
Simple test script to verify the system components work
"""
from auth import AuthManager
from document_manager import DocumentManager
from rag.vectorstore import VectorStore
from rag.embeddings import Embedder, EmbedConfig

def test_auth():
    """Test authentication system"""
    print("Testing authentication...")
    auth = AuthManager()
    
    # Test admin login
    role = auth.authenticate("admin", "admin123")
    print(f"Admin login: {'✅' if role == 'admin' else '❌'}")
    
    # Test user login
    role = auth.authenticate("demo_user", "password123")
    print(f"User login: {'✅' if role == 'user' else '❌'}")
    
    print("Authentication tests completed.\n")

def test_document_manager():
    """Test document management"""
    print("Testing document manager...")
    doc_manager = DocumentManager()
    
    # Test database initialization
    docs = doc_manager.get_all_documents()
    print(f"Document database initialized: ✅")
    print(f"Current documents: {len(docs)}")
    
    print("Document manager tests completed.\n")

def test_vector_store():
    """Test vector store"""
    print("Testing vector store...")
    try:
        vs = VectorStore(".chroma_test")
        print("Vector store initialized: ✅")
        
        # Test basic operations
        test_docs = ["This is a test document.", "Another test document."]
        test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        test_metas = [{"source": "test1.txt"}, {"source": "test2.txt"}]
        test_ids = ["test1", "test2"]
        
        vs.add(test_ids, test_embeddings, test_metas, test_docs)
        print("Vector store add operation: ✅")
        
        # Test query
        result = vs.query([0.1, 0.2, 0.3], top_k=1)
        print(f"Vector store query: {'✅' if result else '❌'}")
        
        # Cleanup
        vs.reset()
        print("Vector store cleanup: ✅")
        
    except Exception as e:
        print(f"Vector store error: ❌ {e}")
    
    print("Vector store tests completed.\n")

if __name__ == "__main__":
    print("🧪 HUDCO Document Management System - Component Tests\n")
    
    test_auth()
    test_document_manager()
    test_vector_store()
    
    print("🎉 All tests completed!")
    print("\nYou can now run the application with: streamlit run app.py")
    print("Login credentials:")
    print("- Admin: admin / admin123")
    print("- User: demo_user / password123")