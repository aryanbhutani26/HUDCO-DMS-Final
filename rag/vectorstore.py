# rag/vectorstore.py
from typing import List, Dict, Any
import os, logging
import chromadb
from chromadb.config import Settings

# Global kill switches (belt-and-suspenders)
os.environ["CHROMA_TELEMETRY__ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["PH_DISABLED"] = "true"
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

def _kill_chroma_telemetry():
    """Monkey-patch Chroma telemetry to a no-op across 0.5.x shapes."""
    try:
        from chromadb.telemetry import telemetry as _t
        def _noop(*args, **kwargs):
            return None
        for attr in ("capture", "flush", "identify", "alias", "group"):
            if hasattr(_t, attr):
                try:
                    setattr(_t, attr, _noop)
                except Exception:
                    pass
        if hasattr(_t, "_telemetry"):
            try:
                _t._telemetry = type("Noop", (), {"capture": _noop, "flush": _noop})()
            except Exception:
                pass
    except Exception:
        pass

_kill_chroma_telemetry()

class VectorStore:
    def __init__(self, persist_dir: str):
        _kill_chroma_telemetry()
        
        # Create settings with telemetry disabled
        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
        
        # Initialize client
        try:
            self.client = chromadb.PersistentClient(path=persist_dir, settings=settings)
        except Exception as e:
            print(f"Failed to create PersistentClient: {e}")
            # Fallback to in-memory client for testing
            self.client = chromadb.Client(settings)
        
        _kill_chroma_telemetry()
        
        # Create or get collection with simple name
        collection_name = "hudco_docs"
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "HUDCO document embeddings"}
            )
        except Exception as e:
            print(f"Error creating collection: {e}")
            # Try to delete and recreate if there's a schema issue
            try:
                self.client.delete_collection(collection_name)
            except:
                pass
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "HUDCO document embeddings"}
            )

    def add(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], documents: List[str]):
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self, query_embedding: List[float], top_k: int = 4):
        _kill_chroma_telemetry()
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

    def reset(self):
        collection_name = "hudco_docs"
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": "HUDCO document embeddings"}
        )
    
    def get_all(self):
        """Return all docs & metadatas currently in the collection."""
        try:
            return self.collection.get(include=["documents", "metadatas"])
        except TypeError:
            return self.collection.get()
    
    
