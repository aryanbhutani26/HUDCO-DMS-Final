"""
Document management system for uploaded files
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class DocumentManager:
    def __init__(self, db_path="documents.db", upload_dir="uploaded_docs"):
        self.db_path = db_path
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Initialize documents database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                uploaded_by TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                is_indexed BOOLEAN DEFAULT FALSE,
                chunk_count INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_document(self, filename: str, original_name: str, file_path: str, 
                    file_type: str, file_size: int, uploaded_by: str, 
                    description: str = "") -> int:
        """Add a new document record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO documents 
            (filename, original_name, file_path, file_type, file_size, uploaded_by, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (filename, original_name, file_path, file_type, file_size, uploaded_by, description))
        
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return doc_id 
   
    def get_all_documents(self) -> List[Dict]:
        """Get all documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, original_name, file_type, file_size, 
                   uploaded_by, upload_date, description, is_indexed, chunk_count
            FROM documents 
            ORDER BY upload_date DESC
        ''')
        
        columns = [desc[0] for desc in cursor.description]
        documents = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return documents
    
    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        """Get a specific document by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, original_name, file_path, file_type, file_size,
                   uploaded_by, upload_date, description, is_indexed, chunk_count
            FROM documents 
            WHERE id = ?
        ''', (doc_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def update_index_status(self, doc_id: int, is_indexed: bool, chunk_count: int = 0):
        """Update document indexing status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE documents 
            SET is_indexed = ?, chunk_count = ?
            WHERE id = ?
        ''', (is_indexed, chunk_count, doc_id))
        
        conn.commit()
        conn.close()
    
    def delete_document(self, doc_id: int) -> bool:
        """Delete a document record and file"""
        doc = self.get_document_by_id(doc_id)
        if not doc:
            return False
        
        # Delete file
        file_path = Path(doc['file_path'])
        if file_path.exists():
            file_path.unlink()
        
        # Delete record
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()
        
        return True
    
    def get_indexed_documents(self) -> List[Dict]:
        """Get only indexed documents for user selection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, original_name, description, chunk_count, upload_date
            FROM documents 
            WHERE is_indexed = TRUE
            ORDER BY upload_date DESC
        ''')
        
        columns = [desc[0] for desc in cursor.description]
        documents = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return documents