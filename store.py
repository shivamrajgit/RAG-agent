import os
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    UnstructuredEmailLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

class DocumentVectorStore:
    """
    A class to handle document loading, processing, and in-memory vector store management.
    """
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = None 
        
        self.loader_mapping = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.doc': Docx2txtLoader,
            '.eml': UnstructuredEmailLoader,
            '.msg': UnstructuredEmailLoader
        }
        
        print("Initialized in-memory DocumentVectorStore")
    
    def _get_file_loader(self, file_path: str):
        """
        Get appropriate loader for the file based on its extension.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension in self.loader_mapping:
            return self.loader_mapping[file_extension]
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def load_single_document(self, file_path: str):
        """
        Load a single document from the given file path.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            loader_class = self._get_file_loader(file_path)
            loader = loader_class(file_path)
            documents = loader.load()
            
            for doc in documents:
                doc.metadata.update({
                    "source_file": os.path.basename(file_path),
                    "file_type": os.path.splitext(file_path)[1].lower(),
                    "full_path": file_path
                })
            
            return documents
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return []
    
    def split_documents(self, documents):
        """
        Split documents into smaller chunks for better retrieval.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        
        return splits
    
    def store_file(self, file_path: str, filename : str):
        """
        Store a single file into the in-memory vectorstore. 
        If vectorstore doesn't exist, create one. If it exists, append the new document to it.
        """
        print(f"Processing file: {filename}")
        
        documents = self.load_single_document(file_path)
        if not documents:
            print("No documents loaded!")
            return False
        
        splits = self.split_documents(documents)
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        else:
            self.vectorstore.add_documents(splits)
        
        return True
    
    
    def get_vectorstore(self):
        """
        Return the current in-memory vectorstore instance.
        """
        return self.vectorstore
    
    def search_documents(self, query: str, k: int = 5):
        """
        Search for documents similar to the query in the in-memory vectorstore.
        """
        if self.vectorstore is None:
            return "No vectorstore available. Please store some documents first."
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            if docs:
                return "\n\n".join([
                    f"Document {i+1} (Source: {doc.metadata.get('source_file', 'Unknown')}):\n{doc.page_content}" 
                    for i, doc in enumerate(docs)
                ])
            else:
                return "No relevant documents found."
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    def clear_vectorstore(self):
        """
        Clear the in-memory vectorstore.
        """
        self.vectorstore = None
        print("Cleared in-memory vectorstore")
