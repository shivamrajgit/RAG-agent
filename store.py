import os
import pickle
import hashlib
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
    A class to handle document loading, processing, and persistent vector store management.
    """
    
    def __init__(self, persist_directory="./vectorstore_cache"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = None 
        self.persist_directory = persist_directory
        self.vectorstore_path = os.path.join(persist_directory, "faiss_index")
        self.metadata_path = os.path.join(persist_directory, "metadata.pkl")
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        self.loader_mapping = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.doc': Docx2txtLoader,
            '.eml': UnstructuredEmailLoader,
            '.msg': UnstructuredEmailLoader
        }
        
        print("Initialized DocumentVectorStore with persistence")
        
        # Try to load existing vectorstore
        self.load_vectorstore()
    
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
    
    def get_file_hash(self, file_path: str) -> str:
        """
        Calculate MD5 hash of a file to detect changes.
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_files_metadata(self, directory_path: str) -> dict:
        """
        Get metadata (hash and modification time) for all files in directory.
        """
        files_metadata = {}
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                try:
                    file_hash = self.get_file_hash(file_path)
                    mod_time = os.path.getmtime(file_path)
                    files_metadata[filename] = {
                        'hash': file_hash,
                        'mod_time': mod_time,
                        'path': file_path
                    }
                except Exception as e:
                    print(f"Error getting metadata for {filename}: {e}")
        return files_metadata
    
    def save_vectorstore(self):
        """
        Save the vectorstore and metadata to disk.
        """
        if self.vectorstore is not None:
            try:
                self.vectorstore.save_local(self.vectorstore_path)
                print(f"Vectorstore saved to {self.vectorstore_path}")
                return True
            except Exception as e:
                print(f"Error saving vectorstore: {e}")
                return False
        return False
    
    def save_metadata(self, files_metadata: dict):
        """
        Save files metadata to disk.
        """
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(files_metadata, f)
            print(f"Metadata saved to {self.metadata_path}")
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def load_metadata(self) -> dict:
        """
        Load files metadata from disk.
        """
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
        return {}
    
    def load_vectorstore(self):
        """
        Load the vectorstore from disk if it exists.
        """
        try:
            if os.path.exists(self.vectorstore_path):
                self.vectorstore = FAISS.load_local(
                    self.vectorstore_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("Loaded existing vectorstore from disk")
                return True
            else:
                print("No existing vectorstore found")
                return False
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            return False
    
    def files_changed(self, directory_path: str) -> tuple[bool, list, dict]:
        """
        Check if files in directory have changed since last processing.
        Returns: (has_changes, changed_files, current_metadata)
        """
        current_metadata = self.get_files_metadata(directory_path)
        saved_metadata = self.load_metadata()
        
        changed_files = []
        
        # Check for new or modified files
        for filename, current_meta in current_metadata.items():
            if filename not in saved_metadata:
                changed_files.append(filename)
                print(f"New file detected: {filename}")
            elif current_meta['hash'] != saved_metadata[filename]['hash']:
                changed_files.append(filename)
                print(f"Modified file detected: {filename}")
        
        # Check for deleted files
        for filename in saved_metadata:
            if filename not in current_metadata:
                print(f"Deleted file detected: {filename}")
                # For deleted files, we'll need to rebuild the entire vectorstore
                # as FAISS doesn't support selective deletion easily
                return True, list(current_metadata.keys()), current_metadata
        
        has_changes = len(changed_files) > 0
        return has_changes, changed_files, current_metadata
    
    def store_file(self, file_path: str, filename: str):
        """
        Store a single file into the vectorstore. 
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
    
    def process_directory(self, directory_path: str, force_rebuild: bool = False):
        """
        Process all files in a directory with intelligent caching.
        Only processes changed files unless force_rebuild is True.
        """
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return False
        
        if force_rebuild:
            print("Force rebuild requested - processing all files")
            self.vectorstore = None
            has_changes = True
            files_to_process = [f for f in os.listdir(directory_path) 
                              if os.path.isfile(os.path.join(directory_path, f))]
            current_metadata = self.get_files_metadata(directory_path)
        else:
            has_changes, files_to_process, current_metadata = self.files_changed(directory_path)
        
        if not has_changes and self.vectorstore is not None:
            print("No changes detected and vectorstore exists - skipping processing")
            return True
        
        if has_changes and not force_rebuild:
            print(f"Changes detected in {len(files_to_process)} files")
        
        # Process files
        processed_count = 0
        for filename in files_to_process:
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                try:
                    if self.store_file(file_path, filename):
                        processed_count += 1
                        print(f"Successfully processed: {filename}")
                    else:
                        print(f"Failed to process: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        if processed_count > 0:
            # Save vectorstore and metadata
            self.save_vectorstore()
            self.save_metadata(current_metadata)
            print(f"Processed {processed_count} files and saved to cache")
        
        return processed_count > 0
    
    
    def get_vectorstore(self):
        """
        Return the current vectorstore instance.
        """
        return self.vectorstore
    
    def search_documents(self, query: str, k: int = 5):
        """
        Search for documents similar to the query in the vectorstore.
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
        Clear the vectorstore and remove cached files.
        """
        self.vectorstore = None
        
        # Remove cached files
        try:
            if os.path.exists(self.vectorstore_path):
                import shutil
                shutil.rmtree(self.vectorstore_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            print("Cleared vectorstore and cache")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def get_cache_info(self):
        """
        Get information about the current cache state.
        """
        info = {
            'vectorstore_exists': self.vectorstore is not None,
            'cache_directory': self.persist_directory,
            'vectorstore_cached': os.path.exists(self.vectorstore_path),
            'metadata_cached': os.path.exists(self.metadata_path)
        }
        
        if info['metadata_cached']:
            metadata = self.load_metadata()
            info['cached_files'] = list(metadata.keys())
            info['cached_files_count'] = len(metadata)
        else:
            info['cached_files'] = []
            info['cached_files_count'] = 0
            
        return info
