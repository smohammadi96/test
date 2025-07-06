import os
from datetime import datetime
import json
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from farsi_text_splitter import FarsiTextSplitter
from metadata import documents_config

# Enhanced QA Template with metadata context
QA_TEMPLATE = """شما یک دستیار هوشمند فارسی زبان هستید. لطفا به سوال کاربر با استفاده از اطلاعات داده شده پاسخ دهید.
پاسخ باید:
1. به زبان فارسی باشد
2. دقیق و مرتبط با سوال باشد
3. فقط بر اساس اطلاعات داده شده باشد
4. اگر پاسخ در متن نیست، صادقانه بگویید که نمی‌دانید
5. در صورت امکان، منبع و سال مربوط به اطلاعات را ذکر کنید

متن و منابع: {context}
سوال: {question}

پاسخ فارسی:"""

class Logger:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = "logs"
        self.ensure_log_dir()
    
    def ensure_log_dir(self):
        """Ensure log directory exists"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def log(self, message: str):
        """Log a message to console and file"""
        print(message)
        with open(os.path.join(self.log_dir, "log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    def save_qa(self, question: str, response: dict):
        """Save Q&A results to a dedicated file with metadata"""
        with open(os.path.join(self.log_dir, "qa_results_qwen.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"Question: {question}\n")
            if "error" in response:
                f.write(f"Error: {response['error']}\n")
            else:
                f.write(f"Answer: {response['answer']}\n")
                if response.get('source_documents'):
                    f.write("Sources with Metadata:\n")
                    for idx, doc in enumerate(response['source_documents'], 1):
                        if hasattr(doc, 'page_content'):
                            f.write(f"- Source {idx}:\n")
                            f.write(f"  Content: {doc.page_content[:200]}...\n")
                            if hasattr(doc, 'metadata'):
                                f.write(f"  Metadata: {doc.metadata}\n")
            f.write("="*50 + "\n")

class LocalFarsiRAG:
    def __init__(self, logger):
        self.logger = logger
        self.persist_dir = "chroma_db"
        
        # Initialize Ollama
        self.llm = Ollama(
            model="mshojaei77/gemma3persian", #""Qwen3:4b",
            temperature=0.3
        )
        
        # Use multilingual embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-4B",
            model_kwargs={'device': 'cpu'}
        )
        
        # Enhanced prompt template with metadata awareness
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=QA_TEMPLATE
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            input_key="question"
        )
        
        self.vectorstore = None
    
    def load_document_with_metadata(self, file_path: str, metadata: dict = None):
        """Load a document with custom metadata"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Default metadata
            default_metadata = {
                'source_file': file_path,
                'file_name': os.path.basename(file_path),
                'load_date': datetime.now().strftime('%Y-%m-%d'),
                'file_size': os.path.getsize(file_path)
            }
            
            # Merge with custom metadata
            if metadata:
                default_metadata.update(metadata)
            
            # Read and validate content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip():
                raise ValueError("File is empty")
            
            self.logger.log(f"Loading {file_path} with metadata: {default_metadata}")
            
            # Load document
            loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
            documents = loader.load()
            
            # Add metadata to all document chunks
            for doc in documents:
                doc.metadata.update(default_metadata)
            
            # Split documents
            text_splitter = FarsiTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                keep_separator=True
            )
            
            texts = text_splitter.split_documents(documents)
            self.logger.log(f"Document split into {len(texts)} chunks")
            
            # Create or update vector store
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=texts,
                    embedding=self.embeddings,
                    persist_directory=self.persist_dir
                )
            else:
                self.vectorstore.add_documents(texts)
            
            self.vectorstore.persist()
            self.logger.log(f"Successfully loaded document: {file_path}")
            
        except Exception as e:
            self.logger.log(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def load_documents_with_metadata_config(self, documents_config: list):
        """Load multiple documents with their respective metadata
        
        Args:
            documents_config: List of dicts with 'file_path' and 'metadata' keys
            Example:
            [
                {
                    'file_path': 'doc1.txt',
                    'metadata': {'year': 2023, 'category': 'regulations', 'author': 'Ministry'}
                },
                {
                    'file_path': 'doc2.txt', 
                    'metadata': {'year': 2024, 'category': 'guidelines', 'department': 'Education'}
                }
            ]
        """
        try:
            for doc_config in documents_config:
                file_path = doc_config.get('file_path')
                metadata = doc_config.get('metadata', {})
                
                if not file_path:
                    self.logger.log("Warning: Missing file_path in document config")
                    continue
                
                if os.path.exists(file_path):
                    self.load_document_with_metadata(file_path, metadata)
                else:
                    self.logger.log(f"Warning: File not found: {file_path}")
            
            self.logger.log("All configured documents loaded successfully!")
            
        except Exception as e:
            self.logger.log(f"Error loading documents with metadata config: {str(e)}")
            raise
    
    def search_by_metadata(self, query: str, metadata_filter: dict = None, k: int = 5):
        """Search documents with metadata filtering"""
        try:
            if not self.vectorstore:
                return {"error": "No documents loaded"}
            
            # Create retriever with metadata filtering
            if metadata_filter:
                retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k,
                        "filter": metadata_filter,
                        "fetch_k": k * 2,
                        "lambda_mult": 0.7
                    }
                )
            else:
                retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k,
                        "fetch_k": k * 2,
                        "lambda_mult": 0.7
                    }
                )
            
            # Get relevant documents
            docs = retriever.get_relevant_documents(query)
            
            return {
                "documents": docs,
                "count": len(docs)
            }
            
        except Exception as e:
            self.logger.log(f"Error in metadata search: {str(e)}")
            return {"error": f"Search error: {str(e)}"}
    
    def ask_with_metadata_filter(self, question: str, metadata_filter: dict = None) -> dict:
        """Ask question with optional metadata filtering"""
        try:
            if not self.vectorstore:
                return {"error": "No documents loaded. Please load documents first."}
            
            # Create retriever with optional metadata filtering
            search_kwargs = {
                "k": 8,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
            
            if metadata_filter:
                search_kwargs["filter"] = metadata_filter
                self.logger.log(f"Applying metadata filter: {metadata_filter}")
            
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs=search_kwargs
            )
            
            # Create chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={
                    "prompt": self.prompt_template,
                    "document_separator": "\n\n---\n\n"
                },
                chain_type="stuff",
                return_source_documents=True,
                verbose=True,
                return_generated_question=False
            )
            
            # Get response
            response = chain.invoke({"question": question})
            
            return {
                "answer": response["answer"],
                "source_documents": response.get("source_documents", [])
            }
            
        except Exception as e:
            self.logger.log(f"Error in ask_with_metadata_filter: {str(e)}")
            return {"error": f"Error processing question: {str(e)}"}
    
    # Keep the original ask method for backward compatibility
    def ask(self, question: str) -> dict:
        """Original ask method without metadata filtering"""
        return self.ask_with_metadata_filter(question)

def main():
    # Create logger
    logger = Logger()
    
    # Initialize the local RAG system
    try:
        rag = LocalFarsiRAG(logger=logger)
    except Exception as e:
        logger.log(f"Failed to initialize RAG system: {str(e)}")
        return
    
    # Configure documents with metadata
    # documents_config = [
    #     {
    #         'file_path': 'merged_new.txt',
    #         'metadata': {
    #             'year': 2024,
    #             'category': 'educational_regulations',
    #             'document_type': 'official_guidelines',
    #             'language': 'persian',
    #             'department': 'education_ministry',
    #             'version': '1.0',
    #             'topic': ['university_regulations', 'academic_policies'],
    #             'classification': 'public'
    #         }
    #     },
    #     # Add more documents with their metadata
    #     # {
    #     #     'file_path': 'document2.txt',
    #     #     'metadata': {
    #     #         'year': 2023,
    #     #         'category': 'technical_specifications',
    #     #         'document_type': 'project_manual',
    #     #         'language': 'persian',
    #     #         'department': 'engineering',
    #     #         'version': '2.1',
    #     #         'project': 'safa_dam',
    #     #         'classification': 'restricted'
    #     #     }
    #     # }
    # ]
    
    try:
        # Load documents with metadata
        rag.load_documents_with_metadata_config(documents_config)
        logger.log("All documents loaded successfully!")
        
        # Test questions with different metadata filters
        test_scenarios = [
            {
                'question': 'حداقل نمره قبولی در هر درس چند است؟',
                'metadata_filter': {
                'year': 1390,
                'category': 'Phd',
                'document_type': 'official_guidelines',
                'department': 'computer',
                'version': '1.0',
                'topic': ['university_regulations', 'academic_policies']}
            }
            # {
            #     'question': 'مدت دوره نمایندگی مجلس خبرگان چند سال است؟',
            #     'metadata_filter': {'year': 2024}
            # },
            # {
            #     'question': 'شرایط کاندیداهای مجلس خبرگان چیست؟',
            #     'metadata_filter': None  # No filter
            # }
        ]
        
        logger.log("\nTesting questions with metadata filtering:")
        for scenario in test_scenarios:
            question = scenario['question']
            metadata_filter = scenario['metadata_filter']
            
            logger.log(f"\nQuestion: {question}")
            if metadata_filter:
                logger.log(f"Metadata Filter: {metadata_filter}")
            
            response = rag.ask_with_metadata_filter(question, metadata_filter)
            
            if "error" in response:
                logger.log(f"Error: {response['error']}")
            else:
                logger.log("Answer:")
                logger.log(response['answer'])
                if response.get('source_documents'):
                    logger.log("\nSources with Metadata:")
                    for idx, doc in enumerate(response['source_documents'], 1):
                        if hasattr(doc, 'page_content'):
                            logger.log(f"Source {idx}:")
                            logger.log(f"  Content: {doc.page_content[:200]}...")
                            if hasattr(doc, 'metadata'):
                                logger.log(f"  Metadata: {doc.metadata}")
                logger.log("-" * 80)
            
            # Save to file
            logger.save_qa(question, response)
        
        # Test metadata search
        logger.log("\n" + "="*50)
        logger.log("Testing metadata search:")
        
        search_result = rag.search_by_metadata(
            "نمره قبولی", 
            metadata_filter={'category': 'educational_regulations'},
            k=3
        )
        
        if 'error' not in search_result:
            logger.log(f"Found {search_result['count']} documents matching metadata filter")
            for idx, doc in enumerate(search_result['documents'], 1):
                logger.log(f"Document {idx}: {doc.metadata}")
        
    except Exception as e:
        logger.log(f"Error: {str(e)}")

if __name__ == "__main__":
    main()