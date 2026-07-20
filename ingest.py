import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_knowledge_base():
    DATA_DIR = "./medical_library"
    DB_DIR = "./chroma_db_storage"
    
    # Safety Check: Ensure the directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created missing '{DATA_DIR}' folder. Please place your text files inside it.")
        return

    print("🔄 Step 1: Reading Text files from the medical library...")
    # This automatically picks up ANY .txt file inside the medical_library folder
    loader = DirectoryLoader(
        DATA_DIR, 
        glob="**/*.txt", 
        loader_cls=TextLoader
    )
    
    try:
        documents = loader.load()
        if not documents:
            print(f"⚠️ No text files found! Make sure your file is saved with a '.txt' extension inside '{DATA_DIR}'.")
            return
        print(f"Successfully loaded {len(documents)} document source(s).")
    except Exception as e:
        print(f"❌ Error reading the text files: {str(e)}")
        return

    print("✂️ Step 2: Splitting tabular data cleanly line-by-line...")
    # Because your data is formatted as a table, we ONLY split at newlines (\n).
    # This groups a few complete rows together without chopping up the symptoms!
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs = text_splitter.split_documents(documents)
    print(f"Generated {len(docs)} unbroken data chunks for the AI to read.")

    print("🧬 Step 3: Activating local vector mathematical model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"💾 Step 4: Storing data vectors permanently inside '{DB_DIR}'...")
    # This builds/updates your local AI memory database
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
    
    print("✅ Scale-up Success! Your text data is fully indexed and ready for the RAG system.")

if __name__ == "__main__":
    build_knowledge_base()