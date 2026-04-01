# ragdemo.ai
Retrieval Augmented Generation(RAG) is a technique that enhances the capabilities of LLMs by combining information retrieval with text generation. Instead of relying on pre-traned knowledge, RAG fetch relevant data from external sources and use it to generate more accurate responses. 


### Pcakges 

streamlit
python-dotenv
PyPDF2
google-generativeai

langchain # core framework
langchain-huggingface # Connect huggingface models to perform embedding
faiss-cpu # Fast vector database to store embedded data
langchain-communitiy # extra integration 
langchain-text-splitters # to split large text into smaller chunks
sentence-transformers # pre-trained embedding models to covert text into vectors 
langchain-core # document, chains etc.. 

##### In simple words

Text -> split text -> convert vector -> store in database -> search similar content -> send to LLM -> Get answers for questions 


'all-MiniLM-L6-v2' -> Simple hugging face embedding model which splits the text and converts the text into vectors



