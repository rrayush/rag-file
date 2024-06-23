from langchain_community.document_loaders import TextLoader

from langchain_community.llms import CTransformers

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS

llm = CTransformers(model="models\llama-2-7b-chat.ggmlv3.q2_K.bin",
                    model_type="llama",
                    max_new_tokens=512,
                    temperature=0.1)
                    


loader = DirectoryLoader('Data', glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()   

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(docs)            

#print(len(texts))

#download sentence transformer embeddings from hugging face

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

library = FAISS.from_documents(texts, embeddings)

query = []

Query_answer = library.similarity_search(query)

