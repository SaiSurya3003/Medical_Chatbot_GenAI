from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY


extracted_data = load_pdf(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

pc.create_index(
    name = index_name,
    dimension = 384,
    metric = "cosine",
    spec = ServerlessSpec(
        cloud = "aws",
        region = "us-east-1"
    )
)


# Embed each chinck and upsert the embeddings into your Pinecone Index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)