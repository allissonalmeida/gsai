from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone
import yaml
import os
import zipfile # Importação da biblioteca zipfile
import streamlit as st

# Importações para Google Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# ---
## Configuração e Inicialização
# ---

# Load configuration from config.yaml
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Configura as chaves de API
os.environ['PINECONE_API_KEY'] = config['PINECONE_API_KEY']
os.environ['GOOGLE_API_KEY'] = config['GOOGLE_API_KEY']

# ---
## Diagnóstico de Modelos Gemini
# Este bloco irá listar os modelos que sua GOOGLE_API_KEY pode acessar.
# É VITAL que você me informe o que ele imprime para que possamos ajustar o nome do modelo.
# ---
print("---")
print("Iniciando verificação de modelos Gemini disponíveis para sua chave API...")
try:
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'], client_options={"api_endpoint": "generativelanguage.googleapis.com"})

    found_gemini_pro = False
    found_embedding = False
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"Nome do Modelo de Chat encontrado: {m.name}")
            if "gemini-pro" in m.name: # Ainda verificando por "gemini-pro" para diagnóstico
                found_gemini_pro = True
        if "embedContent" in m.supported_generation_methods:
            print(f"Nome do Modelo de Embedding encontrado: {m.name}")
            if "embedding-001" in m.name:
                found_embedding = True
    if not found_gemini_pro:
        print("\nAVISO: 'gemini-pro' ou similar NÃO foi encontrado na lista de modelos de chat disponíveis.")
        print("Isso indica que sua chave API ou região não tem acesso a este modelo específico.")
    if not found_embedding:
        print("\nAVISO: 'embedding-001' ou similar NÃO foi encontrado na lista de modelos de embedding disponíveis.")
        print("Isso indica que sua chave API ou região não tem acesso a este modelo de embedding.")

except Exception as e:
    print(f"ERRO CRÍTICO ao tentar listar modelos: {e}")
    print("Por favor, verifique sua GOOGLE_API_KEY e sua conexão com a internet.")

print("Verificação de modelos concluída. Prosseguindo com o script principal...\n---")

# ---
## Continuação do Código Principal
# ---

# Inicializa o cliente Pinecone
pinecone_client = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# ---
## Carregamento e Processamento de Documentos
# ---

zip_file_path = 'documentos.zip'
extracted_folder_path = 'docs'

# CORREÇÃO AQUI: zipfile.ZipFile (com 'F' maiúsculo)
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

documents = []
for filename in os.listdir(extracted_folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(extracted_folder_path, filename)
        loader = PyMuPDFLoader(file_path)
        documents.extend(loader.load())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)
chunks = text_splitter.create_documents([doc.page_content for doc in documents])

# ---
## Embeddings e Vector Store
# ---

# Inicializa os embeddings com o modelo do Google Gemini
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", # Nome do modelo de embedding encontrado
    client_options={"api_endpoint": "generativelanguage.googleapis.com"}
)

# Cria ou obtém o índice Pinecone
index_name = 'llm'
pinecone_index = pinecone_client.Index(index_name)

# Inicializa o PineconeVectorStore
vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings, text_key='page_content')

# ---
## Configuração do Modelo de Linguagem (LLM)
# ---

# Inicializa o LLM com um modelo Gemini que está disponível para sua chave
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-latest", # Nome do modelo de chat disponível
    temperature=0.2,
    client_options={"api_endpoint": "generativelanguage.googleapis.com"}
)

# Configura o retriever para buscar documentos semelhantes
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

# Cria a cadeia de RetrievalQA
chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

# ---
## Interface Streamlit
# ---

# Configurações da página Streamlit
st.set_page_config(
    page_title="JuridicaMente",
    page_icon="⚖️",
)

# Estilos CSS para os campos de texto
st.markdown(
    """
    <style>
    input[type="text"] {
        font-size: 16px !important;
    }
    textarea {
        font-size: 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Colunas para o layout do logo
col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    st.image("logo.png", caption="JuridicaMente", width=200)

# Título principal da aplicação
st.header("Busca semântica de resultados jurídicos por IA para aproximar você das decisões, alegações e processos mais relevantes.")

# Formulário de busca
query = st.text_area("Digite sua pergunta:")
buscar = st.button("Buscar")

# Lógica de busca quando o botão é clicada
if buscar and query:
    with st.spinner("Buscando resposta..."):
        response = chain.invoke(query)
        st.subheader("Resposta:")
        st.write(response['result'])