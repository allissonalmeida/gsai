from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone
import os
import zipfile
import streamlit as st

# Importações para DeepSeek (a API do DeepSeek é compatível com o padrão OpenAI)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# ---
## Configuração e Inicialização
# ---

# Configura as chaves de API
os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY']
os.environ['DEEPSEEK_API_KEY'] = st.secrets['DEEPSEEK_API_KEY'] # Nova chave de API para o DeepSeek

# ---
## Continuação do Código Principal
# Não há mais o bloco de diagnóstico de modelos Gemini aqui, pois estamos usando DeepSeek.
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

# Inicializa os embeddings com o modelo DeepSeek (usando a interface OpenAI)
# ATENÇÃO: Se você mudar o modelo de embedding, DEVE REINDEXAR SEUS DOCUMENTOS NO PINECONE.
# Isso significa que você precisará apagar o índice 'llm' existente no Pinecone
# e então rodar este código para recriá-lo com os novos embeddings do DeepSeek.
embeddings = OpenAIEmbeddings(
    model="deepseek-embeddings", # << VERIFIQUE O NOME EXATO DO MODELO DE EMBEDDING DO DEEPSEEK
    api_key=os.environ['DEEPSEEK_API_KEY'],
    base_url="https://api.deepseek.com/v1", # Endpoint padrão da API do DeepSeek
)

# Cria ou obtém o índice Pinecone
index_name = 'llm'
pinecone_index = pinecone_client.Index(index_name)

# Inicializa o PineconeVectorStore
# Ao fazer isso pela primeira vez com o novo modelo de embedding,
# ele irá popular seu índice no Pinecone com os novos vetores.
vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings, text_key='page_content')


# ---
## Configuração do Modelo de Linguagem (LLM)
# ---

# Inicializa o LLM com um modelo de chat DeepSeek (usando a interface OpenAI)
llm = ChatOpenAI(
    model="deepseek-chat", # << VERIFIQUE O NOME EXATO DO MODELO DE CHAT DO DEEPSEEK
    api_key=os.environ['DEEPSEEK_API_KEY'],
    base_url="https://api.deepseek.com/v1", # Endpoint padrão da API do DeepSeek
    temperature=0.2,
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