from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone, PodSpec # Mantenha a importação de PodSpec
import os
import zipfile
import streamlit as st
import time # Importação adicionada para possível atraso

# Importações para o LLM DeepSeek (via interface OpenAI)
from langchain_openai import ChatOpenAI
# Importação para o modelo de Embedding de Código Aberto (HuggingFace)
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---
## Configuração e Inicialização
# ---

# Configura as chaves de API
os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY']
os.environ['DEEPSEEK_API_KEY'] = st.secrets['DEEPSEEK_API_KEY'] # Chave de API para o LLM DeepSeek

# ---
## Continuação do Código Principal
# ---

# INICIALIZA O CLIENTE PINECONE AQUI, FORA DE QUALQUER BLOCO CONDICIONAL/FUNÇÃO.
# ISSO GARANTE QUE 'pinecone_client' ESTEJA DEFINIDO GLOBALMENTE.
try:
    pinecone_client = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
except Exception as e:
    st.error(f"Erro ao inicializar o cliente Pinecone. Verifique sua 'PINECONE_API_KEY'. Erro: {e}")
    st.stop()


# ---
## Carregamento e Processamento de Documentos
# ---

zip_file_path = 'documentos.zip'
extracted_folder_path = 'docs'

# Garante que a pasta 'docs' exista
if not os.path.exists(extracted_folder_path):
    os.makedirs(extracted_folder_path)

# Extrai os documentos do ZIP
try:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        st.info(f"Extraindo documentos de '{zip_file_path}' para '{extracted_folder_path}'...")
        zip_ref.extractall(extracted_folder_path)
        st.success("Documentos extraídos com sucesso!")
except FileNotFoundError:
    st.error(f"Erro: O arquivo ZIP '{zip_file_path}' não foi encontrado. Certifique-se de que ele está na mesma pasta do seu script.")
    st.stop()
except Exception as e:
    st.error(f"Erro ao extrair o arquivo ZIP. Verifique se ele está válido. Erro: {e}")
    st.stop()

documents = []
failed_pdf_loads = []

st.info("Carregando e processando documentos PDF...")
for filename in os.listdir(extracted_folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(extracted_folder_path, filename)
        try:
            loader = PyMuPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            failed_pdf_loads.append(filename)
            print(f"Aviso: Não foi possível carregar o PDF '{filename}'. Erro: {e}")

if failed_pdf_loads:
    st.warning(f"Atenção: Não foi possível carregar os seguintes PDFs: {', '.join(failed_pdf_loads)}. Eles foram ignorados.")

if not documents:
    st.error("Erro: Nenhum documento PDF foi carregado ou extraído com sucesso. Verifique o arquivo ZIP e os PDFs.")
    st.stop()
st.success(f"{len(documents)} documentos PDF carregados.")

st.info("Dividindo documentos em chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)

processed_documents_content = []
for doc in documents:
    if hasattr(doc, 'page_content'):
        processed_documents_content.append(doc.page_content)
    elif isinstance(doc, str):
        processed_documents_content.append(doc)
    else:
        print(f"Aviso: Documento de tipo inesperado ignorado: {type(doc)}")

if not processed_documents_content:
    st.error("Erro: Nenhum conteúdo válido foi extraído dos documentos para chunking.")
    st.stop()

chunks = text_splitter.create_documents(processed_documents_content)
st.success(f"{len(chunks)} chunks de texto criados.")

# ---
## Embeddings e Vector Store
# ---

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _ = embeddings.embed_query("teste de embedding")
    st.success("Modelo de embedding HuggingFace inicializado com sucesso!")
except Exception as e:
    st.error(f"Erro ao inicializar ou testar embeddings HuggingFace. Verifique a instalação de 'sentence-transformers'. Erro: {e}")
    st.stop()

index_name = 'llm'
dimension = 384
metric = 'cosine'

try:
    st.info(f"Verificando a existência do índice '{index_name}' no Pinecone...")
    existing_indexes = [index['name'] for index in pinecone_client.list_indexes()]

    if index_name not in existing_indexes:
        st.warning(f"Índice '{index_name}' não encontrado. Criando novo índice no Pinecone...")

        # --- AQUI ESTÁ A CORREÇÃO FINAL PARA O NOVO PROJETO 'gsai-project' ---
        # SUBSTITUA "SEU_NOVO_ENVIRONMENT_DO_GSAI_PROJECT" pelo ambiente exato (e.g., "us-east-1-aws")
        # que o Pinecone atribuiu ao seu novo projeto 'gsai-project'.
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=PodSpec(environment="SEU_NOVO_ENVIRONMENT_DO_GSAI_PROJECT") # <-- AJUSTE AQUI!
        )

        st.success(f"Índice '{index_name}' criado com sucesso! Aguardando o índice ficar pronto...")
        while not pinecone_client.describe_index(index_name).status.ready:
            time.sleep(1)
        st.success("Índice pronto para uso!")

    pinecone_index = pinecone_client.Index(index_name)
    index_stats = pinecone_index.describe_index_stats()
    if index_stats.total_vector_count == 0:
        st.info("Índice Pinecone está vazio. Preenchendo com novos embeddings dos documentos (isso pode levar tempo)...")
        vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name, text_key='page_content')
        st.success("Índice Pinecone preenchido com sucesso!")
    else:
        st.info(f"Índice Pinecone '{index_name}' já existe com {index_stats.total_vector_count} vetores. Usando índice existente.")
        vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings, text_key='page_content')

except Exception as e:
    st.error(f"Erro crítico ao gerenciar ou conectar ao índice Pinecone. Por favor, verifique: 'PINECONE_API_KEY', o nome do índice, DIMENSÃO (384), e a CONFIGURAÇÃO 'spec' (environment/cloud/region). O erro foi: {e}")
    st.stop()

# ---
## Configuração do Modelo de Linguagem (LLM)
# ---

try:
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ['DEEPSEEK_API_KEY'],
        base_url="https://api.deepseek.com/v1",
        temperature=0.2,
    )
    _ = llm.invoke("Olá")
    st.success("Modelo de chat DeepSeek inicializado com sucesso!")
except Exception as e:
    st.error(f"Erro ao inicializar ou testar LLM DeepSeek. Verifique 'DEEPSEEK_API_KEY', o nome do 'model' e o 'base_url'. Erro: {e}")
    st.stop()

retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

# ---
## Interface Streamlit
# ---

st.set_page_config(
    page_title="JuridicaMente",
    page_icon="⚖️",
)

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

col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    st.image("logo.png", caption="JuridicaMente", width=200)

st.header("Busca semântica de resultados jurídicos por IA para aproximar você das decisões, alegações e processos mais relevantes.")

query = st.text_area("Digite sua pergunta:")
buscar = st.button("Buscar")

if buscar and query:
    if not os.environ.get('DEEPSEEK_API_KEY'):
        st.error("A chave 'DEEPSEEK_API_KEY' não está configurada. Por favor, adicione-a aos seus segredos do Streamlit.")
        st.stop()
    if not os.environ.get('PINECONE_API_KEY'):
        st.error("A chave 'PINECONE_API_KEY' não está configurada. Por favor, adicione-a aos seus segredos do Streamlit.")
        st.stop()

    with st.spinner("Buscando resposta..."):
        try:
            response = chain.invoke(query)
            st.subheader("Resposta:")
            st.write(response['result'])
        except Exception as e:
            st.error(f"Ocorreu um erro ao buscar a resposta. Detalhes: {e}")
            st.info("Por favor, verifique os logs do Streamlit Cloud para mais detalhes sobre o erro da API DeepSeek (NotFoundError, QuotaExceeded, etc.).")