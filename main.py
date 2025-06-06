from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone, PodSpec # Importação adicionada para PodSpec ou ServerlessSpec
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

# Split documents into chunks
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
## Embeddings e Vector Store (Lógica de CRIAÇÃO DE ÍNDICE APRIMORADA)
# ---

try:
    # Inicializa os embeddings com um modelo HuggingFace (modelo leve e eficaz)
    # Certifique-se de ter 'sentence-transformers' instalado (pip install sentence-transformers)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Testar se o modelo de embedding está funcionando (opcional, mas útil para depuração)
    _ = embeddings.embed_query("teste de embedding")
    st.success("Modelo de embedding HuggingFace inicializado com sucesso!")
except Exception as e:
    st.error(f"Erro ao inicializar ou testar embeddings HuggingFace. Verifique a instalação de 'sentence-transformers'. Erro: {e}")
    st.stop()


# NOME DO ÍNDICE PINECONE
index_name = 'llm'
# --- CRÍTICO: A dimensão para 'sentence-transformers/all-MiniLM-L6-v2' é 384 ---
dimension = 384 # AJUSTADO PARA A DIMENSÃO DO MODELO all-MiniLM-L6-v2
metric = 'cosine' # Métrica de similaridade (geralmente 'cosine' para embeddings)

# Lógica para verificar e criar o índice Pinecone
try:
    st.info(f"Verificando a existência do índice '{index_name}' no Pinecone...")
    existing_indexes = [index['name'] for index in pinecone_client.list_indexes()]

    if index_name not in existing_indexes:
        st.warning(f"Índice '{index_name}' não encontrado. Criando novo índice no Pinecone...")
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            # --- CRÍTICO: AJUSTE cloud e region CONFORME SEU PLANO/PREFERÊNCIA NO PINECONE ---
            # Para planos "Starter" ou "Free", geralmente é "aws" e "us-west-2" ou "us-east-1".
            # VERIFIQUE SEU PAINEL DO PINECONE para a região exata onde você tem permissão para criar.
            # Exemplo para Pod-based Starter, AJUSTE CONFORME O SEU!
            spec=PodSpec(environment="gcp-starter", region="us-central1") # OU seu ambiente/região
            # Ou para serverless (se disponível no seu plano):
            # spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )
        st.success(f"Índice '{index_name}' criado com sucesso! Aguardando o índice ficar pronto...")
        # É importante esperar um pouco para o índice ser provisionado
        while not pinecone_client.describe_index(index_name).status.ready:
            time.sleep(1) # Espera 1 segundo
        st.success("Índice pronto para uso!")

    # Agora que sabemos que o índice existe (ou foi criado), podemos nos conectar a ele.
    pinecone_index = pinecone_client.Index(index_name)

    # Verificar se o índice está vazio e preenchê-lo
    index_stats = pinecone_index.describe_index_stats()
    if index_stats.total_vector_count == 0:
        st.info("Índice Pinecone está vazio. Preenchendo com novos embeddings dos documentos (isso pode levar tempo)...")
        # from_documents é o método para adicionar documentos e seus embeddings
        vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name, text_key='page_content')
        st.success("Índice Pinecone preenchido com sucesso!")
    else:
        st.info(f"Índice Pinecone '{index_name}' já existe com {index_stats.total_vector_count} vetores. Usando índice existente.")
        # Se o índice já tem vetores, apenas inicializamos o vector_store com ele
        vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings, text_key='page_content')

except Exception as e:
    st.error(f"Erro crítico ao gerenciar ou conectar ao índice Pinecone. Por favor, verifique: 'PINECONE_API_KEY', nome do índice, DIMENSÃO do embedding (384), e a CONFIGURAÇÃO DE REGIÃO/CLOUD (spec). Erro: {e}")
    st.stop()


# ---
## Configuração do Modelo de Linguagem (LLM)
# ---

# Inicializa o LLM com um modelo de chat DeepSeek (usando a interface OpenAI)
try:
    llm = ChatOpenAI(
        model="deepseek-chat", # << VERIFIQUE O NOME EXATO DO MODELO DE CHAT DO DEEPSEEK (ex: "deepseek-coder", "deepseek-math", etc.)
        api_key=os.environ['DEEPSEEK_API_KEY'],
        base_url="https://api.deepseek.com/v1", # Endpoint padrão da API do DeepSeek
        temperature=0.2,
    )
    # Teste simples para ver se o LLM está acessível
    _ = llm.invoke("Olá")
    st.success("Modelo de chat DeepSeek inicializado com sucesso!")
except Exception as e:
    st.error(f"Erro ao inicializar ou testar LLM DeepSeek. Verifique 'DEEPSEEK_API_KEY', o nome do 'model' e o 'base_url'. Erro: {e}")
    st.stop()


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

# Lógica de busca quando o botão é clicado
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