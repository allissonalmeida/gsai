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
# ---

# Inicializa o cliente Pinecone
pinecone_client = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

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
        zip_ref.extractall(extracted_folder_path)
except FileNotFoundError:
    st.error(f"Erro: O arquivo ZIP '{zip_file_path}' não foi encontrado. Certifique-se de que ele está na mesma pasta do seu script.")
    st.stop() # Para a execução do script se o arquivo não for encontrado

documents = []
# Lista para armazenar nomes de arquivos PDF que falharam no carregamento
failed_pdf_loads = []

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

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)
# É crucial que 'documents' contenha objetos de Document ou apenas strings
# Aqui, presumimos que loader.load() retorna objetos Document, então pegamos page_content
# Se loader.load() já retornar strings, removemos o .page_content
# chunks = text_splitter.create_documents([doc.page_content for doc in documents])
# Uma maneira mais robusta, caso 'documents' já sejam strings ou mistos:
processed_documents = []
for doc in documents:
    if hasattr(doc, 'page_content'):
        processed_documents.append(doc.page_content)
    elif isinstance(doc, str):
        processed_documents.append(doc)
    else:
        print(f"Aviso: Documento de tipo inesperado ignorado: {type(doc)}")

if not processed_documents:
    st.error("Erro: Nenhum conteúdo válido foi extraído dos documentos para chunking.")
    st.stop()

chunks = text_splitter.create_documents(processed_documents)

# ---
## Embeddings e Vector Store
# ---

# Inicializa os embeddings com o modelo DeepSeek (usando a interface OpenAI)
# ATENÇÃO: Se você mudar o modelo de embedding, DEVE REINDEXAR SEUS DOCUMENTOS NO PINECONE.
# Isso significa que você precisará apagar o índice 'llm' existente no Pinecone
# e então rodar este código para recriá-lo com os novos embeddings do DeepSeek.
try:
    embeddings = OpenAIEmbeddings(
        model="deepseek-embed", # << VERIFIQUE O NOME EXATO DO MODELO DE EMBEDDING DO DEEPSEEK
        api_key=os.environ['DEEPSEEK_API_KEY'],
        base_url="https://api.deepseek.com/v1", # Endpoint padrão da API do DeepSeek - VERIFIQUE PARA EMBEDDINGS
    )
except Exception as e:
    st.error(f"Erro ao inicializar embeddings DeepSeek. Verifique 'DEEPSEEK_API_KEY', 'model' e 'base_url'. Erro: {e}")
    st.stop()


# Cria ou obtém o índice Pinecone
index_name = 'llm'
try:
    pinecone_index = pinecone_client.Index(index_name)
    # Verifica se o índice está vazio ou precisa ser preenchido
    index_stats = pinecone_index.describe_index_stats()
    if index_stats.total_vector_count == 0:
        st.info("Índice Pinecone vazio. Preenchendo com novos embeddings...")
        vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name, text_key='page_content')
        st.success("Índice Pinecone preenchido com sucesso!")
    else:
        st.info(f"Índice Pinecone '{index_name}' já existe com {index_stats.total_vector_count} vetores. Usando índice existente.")
        vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings, text_key='page_content')

except Exception as e:
    st.error(f"Erro ao conectar ou criar índice Pinecone. Verifique 'PINECONE_API_KEY' e o nome do índice. Erro: {e}")
    st.stop()


# ---
## Configuração do Modelo de Linguagem (LLM)
# ---

# Inicializa o LLM com um modelo de chat DeepSeek (usando a interface OpenAI)
try:
    llm = ChatOpenAI(
        model="deepseek-chat", # << VERIFIQUE O NOME EXATO DO MODELO DE CHAT DO DEEPSEEK
        api_key=os.environ['DEEPSEEK_API_KEY'],
        base_url="https://api.deepseek.com/v1", # Endpoint padrão da API do DeepSeek
        temperature=0.2,
    )
except Exception as e:
    st.error(f"Erro ao inicializar LLM DeepSeek. Verifique 'DEEPSEEK_API_KEY', 'model' e 'base_url'. Erro: {e}")
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