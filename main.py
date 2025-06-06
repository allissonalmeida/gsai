from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
import os
import zipfile
import streamlit as st
import time

# --- Importações para a solução local com FAISS ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings # Mantém o embedding HuggingFace
from langchain_openai import ChatOpenAI # Mantém o LLM DeepSeek

# ---
## Configuração e Inicialização
# ---

# Configura as chaves de API (apenas DEEPSEEK_API_KEY será usada, pois Pinecone não é mais necessário)
os.environ['DEEPSEEK_API_KEY'] = st.secrets['DEEPSEEK_API_KEY']
# PINECONE_API_KEY não é mais necessário para embeddings, mas pode deixar no secrets se usar em outro lugar

# ---
## Carregamento e Processamento de Documentos
# ---

zip_file_path = 'documentos.zip'
extracted_folder_path = 'docs'
faiss_index_path = "faiss_index" # Caminho para salvar/carregar o índice FAISS

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
## Embeddings e Vector Store (USANDO FAISS LOCALMENTE)
# ---

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _ = embeddings.embed_query("teste de embedding")
    st.success("Modelo de embedding HuggingFace inicializado com sucesso!")
except Exception as e:
    st.error(f"Erro ao inicializar ou testar embeddings HuggingFace. Verifique a instalação de 'sentence-transformers'. Erro: {e}")
    st.stop()

try:
    st.info("Verificando se o índice FAISS já existe localmente...")
    if os.path.exists(faiss_index_path):
        st.success(f"Índice FAISS encontrado em '{faiss_index_path}'. Carregando...")
        vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        st.success("Índice FAISS carregado com sucesso!")
    else:
        st.warning(f"Índice FAISS não encontrado. Criando novo índice e salvando em '{faiss_index_path}' (isso pode levar tempo)...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(faiss_index_path)
        st.success("Índice FAISS criado e salvo com sucesso!")

except Exception as e:
    st.error(f"Erro crítico ao gerenciar ou conectar ao índice FAISS. Erro: {e}")
    st.stop()

# ---
## Configuração do Modelo de Linguagem (LLM)
# ---

# Inicializa o LLM com um modelo de chat DeepSeek (usando a interface OpenAI)
try:
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ['DEEPSEEK_API_KEY'],
        base_url="https://api.deepseek.com/v1",
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
from langchain.chains import RetrievalQA # Já importado, mas reforçando
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

    with st.spinner("Buscando resposta..."):
        try:
            response = chain.invoke(query)
            st.subheader("Resposta:")
            st.write(response['result'])
        except Exception as e:
            st.error(f"Ocorreu um erro ao buscar a resposta. Detalhes: {e}")
            st.info("Por favor, verifique os logs do Streamlit Cloud para mais detalhes sobre o erro da API DeepSeek (NotFoundError, QuotaExceeded, etc.).")