import os
import streamlit as st
import chromadb

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# ---
## 0. Configuração da Página Streamlit
# ---
st.set_page_config(
    page_title="JuridicaMente",
    page_icon="📈",
    layout="wide"
)

# ---
## 1. Configuração e Inicialização de Chaves API
# ---

os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Variáveis globais para armazenar os nomes dos modelos encontrados
GLOBAL_CHAT_MODEL_NAME = None
GLOBAL_EMBEDDING_MODEL_NAME = "models/embedding-001" 

# ---
## 2. Diagnóstico de Modelos Gemini
# ---

# Configura e lista modelos imediatamente.
st.write("Iniciando verificação de modelos Gemini disponíveis...")
try:
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'], client_options={"api_endpoint": "generativelanguage.googleapis.com"})

    chat_models_found = []
    embedding_models_found = []

    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            chat_models_found.append(m.name)
        if "embedContent" in m.supported_generation_methods:
            embedding_models_found.append(m.name)

    if "models/gemini-1.5-flash-latest" in chat_models_found:
        GLOBAL_CHAT_MODEL_NAME = "models/gemini-1.5-flash-latest"
    elif "models/gemini-1.5-flash" in chat_models_found:
        GLOBAL_CHAT_MODEL_NAME = "models/gemini-1.5-flash"
    elif "models/gemini-1.5-pro-latest" in chat_models_found:
        GLOBAL_CHAT_MODEL_NAME = "models/gemini-1.5-pro-latest"
    elif "models/gemini-1.5-pro" in chat_models_found:
        GLOBAL_CHAT_MODEL_NAME = "models/gemini-1.5-pro"
    elif "models/gemini-pro" in chat_models_found: 
        GLOBAL_CHAT_MODEL_NAME = "models/gemini-pro"
    else:
        for model_name in chat_models_found:
            
            if "gemini" in model_name and \
               "vision" not in model_name and \
               "exp" not in model_name and \
               "preview" not in model_name and \
               "tts" not in model_name and \
               "tuning" not in model_name and \
               "8b" not in model_name: 
                GLOBAL_CHAT_MODEL_NAME = model_name
                break
        if not GLOBAL_CHAT_MODEL_NAME and chat_models_found: 
            st.warning("Não foi possível encontrar um modelo preferencial.")
            GLOBAL_CHAT_MODEL_NAME = chat_models_found[0] 

    if "models/embedding-001" in embedding_models_found:
        GLOBAL_EMBEDDING_MODEL_NAME = "models/embedding-001"
    elif embedding_models_found:
        GLOBAL_EMBEDDING_MODEL_NAME = embedding_models_found[0]

except Exception as e:
    st.error(f"ERRO na inicialização da API Gemini: {e}")
    st.error("Por favor, verifique sua `GOOGLE_API_KEY` em `.streamlit/secrets.toml` e sua conexão com a internet.")
    st.info("Se você estiver usando uma chave gratuita, pode ter atingido os limites de uso.")
    st.stop() 
    
# ---
# Mostra os resultados do diagnóstico que já foi feito
# with st.expander("Verificar Detalhes dos Modelos Gemini Encontrados"):
#     if GLOBAL_CHAT_MODEL_NAME:
#         st.success("Modelos de Chat disponíveis:")
#         for model_name in chat_models_found:
#             st.write(f"- `{model_name}`")
#         st.info(f"💡 **Modelo de chat selecionado para uso:** `{GLOBAL_CHAT_MODEL_NAME}`")
#     else:
#         st.warning("Nenhum modelo de chat adequado foi encontrado. A funcionalidade de chat não estará disponível.")

#     if GLOBAL_EMBEDDING_MODEL_NAME:
#         st.success("Modelos de Embedding disponíveis:")
#         for model_name in embedding_models_found:
#             st.write(f"- `{model_name}`")
#         st.info(f"💡 **Modelo de embedding selecionado para uso:** `{GLOBAL_EMBEDDING_MODEL_NAME}`")
#     else:
#         st.warning("Nenhum modelo de embedding adequado foi encontrado. A funcionalidade de embeddings não estará disponível.")

# st.write("---") 

# ---
## 3. Carregamento e Processamento de Documentos
# ---

extracted_folder_path = 'temp_financial_reports'
os.makedirs(extracted_folder_path, exist_ok=True)

@st.cache_data(show_spinner="Processando relatórios e preparando para análise...")
def load_and_process_documents(uploaded_files):
    all_documents = []
    temp_pdf_paths = []

    if not uploaded_files:
        return []

    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(extracted_folder_path, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        temp_pdf_paths.append(temp_file_path)

        loader = PyMuPDFLoader(temp_file_path)
        docs_from_pdf = loader.load()

        for doc in docs_from_pdf:
            doc.metadata['source'] = uploaded_file.name
            all_documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(all_documents)

    for path in temp_pdf_paths:
        if os.path.exists(path):
            os.remove(path)
    try:
        os.rmdir(extracted_folder_path)
    except OSError:
        pass

    return chunks

# ---
## 4. Embeddings e Vector Store (ChromaDB)
# ---
embeddings = None
if GLOBAL_EMBEDDING_MODEL_NAME:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GLOBAL_EMBEDDING_MODEL_NAME,
        client_options={"api_endpoint": "generativelanguage.googleapis.com"}
    )
else:
    st.error("Modelo de Embedding não inicializado.")


persist_directory = "./chroma_db_data"
os.makedirs(persist_directory, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=persist_directory)

@st.cache_resource(show_spinner="Configurando o banco de dados vetorial...")
def get_vector_store(_chunks, _embeddings, _chroma_client, collection_name="financial_reports_collection"):
    if collection_name in [c.name for c in _chroma_client.list_collections()]:
        _chroma_client.delete_collection(collection_name)
        st.info(f"Coleção '{collection_name}' existente foi resetada para carregar novos documentos.")

    vector_store = Chroma.from_documents(
        documents=_chunks,
        embedding=_embeddings,
        client=_chroma_client,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    vector_store.persist()
    #st.success(f"Embeddings gerados e armazenados em ChromaDB (coleção: '{collection_name}'). Total de chunks: {len(_chunks)}")
    return vector_store

# ---
## 5. Configuração do Modelo de Linguagem (LLM)
# ---
llm = None
if GLOBAL_CHAT_MODEL_NAME:
    llm = ChatGoogleGenerativeAI(
        model=GLOBAL_CHAT_MODEL_NAME,
        temperature=0.2,
        client_options={"api_endpoint": "generativelanguage.googleapis.com"}
    )
else:
    st.error("Modelo de Chat (LLM) não inicializado.")


# ---
## 6. Interface Streamlit: Melhorias na UI
# ---

# Estilos CSS personalizados (sem alterações neste bloco)
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Arial', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        padding-left: 5rem;
        padding-right: 5rem;
        max-width: 1200px;
    }

    .header-content {
        display: flex;
        flex-direction: column;
        align-items: center; /* Centraliza horizontalmente */
        text-align: center; /* Centraliza o texto */
        margin-bottom: 20px; /* Espaço abaixo do cabeçalho */
    }

    .logo-title-container {
        display: flex;
        align-items: center;
        gap: 20px; /* Espaço entre logo e título */
        margin-bottom: 0px; /* Reduz espaçamento entre logo/titulo e subtitulo */
    }

    .logo-title-container img {
        max-height: 80px;
        border-radius: 10px;
    }

    h1 {
        font-size: 2.5em;
        color: #2F4F4F;
        margin-bottom: 0.2em;
    }
    h2 {
        font-size: 1.8em;
        color: #36454F;
        border-bottom: 2px solid #D3D3D3;
        padding-bottom: 5px;
        margin-top: 2em;
        margin-bottom: 1em;
    }
    h3 {
        font-size: 1.4em;
        color: #4F6F4F;
        margin-top: 0.5em; /* Ajuste para o subtitulo principal */
    }

    .stFileUploader {
        border: 2px dashed #ADD8E6;
        border-radius: 10px;
        padding: 20px;
        background-color: #F8F8F8;
        margin-bottom: 20px;
    }

    textarea {
        border: 1px solid #ADD8E6 !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-size: 1.1em !important;
        min-height: 120px !important;
    }
    .stButton > button {
        background-color: #4F6F4F;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1em;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #6B8E23;
    }

    /* Container de mensagens de sucesso/erro */
    [data-testid="stAlert"] {
        border-radius: 8px;
        font-size: 0.95em;
    }
    [data-testid="stSuccess"] {
        background-color: #e6ffe6;
        color: #006400;
        border-left: 5px solid #00C800;
    }
    [data-testid="stWarning"] {
        background-color: #fffacd;
        color: #FF8C00;
        border-left: 5px solid #FFA500;
    }
    [data-testid="stError"] {
        background-color: #ffe6e6;
        color: #8B0000;
        border-left: 5px solid #FF0000;
    }

    .response-container {
        background-color: #F0F8FF;
        border: 1px solid #ADD8E6;
        border-radius: 10px;
        padding: 25px;
        margin-top: 30px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .response-container h3 {
        margin-top: 0;
        color: #2F4F4F;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cabeçalho 
st.markdown("<div class='header-content'>", unsafe_allow_html=True)
st.markdown("<div class='logo-title-container'>", unsafe_allow_html=True)
if os.path.exists("logo.png"):
    st.image("logo.png", width=120)
else:
    st.write("*(Logo FinAI)*")
st.markdown("<h1>JuridicaMente</h1>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True) 
st.markdown("<h3>Análise de Processos Jurídicos por IA</h3>", unsafe_allow_html=True)
st.markdown("Busca semântica de resultados para aproximar você das decisões, alegações e processos mais relevantes.")
st.markdown("</div>", unsafe_allow_html=True) 

#st.markdown("---")

st.subheader("Carregue seus documentos (PDF)")
uploaded_files = st.file_uploader(
    "Selecione um ou mais arquivos PDF para análise:",
    type="pdf",
    accept_multiple_files=True,
    key="pdf_uploader"
)

vector_store = None
chunks = []

if uploaded_files:
    if not embeddings:
        st.error("Não foi possível processar os documentos porque o modelo de embedding não foi inicializado corretamente.")
    else:
        chunks = load_and_process_documents(uploaded_files)

        if chunks:
            vector_store = get_vector_store(chunks, embeddings, chroma_client)
            retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 10}) 
            if llm:
                chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
                st.success("✅ Relatórios processados e prontos para consulta!")
            else:
                st.error("Não foi possível criar a cadeia de QA porque o modelo de chat (LLM) não foi inicializado corretamente.")
        else:
            st.warning("⚠️ Nenhum texto útil foi extraído dos PDFs carregados.")
else:
    st.info("⬆️ Por favor, carregue um ou mais documentos jurídicos em PDF para começar a análise.")


if vector_store and llm:
    st.subheader("Pergunte aos Seus Documentos Jurídicos")
    query = st.text_area(
        "Formule sua pergunta sobre os relatórios carregados:",
        placeholder="",
        height=120,
        key="query_input"
    )

    col_btn_examples, col_btn_search = st.columns([1, 4])

    with col_btn_examples:
        if st.button("Exemplos de Perguntas", key="show_examples"):
            st.markdown("""
            - Responda apenas com base no input fornecido. Qual o número do processo que trata de Violação de normas ambientais pela Empresa de Construção?
            - Responda apenas com base no input fornecido. Quais foram as alegações no caso de negligência médica?
            - Responda apenas com base no input fornecido. Quais foram as alegações no caso de Número do Processo: XXXXXXX
            - Responda apenas com base no input fornecido. Qual foi a decisão no caso de fraude financeira?
            """)
  
    with col_btn_search:
        buscar = st.button("Obter Insights", key="search_button")

    if buscar and query:
        with st.spinner("Buscando e gerando insights..."):
            try:
                response = chain.invoke({"query": query})

                st.markdown("<div class='response-container'>", unsafe_allow_html=True)
                st.markdown("<h3>Insights Gerados:</h3>", unsafe_allow_html=True)
                st.write(response['result'])
                st.markdown("</div>", unsafe_allow_html=True)


                if 'source_documents' in response and response['source_documents']:
                    st.markdown("---")
                    st.subheader("Documentos de Referência:")
                    for i, doc in enumerate(response['source_documents']):
                        source_info = doc.metadata.get('source', 'Documento Desconhecido')
                        page_info = doc.metadata.get('page', 'N/A')
                        st.markdown(f"**{i+1}. Fonte:** `{source_info}` (Página: `{page_info}`)")
                        st.markdown(f"```text\n{doc.page_content[:300]}...\n```")
            except Exception as e:
                st.error(f"🚫 Ocorreu um erro ao gerar os insights: {e}")
                st.error("Por favor, tente reformular sua pergunta ou verifique se os documentos contêm a informação.")
else:
    if uploaded_files and (not llm or not embeddings):
         st.error("Funcionalidade de análise indisponível.")