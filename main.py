# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore  
# from langchain.chains import RetrievalQA
# from langchain_openai import ChatOpenAI
# from pinecone import Pinecone
# import yaml
# import os
# import zipfile
import streamlit as st


# # Load configuration from config.yaml
# with open('config.yaml', 'r') as config_file:
#     config = yaml.safe_load(config_file)
# os.environ['PINECONE_API_KEY'] = config['PINECONE_API_KEY']
# os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

# # Inicialize o cliente Pinecone
# pinecone_client = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# # Load documents from the zip file
# zip_file_path = 'documentos.zip'
# extracted_folder_path = 'docs'

# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(extracted_folder_path)

# documents = []
# for filename in os.listdir(extracted_folder_path):
#     if filename.endswith(".pdf"):
#         file_path = os.path.join(extracted_folder_path, filename)
#         loader = PyMuPDFLoader(file_path)
#         documents.extend(loader.load())


# # Split documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100,
#     length_function=len
# )
# chunks = text_splitter.create_documents([doc.page_content for doc in documents])


# # Initialize
# embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')


# # Create Pinecone index
# index_name = 'llm'
# # Obtenha o índice Pinecone
# pinecone_index = pinecone_client.Index(index_name)

# # Inicialize o PineconeVectorStore da forma recomendada pelo pacote langchain-pinecone
# vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings, text_key='page_content')


# # Questions
# query_1 = '''Responda apenas com base no input fornecido. Qual o número do processo que trata de Violação
# de normas ambientais pela Empresa de Construção?'''
# query_2 = 'Responda apenas com base no input fornecido. Qual foi a decisão no caso de fraude financeira?'
# query_3 = 'Responda apenas com base no input fornecido. Quais foram as alegações no caso de negligência médica?'
# query_4 = 'Responda apenas com base no input fornecido. Quais foram as alegações no caso de Número do Processo: 822162' #disputa contratual


# llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.2)

# retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

# chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

# answer_1 = chain.invoke(query_1)
# answer_2 = chain.invoke(query_2)
# answer_3 = chain.invoke(query_3)
# answer_4 = chain.invoke(query_4)


# print('Pergunta: ',answer_1['query'])
# print('Resultado: ',answer_1['result'],'\n')
# #---
# print('Pergunta: ',answer_2['query'])
# print('Resultado: ',answer_2['result'],'\n')
# #---
# print('Pergunta: ',answer_3['query'])
# print('Resultado: ',answer_3['result'],'\n')
# #---
# print('Pergunta: ',answer_4['query'])
# print('Resultado: ',answer_4['result'])


# Configuration
col1, col2, col3 = st.columns([1, 3, 1])  # Ajuste as proporções conforme necessário
with col2:
    st.image("logo.png", caption="JuridicaMente", width=200)
    
st.markdown(
    """
    <style>
    input[type="text"] {
        font-size: 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    textarea {
        font-size: 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    
st.title("Busca semântica de resultados jurídicos alimentada por IA para aproximar você das decisões, alegações e processos mais relevantes.")


# Formulario de busca
query = st.text_area("Digite sua pergunta:")
buscar = st.button("Buscar")

if buscar and query:
    with st.spinner("Buscando resposta..."):
        #response = chain.invoke(query)
        st.subheader("Resposta:")
        #st.write(response['result'])
        st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")