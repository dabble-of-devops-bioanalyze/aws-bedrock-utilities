import hashlib
import math
import logging
import numpy as np
from io import StringIO
import os
from typing import Optional, List, Dict, Any
import glob
import boto3
from toolz.itertoolz import partition_all
import pandas as pd
import unstructured
from langchain_core.documents.base import Document

import funcy
import psycopg
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import (
    BedrockEmbeddings,
)  # to create embeddings for the documents.
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import CharacterTextSplitter
from rich.logging import RichHandler

from aws_bedrock_utilities.models.base import BedrockBase
from langchain_community.document_loaders import (
    WebBaseLoader,
    JSONLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredEPubLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredRSTLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    DataFrameLoader,
)
import logging

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.retrievers.bedrock import (
    AmazonKnowledgeBasesRetriever,
    RetrievalConfig,
    VectorSearchConfig,
)

from aws_bedrock_utilities.models.base import BedrockBase, RAGResults

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

known_source_ext = [
    "go",
    "py",
    "java",
    "sh",
    "bat",
    "ps1",
    "cmd",
    "js",
    "ts",
    "css",
    "cpp",
    "hpp",
    "h",
    "c",
    "cs",
    "sql",
    "log",
    "ini",
    "pl",
    "pm",
    "r",
    "dart",
    "dockerfile",
    "env",
    "php",
    "hs",
    "hsc",
    "lua",
    "nginxconf",
    "conf",
    "m",
    "mm",
    "plsql",
    "perl",
    "rb",
    "rs",
    "db2",
    "scala",
    "bash",
    "swift",
    "vue",
    "svelte",
]


def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    return pages


def insert_pdf_embeddings(files: List[str], vectorstore: PGVector):
    logging.info(f"Inserting {len(files)}")
    x = 1
    y = len(files)
    for file_path in files:
        logging.info(f"Splitting {file_path} {x}/{y}")

        try:
            with funcy.print_durations("process pdf"):
                docs = load_and_split_pdf(file_path)
        except Exception as e:
            logging.warning(f"Error loading docs")
            docs = []

        # Sometimes we get a document that doesn't have any content - probably an error with loading
        filtered_docs = []
        for d in docs:
            if len(d.page_content):
                filtered_docs.append(d)
        # Add documents to the vectorstore
        ids = []
        for d in filtered_docs:
            ids.append(hashlib.sha256(d.page_content.encode()).hexdigest())

        if len(filtered_docs):
            texts = [i.page_content for i in filtered_docs]
            metadatas = [i.metadata for i in filtered_docs]
            # logging.info(f"Adding N: {len(filtered_docs)}")
            try:
                with funcy.print_durations("load psql"):
                    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            except Exception as e:
                logging.warning(f"Error {x - 1}/{y}")
            # logging.info(f"Complete {x}/{y}")
        x = x + 1
    return files


def get_loader(filepath: str) -> Dict[str, Any]:
    file_ext = filepath.split(".")[-1].lower()
    known_type = True

    if file_ext == "pdf":
        loader = PyPDFLoader(filepath, extract_images=True)
    elif file_ext == "csv":
        loader = CSVLoader(filepath)
    elif file_ext == "rst":
        loader = UnstructuredRSTLoader(filepath, mode="elements")
    elif file_ext == "xml":
        loader = UnstructuredXMLLoader(filepath)
    elif file_ext == "md":
        loader = UnstructuredMarkdownLoader(filepath)
    elif file_ext == ".epub":
        loader = UnstructuredEPubLoader(filepath)
    elif file_ext in ["doc", "docx"]:
        loader = Docx2txtLoader(filepath)
    elif file_ext == "pptx":
        loader = UnstructuredPowerPointLoader(filepath, mode="elements")
    elif file_ext in ["xls", "xlsx"]:
        loader = UnstructuredExcelLoader(filepath)
    elif file_ext == "json":
        loader = TextLoader(filepath, autodetect_encoding=True)
    elif file_ext in known_source_ext:
        loader = TextLoader(filepath, autodetect_encoding=True)
    else:
        loader = TextLoader(filepath, autodetect_encoding=True)
        known_type = False

    return dict(loader=loader, known_type=known_type, file_ext=file_ext)


class BedrockPGWrapper(BedrockBase):
    def __init__(self, collection_name: str = "default", **kwargs):
        super().__init__(**kwargs)
        if "prompt_template" not in kwargs:
            self.prompt_template = self.kb_prompt_template
        self.bedrock_embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1", client=self.bedrock_client
        )
        self.bedrock_embeddings_image = BedrockEmbeddings(
            model_id="amazon.titan-embed-image-v1", client=self.bedrock_client
        )
        self.s3 = boto3.client("s3")
        self.collection_name = collection_name

    @property
    def connection_string(self):
        driver = "psycopg2"
        user = os.environ.get("POSTGRES_USER", "postgres")
        password = os.environ.get("POSTGRES_PASSWORD")
        host = os.environ.get("POSTGRES_HOST")
        port = os.environ.get("POSTGRES_PORT", "5432")
        database = os.environ.get("POSTGRES_DB", "postgres")
        connection = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
        return connection

    @property
    def conn(self):
        driver = "psycopg2"
        user = os.environ.get("POSTGRES_USER", "postgres")
        password = os.environ.get("POSTGRES_PASSWORD")
        host = os.environ.get("POSTGRES_HOST")
        port = os.environ.get("POSTGRES_PORT", "5432")
        database = os.environ.get("POSTGRES_DB", "postgres")
        connection = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
        # Establish the connection to the database
        conn = psycopg.connect(
            conninfo=f"postgresql://{user}:{password}@{host}:{port}/{database}"
        )
        return conn

    @property
    def cur(self):
        conn = self.connection_string
        return conn.cursor()

    @property
    def embeddings(self):
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
        )
        return embeddings

    def create_vectorstore(self, collection_name: str):
        return PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=self.connection_string,
            use_jsonb=True,
        )

    @property
    def vectorstore(self):
        return PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection_string,
            use_jsonb=True,
        )

    def load_local_file_to_document(self, file) -> List[Document]:
        loader_data = get_loader(file)
        data = []
        try:
            data = loader_data["loader"].load_and_split()
        except NotImplementedError:
            data = loader_data["loader"].load()
        except Exception as e:
            data = []
        finally:
            if not isinstance(data, list):
                data = []

        filtered_docs = []
        for d in data:
            if len(d.page_content):
                filtered_docs.append(d)
        return filtered_docs

    def run_ingestion_job(
        self,
        documents=list[Document],
    ):
        y = len(documents)
        logging.info(f"Starting ingestion job with {y} documents")
        ids = []
        for d in documents:
            ids.append(
                hashlib.sha256(d.page_content.encode()).hexdigest()
            )
        texts = [i.page_content for i in documents]
        # metadata is a dictionary. You can add to it!
        metadatas = [i.metadata for i in documents]
        # logging.info(f"Adding N: {len(filtered_docs)}")

        doc_ids = []
        if len(documents):
            try:
                with funcy.print_durations(f'load psql: {len(documents)}'):
                    doc_ids = self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            except Exception as e:
                logging.warning(f"{e}")
        return doc_ids

    def setup_local_injestion_job(
        self,
        chunk_size: int = 1,
        glob_pattern: Optional[str] = None,
        files: Optional[List[str]] = None,
    ):
        if not files:
            files = glob.glob(glob_pattern)
        docs = []
        x = 0
        total_chunks = math.ceil(len(files) / chunk_size)
        all_ids = []
        for p in partition_all(chunk_size, files):
            logging.info(f"Loading x: {x} of {total_chunks} for")
            for file in p:
                logging.info(f"Processing: {file}")
                t_docs = self.load_local_file_to_document(file)
                docs = docs + t_docs
            ids = self.run_ingestion_job(documents=docs)
            all_ids = all_ids + ids
        return all_ids

    def setup_local_parquet_job(
        self,
        files,
        page_content_column: str = "id",
        chunk_size: int = 1000,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ):
        x = 0
        y = len(files)
        total_chunks = math.ceil(y / chunk_size)
        docs = []
        for p in partition_all(chunk_size, files):
            logging.info(f"Processing chunk {x} of {total_chunks}")
            for file in p:
                df = pd.read_parquet(file)
                df = df.replace(np.nan, None)
                text = df.to_json(orient='records', lines=True)
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    # chunk_size=1000,
                    # chunk_overlap=200,
                    length_function=len,
                    is_separator_regex=False,
                )
                docs: List[Document] = text_splitter.create_documents([text])
                # loader = DataFrameLoader(df, page_content_column=page_content_column)
                # data: List[Document] = loader.load()
                if additional_metadata:
                    for d in docs:
                        d["metadata"].update(additional_metadata)
            logging.info(f"Running ingestion job")
            ids = self.run_ingestion_job(
                documents=docs,
            )
            x = x + 1
        return

    def run_kb_chat(
        self,
        query: str,
        collection_name: str,
        prompt_template=None,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> RAGResults:
        if not prompt_template:
            prompt_template = self.kb_prompt_template
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "input"]
        )
        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        combine_docs_chain = create_stuff_documents_chain(
            llm=self.get_llm(model_id=model_id),
            prompt=prompt,
        )
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        response = retrieval_chain.invoke(
            {
                "input": query,
            }
        )
        source_documents = response["context"]
        answer = response["answer"]
        query = query
        answer = RAGResults(
            source_documents=source_documents,
            result=answer,
            query=query,
        )
        return answer
