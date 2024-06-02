"""Main module."""

import hashlib
import logging
import os
from typing import Optional, List, Dict, Any
import glob
import boto3
from toolz.itertoolz import partition_all
import pandas as pd
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
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredEPubLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredRSTLoader,
    UnstructuredExcelLoader,
    DataFrameLoader,
)

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
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)

    return docs


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
    def __init__(self, **kwargs):
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

    @property
    def connection_string(self):
        driver = "psycopg2"
        user = os.environ.get("POSTGRES_USER", "postgres")
        password = os.environ.get("POSTGRES_PASSWORD")
        host = os.environ.get("POSTGRES_HOST")
        port = os.environ.get("POSTGRES_PORT")
        database = os.environ.get("POSTGRES_DB")
        connection = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
        return connection

    @property
    def conn(self):
        driver = "psycopg2"
        user = os.environ.get("POSTGRES_USER", "postgres")
        password = os.environ.get("POSTGRES_PASSWORD")
        host = os.environ.get("POSTGRES_HOST")
        port = os.environ.get("POSTGRES_PORT")
        database = os.environ.get("POSTGRES_DB")
        connection = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
        # Establish the connection to the database
        conn = psycopg.connect(
            conninfo=f"postgresql://{user}:{password}@{host}:{port}/{database}"
        )
        return conn

    @property
    def cur(self):
        conn = self.connection_string()
        return conn.cursor()

    def create_vectorstore(
        self, collection_name: str, embeddings: Optional[BedrockEmbeddings] = None
    ):
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=self.connection_string,
            use_jsonb=True,
        )
        return vectorstore

    def load_local_file_to_document(self, file) -> List[Document]:
        loader_data = get_loader(file)
        data = loader_data["loader"].load()
        # data.id = hashlib.sha256(data.page_content.encode()).hexdigest()
        return data

    def run_ingestion_job(
        self,
        documents=List[Document],
        collection_name: str = "default",
    ):
        y = len(documents)
        filtered_docs = []
        for d in documents:
            if len(d.page_content):
                filtered_docs.append(d)
        ids = []
        for d in filtered_docs:
            ids.append(hashlib.sha256(d.page_content.encode()).hexdigest())

        if len(filtered_docs):
            vectorstore = self.create_vectorstore(collection_name=collection_name)
            # texts = [i.page_content for i in filtered_docs]
            # metadatas = [i.metadata for i in filtered_docs]
            # logging.info(f"Adding N: {len(filtered_docs)}")
            try:
                with funcy.print_durations("load psql"):
                    vectorstore.documents(documents=filtered_docs, ids=ids)
            except Exception as e:
                logging.warning(f"{e}")
            # logging.info(f"Complete {x}/{y}")
        return ids

    def setup_local_injestion_job(
        self,
        glob_pattern: str,
        chunk_size: int = 1000,
        collection_name: str = "default",
    ):
        files = glob.glob(glob_pattern)
        docs = []
        for p in partition_all(chunk_size, files):
            for file in p:
                doc = self.load_local_file_to_document(file)
                docs = docs + doc
            ids = self.run_ingestion_job(
                documents=docs, collection_name=collection_name
            )
        return

    def setup_local_parquet_job(
        self,
        files,
        page_content_column: str = "id",
        chunk_size: int = 1000,
        collection_name: str = "default",
    ):
        x = 0
        y = len(files)
        ids = []
        docs = []
        for p in partition_all(chunk_size, files):
            for file in p:
                df = pd.read_parquet(file)
                loader = DataFrameLoader(df, page_content_column=page_content_column)
                data: List[Document] = loader.load()
                docs = docs + data
            ids = self.run_ingestion_job(
                documents=docs, collection_name=collection_name
            )
        return
