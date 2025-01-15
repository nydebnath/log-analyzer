#!/usr/bin/env python
# coding: utf-8

# -----------------------------------------------------------------------------
# Module: log_chatter.py
#
# Description: This is a RAG based application that uses LLM to
# analyze log data and provide insights to users. It includes functionality
# to search through log files and retrieve relevant log entries based on
# specific tool IDs. The system is designed to assist users in debugging
# and troubleshooting log data effectively.
#
# Copyright (c) 2025 Niloy Debnath
# -----------------------------------------------------------------------------


import os
from openai import OpenAI
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import gradio as gr

load_dotenv()
client = OpenAI()

class LogMonitor(FileSystemEventHandler):
    def __init__(self, log_file_path, on_update_callback):
        self.log_file_path = log_file_path
        self.on_update_callback = on_update_callback

    def on_modified(self, event):
        if event.src_path == self.log_file_path:
            with open(self.log_file_path, "r") as f:
                lines = f.readlines()
            self.on_update_callback(lines)


def preprocess_logs(log_lines):
    processed_logs = [line.strip().lower() for line in log_lines if not line.startswith("Payload:")]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjust chunk size as needed
        chunk_overlap=50,  # Overlap for context continuity
    )
    chunks = text_splitter.create_documents(processed_logs)
    return [chunk.page_content for chunk in chunks]


def create_system_context():
    return (
        "You are a proficient developer who is very good in log analysis tasked with assisting users by answering queries based on log data. "
        "Your primary task is to search through Python log files formatted as follows: '%(asctime)s [%(process)d] [%(levelname)s] [%(tool_id)s] [%(module)s:%(funcName)s:%(lineno)s] %(message)'. "
        "Focus on retrieving log entries that match the specific tool_id: 'PR.<org>.<repo>.<pr_number>', replace <org>, <repo>, <pr_number> with the values user provides to create a complete tool_id.\n"
        "Guidelines:\n"
        "1. Ensure that only entries with the specified tool_id are returned.\n"
        "2. If any of the element - org (GitHub organization), repo (GitHub repository), or pr_number (GitHub pull request number) is missing, use previously provided information or prompt the user to provide the missing information.\n"
        "3. Exclude any lines starting with 'Payload:' and omit the first dictionary following such lines, as they contain irrelevant GitHub event data.\n"
        "4. Provide the user with the most relevant log entries based on the query."
    )


class LangChainFAISSHandler:
    def __init__(self):
        self.embedding = OpenAIEmbeddings()
        self.vector_store = None  # Initialize empty vector store
        self.previous_tool_id = None
        self.retrieved_logs = []  # Store retrieved logs temporarily

    def initialize_vector_store(self, logs):
        processed_logs = preprocess_logs(logs)
        if processed_logs:
            self.vector_store = FAISS.from_texts(
                texts=processed_logs,
                embedding=self.embedding,
            )

    def add_logs(self, logs):
        processed_logs = preprocess_logs(logs)
        if processed_logs:
            if self.vector_store:
                self.vector_store.add_texts(processed_logs)
            else:
                self.initialize_vector_store(processed_logs)

    def extract_tool_id_details(self, user_query):
        org = None
        repo = None
        pr_number = None

        if "org:" in user_query:
            org = user_query.split("org:")[1].split()[0]
        if "repo:" in user_query:
            repo = user_query.split("repo:")[1].split()[0]
        if "pr_number:" in user_query:
            pr_number = user_query.split("pr_number:")[1].split()[0]

        return org, repo, pr_number

    def get_or_prompt_tool_id(self, user_query):
        org, repo, pr_number = self.extract_tool_id_details(user_query)

        if not org or not repo or not pr_number:
            if self.previous_tool_id:
                org = org or self.previous_tool_id.get("org")
                repo = repo or self.previous_tool_id.get("repo")
                pr_number = pr_number or self.previous_tool_id.get("pr_number")

        if not org or not repo or not pr_number:
            missing_fields = []
            if not org:
                missing_fields.append("organization (org)")
            if not repo:
                missing_fields.append("repository (repo)")
            if not pr_number:
                missing_fields.append("pull request number (pr_number)")

            return None, f"Please provide the following information to proceed: {', '.join(missing_fields)}."

        self.previous_tool_id = {"org": org, "repo": repo, "pr_number": pr_number}
        return f"PR.{org}.{repo}.{pr_number}", None

    def retrieve_logs_paginated(self, tool_id, batch_size=100):
        if not self.vector_store:
            return "No logs have been indexed yet. Please ensure logs are uploaded and processed."

        retrieved_docs = self.vector_store.similarity_search(query=tool_id, k=1000)
        self.retrieved_logs = [doc.page_content for doc in retrieved_docs]

        if not self.retrieved_logs:
            return (
                f"No relevant logs were found for the tool_id '{tool_id}'. "
                "Please ensure the details are correct, or check if the logs are uploaded."
            )

        paginated_logs = [self.retrieved_logs[i:i + batch_size] for i in range(0, len(self.retrieved_logs), batch_size)]
        return paginated_logs

    def query_logs(self, conversation):
        user_query = conversation[-1]["content"].lower()
        tool_id, prompt_message = self.get_or_prompt_tool_id(user_query)

        if prompt_message:
            return prompt_message

        # If logs are not yet retrieved, retrieve them first
        if not self.retrieved_logs:
            paginated_logs = self.retrieve_logs_paginated(tool_id)
            if isinstance(paginated_logs, str):
                return paginated_logs

            # Summarize or process the logs in manageable chunks
            summaries = []
            for i, batch in enumerate(paginated_logs):
                context = "\n".join(batch)
                conversation[0]["content"] = f"Relevant logs (Batch {i + 1}):\n{context}"
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=conversation,
                        temperature=0.2
                    )
                    summaries.append(response.choices[0].message.content.strip())
                except Exception as e:
                    return f"Error querying OpenAI: {str(e)}"

            # Combine all summaries and provide a single response
            self.retrieved_logs = []  # Clear logs after summarization
            return "\n\n".join(summaries)

        else:
            # Use retrieved logs to answer the user's specific query
            context = "\n".join(self.retrieved_logs)
            conversation[0]["content"] += f"\n\nRelevant logs:\n{context}"

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=conversation,
                    temperature=0.2
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error querying OpenAI: {str(e)}"


def setup_log_monitor(log_file_path, faiss_handler):
    def on_log_update(new_logs):
        faiss_handler.add_logs(new_logs)

    log_monitor = LogMonitor(log_file_path, on_log_update)
    observer = Observer()
    observer.schedule(log_monitor, path=os.path.dirname(log_file_path), recursive=False)
    observer.start()
    return observer


def chat(message, history):
    log_file_path = r"C:\Users\Niloy Debnath\code-workspace\log-analyzer\logs\sample.log"

    if not hasattr(chat, "faiss_handler"):
        chat.faiss_handler = LangChainFAISSHandler()
        with open(log_file_path, "r") as log_file:
            initial_logs = log_file.readlines()
        chat.faiss_handler.initialize_vector_store(initial_logs)
        chat.observer = setup_log_monitor(log_file_path, chat.faiss_handler)

    conversation = (
        [{"role": "system", "content": create_system_context()}]
        + history
        + [{"role": "user", "content": message}]
    )

    response = chat.faiss_handler.query_logs(conversation)
    return response

FAISS_HANDLER = LangChainFAISSHandler()
chat_ui = gr.ChatInterface(fn=chat, type="messages")
chat_ui.launch()
