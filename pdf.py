import os
from llama_index.core.storage import StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader


def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("Building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
    return index

def load_pdfs(pdf_paths):
    engines = {}
    for pdf_path in pdf_paths:
        pdf_name = os.path.basename(pdf_path).split('.')[0]
        pdf_data = PDFReader().load_data(file=pdf_path)
        engines[pdf_name] = get_index(pdf_data, pdf_name)
    return engines

# List of PDF files
pdf_paths = [os.path.join("data", "Canada.pdf"), os.path.join("data", "Croatia.pdf")]
pdf_engines = load_pdfs(pdf_paths)

