import os

import chromadb
from chromadb.utils import embedding_functions
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class Vector_Data_Base():
    def __init__(self,
                 path_pdf = 'manual.pdf',
                 path_db = "./data/chroma_db", 
                 collection_name = 'manual_doc'):
        
        self.path_pdf = path_pdf
        self.path_db = path_db
        self.collection_name = collection_name

        self.embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='intfloat/multilingual-e5-small',
            device='cpu',
            normalize_embeddings=True
        )

        self.client = chromadb.PersistentClient(path=self.path_db)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_model,
            metadata={'hnsw:space': 'cosine'}
        )

    def prepare_and_load_data(self, flag_rebuild= False):
        
        if flag_rebuild ==False and self.collection.count() > 0:
            print("База данных существует")
            return

        if not os.path.exists(self.path_pdf):
            raise FileNotFoundError(f"Файл {self.path_pdf} не найден!")
        
        pages_data = pymupdf4llm.to_markdown(self.path_pdf, page_chunks=True)
        headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )

            
        text_to_add = []  
        metadatas_to_add = []  
        ids_to_add = []        

        global_id = 0

        for page in pages_data:
                page_num = page["metadata"]["page"] 
                page_text = page["text"]

                header_splits = markdown_splitter.split_text(page_text)

                final_splits = text_splitter.split_documents(header_splits)

                for split in final_splits:
                    
                    breadcrumbs = [split.metadata.get(h) for _, h in headers_to_split_on if split.metadata.get(h)]
                    section_path = " > ".join(breadcrumbs) if breadcrumbs else "Общее"

                    text_for_llm = f"Раздел: {section_path}\n{split.page_content}"

                    meta = {
                        "source_page": page_num,      
                        "section": section_path,
                        "raw_text": split.page_content 
                    }

                    text_to_add.append(text_for_llm)
                    metadatas_to_add.append(meta)
                    ids_to_add.append(f"id_{global_id}")
                    global_id += 1


        batch_size = 50
        for i in range(0, len(text_to_add), batch_size):
                end = min(i + batch_size, len(text_to_add))
                self.collection.add(
                    ids=ids_to_add[i:end],
                    documents=text_to_add[i:end],
                    metadatas=metadatas_to_add[i:end]
                )
            
       
    def search(self, querry, n_results = 2):

         results = self.collection.query(
              query_texts=[querry],
              n_results= n_results
         )

         structured_results = []
         if not results['ids']:
            return []  

         for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]    # text

            score = round(1.0 - distance, 4)

            clean_text = metadata.get("original_text", document)

            structured_results.append({
                "text": clean_text,
                "page": metadata.get("source_page", 0),
                "section": metadata.get("section", "Unknown"),
                "score": score
            })

         return structured_results
    
    def view_data(self, limit = 10):
         data = self.collection.peek(limit=limit)

         for i in range(len(data['ids'])):
            meta = data['metadatas'][i]
            print(f"[{meta.get('source_page')}] {data['documents'][i][:]}...")
    


