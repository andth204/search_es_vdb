# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from langchain_huggingface import HuggingFaceEmbeddings
# import numpy as np
# import faiss
# import pickle
# import requests
# import os
# from typing import List, Dict, Any
# from datetime import datetime
# from pymongo import MongoClient
# from bson import ObjectId

# # === CẤU HÌNH ===
# ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
# INDEX_NAME = "semantic_chunks"
# MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://username:rad123123@cds-cluster.egvkeod.mongodb.net/")
# MONGO_DB_NAME = "cdsdb"

# # FAISS files (mount từ volume chung)
# FAISS_DIR = os.getenv("FAISS_DIR", "/app/faiss_data")
# FAISS_INDEX_PATH = f"{FAISS_DIR}/faiss.index"
# CHUNKS_METADATA_PATH = f"{FAISS_DIR}/chunks_metadata.pkl"
# QA_FAISS_PATH = f"{FAISS_DIR}/faiss_qa.index"
# QA_METADATA_PATH = f"{FAISS_DIR}/qa_metadata.pkl"

# # === KẾT NỐI MONGODB ===
# mongo_client = MongoClient(MONGO_URI)
# db = mongo_client[MONGO_DB_NAME]
# fulltexts_collection = db["fulltexts"]
# qa_collection = db["qa"]

# # === MODEL TIẾNG VIỆT ===
# MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
# embedding_model = HuggingFaceEmbeddings(
#     model_name=MODEL_NAME,
#     model_kwargs={'device': 'cpu', 'trust_remote_code': True},
#     encode_kwargs={'normalize_embeddings': True}
# )

# EMBEDDING_DIM = 768

# app = FastAPI(title="Search API", description="API tìm kiếm keyword, semantic và Q&A")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # === HÀM HELPER ===
# def create_embedding(text: str) -> List[float]:
#     """Tạo embedding từ text"""
#     return embedding_model.embed_query(text)

# def load_metadata(path: str) -> List[Dict[str, Any]]:
#     """Load metadata từ file pkl"""
#     if not os.path.exists(path):
#         return []
#     try:
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     except Exception as e:
#         print(f"[Warning] Lỗi khi load metadata: {e}")
#         return []

# def save_metadata(metadata: List[Dict[str, Any]], path: str) -> None:
#     """Save metadata ra file pkl"""
#     with open(path, "wb") as f:
#         pickle.dump(metadata, f)

# def load_qa_index_and_metadata():
#     """Load FAISS Q&A index và metadata"""
#     if os.path.exists(QA_FAISS_PATH) and os.path.exists(QA_METADATA_PATH):
#         index = faiss.read_index(QA_FAISS_PATH)
#         with open(QA_METADATA_PATH, "rb") as f:
#             metadata = pickle.load(f)
#         return index, metadata
#     return faiss.IndexFlatIP(EMBEDDING_DIM), []

# def save_qa_index_and_metadata(index, metadata):
#     """Save FAISS Q&A index và metadata"""
#     faiss.write_index(index, QA_FAISS_PATH)
#     with open(QA_METADATA_PATH, "wb") as f:
#         pickle.dump(metadata, f)

# def save_qa_from_search(question: str, answer: str, fulltexts_id: ObjectId, type_qa: str):
#     """
#     Lưu cặp Q&A vào MongoDB và FAISS từ kết quả search
    
#     Args:
#         question: Câu hỏi (query của user)
#         answer: Câu trả lời (content của chunk)
#         fulltexts_id: ObjectId của fulltext trong MongoDB
#         type_qa: Loại search ('keyword' hoặc 'semantic')
#     """
#     try:
#         # Lưu vào MongoDB
#         qa_doc = {
#             "question": question,
#             "answer": answer,
#             "fulltexts_id": fulltexts_id,
#             "type_qa": type_qa,
#             "create_at": datetime.utcnow()
#         }
#         result = qa_collection.insert_one(qa_doc)
#         print(f"[Q&A] Saved to MongoDB: {result.inserted_id}")

#         # Cập nhật FAISS Q&A
#         index, metadata = load_qa_index_and_metadata()
#         q_emb = create_embedding(question)
#         q_emb = np.array([q_emb]).astype('float32')
#         faiss.normalize_L2(q_emb)
#         index.add(q_emb)
#         metadata.append({
#             "question": question,
#             "answer": answer,
#             "fulltexts_id": str(fulltexts_id),
#             "type_qa": type_qa,
#             "qa_id": str(result.inserted_id)
#         })
#         save_qa_index_and_metadata(index, metadata)
#         print(f"[Q&A] Updated FAISS Q&A index, total: {index.ntotal}")
#     except Exception as e:
#         print(f"[Warning] Không lưu được Q&A: {e}")

# # === PYDANTIC MODELS ===
# class SearchKeywordRequest(BaseModel):
#     query: str
#     size: int = 10

# class SearchSemanticRequest(BaseModel):
#     query: str
#     k: int = 5

# class SearchQARequest(BaseModel):
#     question: str
#     k: int = 1

# # ========================================
# # API 1: TÌM KIẾM THEO TỪ KHÓA
# # ========================================
# @app.post("/search-keyword")
# def search_keyword(req: SearchKeywordRequest):
#     """
#     Tìm kiếm theo từ khóa trong Elasticsearch
#     Tự động lưu Q&A với type_qa = 'keyword'
#     """
#     search_url = f"{ELASTICSEARCH_URL}/{INDEX_NAME}/_search"
#     search_body = {
#         "query": {"match": {"content": req.query}},
#         "size": req.size
#     }
    
#     try:
#         res = requests.post(
#             search_url, 
#             json=search_body, 
#             headers={"Content-Type": "application/json"}, 
#             timeout=10
#         )
#         if res.status_code != 200:
#             raise HTTPException(
#                 status_code=500, 
#                 detail=f"Lỗi khi tìm kiếm trong Elasticsearch: {res.text}"
#             )
#     except requests.exceptions.RequestException as e:
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Lỗi kết nối Elasticsearch: {str(e)}"
#         )
    
#     results = res.json()
#     hits = []
#     for hit in results["hits"]["hits"]:
#         source = hit["_source"]
#         hits.append({
#             "chunk_id": source["chunk_id"],
#             "document_id": source["document_id"],
#             "content": source["content"],
#             "source_url": source["source_url"],
#             "fulltexts_id": source.get("fulltexts_id"),
#             "score": hit["_score"]
#         })

#     # Tự động lưu Q&A từ kết quả đầu tiên
#     if hits:
#         top = hits[0]
#         if top.get("fulltexts_id"):
#             save_qa_from_search(
#                 question=req.query,
#                 answer=top["content"],
#                 fulltexts_id=ObjectId(top["fulltexts_id"]),
#                 type_qa="keyword"
#             )

#     return {
#         "query": req.query,
#         "total_hits": len(hits),
#         "results": hits
#     }

# # ========================================
# # API 2: TÌM KIẾM THEO NGỮ NGHĨA
# # ========================================
# @app.post("/search-semantic")
# def search_semantic(req: SearchSemanticRequest):
#     """
#     Tìm kiếm theo ngữ nghĩa sử dụng FAISS
#     Tự động lưu Q&A với type_qa = 'semantic'
#     """
#     if not os.path.exists(FAISS_INDEX_PATH):
#         raise HTTPException(
#             status_code=404, 
#             detail="FAISS index chưa được tạo. Vui lòng chạy process-document trước."
#         )

#     # Tạo embedding cho query
#     query_embedding = create_embedding(req.query)
#     query_embedding = np.array([query_embedding]).astype('float32')
#     faiss.normalize_L2(query_embedding)
    
#     # Search trong FAISS
#     faiss_index = faiss.read_index(FAISS_INDEX_PATH)
#     D, I = faiss_index.search(query_embedding, k=req.k)
    
#     # Load metadata để lấy thông tin chunks
#     saved_chunks = load_metadata(CHUNKS_METADATA_PATH)
    
#     hits = []
#     for i in range(len(I[0])):
#         idx = I[0][i]
#         if idx != -1 and idx < len(saved_chunks):
#             chunk_data = saved_chunks[idx]
#             hits.append({
#                 "rank": i + 1,
#                 "score": float(D[0][i]),
#                 "document_id": chunk_data["document_id"],
#                 "content": chunk_data["chunk"],
#                 "source_url": chunk_data["source_url"],
#                 "fulltexts_id": chunk_data.get("fulltexts_id")
#             })

#     # Tự động lưu Q&A từ kết quả đầu tiên
#     if hits:
#         top_hit = hits[0]
#         fulltexts_id_str = top_hit.get("fulltexts_id")
#         if fulltexts_id_str:
#             save_qa_from_search(
#                 question=req.query,
#                 answer=top_hit["content"],
#                 fulltexts_id=ObjectId(fulltexts_id_str),
#                 type_qa="semantic"
#             )

#     return {
#         "query": req.query,
#         "total_hits": len(hits),
#         "results": hits
#     }

# # ========================================
# # API 3: TÌM CÂU TRẢ LỜI TỪ Q&A ĐÃ HỌC
# # ========================================
# @app.post("/search-answer")
# def search_answer(req: SearchQARequest):
#     """
#     Tìm câu trả lời từ knowledge base Q&A đã học
#     Sử dụng FAISS để tìm câu hỏi tương tự nhất
#     """
#     index, metadata = load_qa_index_and_metadata()
    
#     if len(metadata) == 0:
#         raise HTTPException(
#             status_code=404, 
#             detail="Chưa có cặp Q&A nào trong knowledge base."
#         )

#     # Tạo embedding cho câu hỏi
#     q_emb = create_embedding(req.question)
#     q_emb = np.array([q_emb]).astype('float32')
#     faiss.normalize_L2(q_emb)
    
#     # Search trong FAISS Q&A
#     D, I = index.search(q_emb, k=min(req.k, len(metadata)))
    
#     results = []
#     for i in range(len(I[0])):
#         idx = I[0][i]
#         if idx != -1 and idx < len(metadata):
#             item = metadata[idx]
#             results.append({
#                 "question_matched": item["question"],
#                 "answer": item["answer"],
#                 "fulltexts_id": item["fulltexts_id"],
#                 "type_qa": item.get("type_qa", "unknown"),
#                 "similarity_score": float(D[0][i])
#             })
    
#     return {
#         "input_question": req.question,
#         "results": results
#     }

# # ========================================
# # API: HEALTH CHECK
# # ========================================
# @app.get("/health")
# def health_check():
#     """Kiểm tra trạng thái hệ thống"""
#     # Check FAISS chunks
#     faiss_status = "ready" if os.path.exists(FAISS_INDEX_PATH) else "not_initialized"
#     faiss_size = 0
#     if os.path.exists(FAISS_INDEX_PATH):
#         try:
#             faiss_index = faiss.read_index(FAISS_INDEX_PATH)
#             faiss_size = faiss_index.ntotal
#         except Exception as e:
#             faiss_status = f"error: {str(e)}"
    
#     # Check Elasticsearch
#     es_status = "unknown"
#     try:
#         res = requests.get(f"{ELASTICSEARCH_URL}/_cluster/health", timeout=5)
#         if res.status_code == 200:
#             es_status = res.json()["status"]
#     except Exception:
#         es_status = "unreachable"

#     # Check Q&A
#     qa_index, qa_meta = load_qa_index_and_metadata()
    
#     return {
#         "status": "healthy",
#         "service": "search_api",
#         "elasticsearch": {
#             "url": ELASTICSEARCH_URL, 
#             "status": es_status, 
#             "index": INDEX_NAME
#         },
#         "mongodb": {
#             "uri": MONGO_URI,
#             "database": MONGO_DB_NAME
#         },
#         "faiss_chunks": {
#             "status": faiss_status,
#             "total_vectors": faiss_size,
#             "path": FAISS_INDEX_PATH
#         },
#         "faiss_qa": {
#             "total_pairs": len(qa_meta),
#             "path": QA_FAISS_PATH
#         },
#         "model": MODEL_NAME,
#         "embedding_dimension": EMBEDDING_DIM
#     }

# # ========================================
# # API: ROOT
# # ========================================
# @app.get("/")
# def root():
#     return {
#         "service": "Search API",
#         "version": "1.0.0",
#         "endpoints": {
#             "search_keyword": "POST /search-keyword",
#             "search_semantic": "POST /search-semantic",
#             "search_answer": "POST /search-answer",
#             "health": "GET /health"
#         }
#     }


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import faiss
import pickle
import requests
import os
from typing import List, Dict, Any
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId

# === CẤU HÌNH ===
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
INDEX_NAME = "semantic_chunks"
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://username:rad123123@cds-cluster.egvkeod.mongodb.net/")
MONGO_DB_NAME = "cdsdb"

# FAISS files (mount từ volume chung)
FAISS_DIR = os.getenv("FAISS_DIR", "/app/faiss_data")
FAISS_INDEX_PATH = f"{FAISS_DIR}/faiss.index"
CHUNKS_METADATA_PATH = f"{FAISS_DIR}/chunks_metadata.pkl"
QA_FAISS_PATH = f"{FAISS_DIR}/faiss_qa.index"
QA_METADATA_PATH = f"{FAISS_DIR}/qa_metadata.pkl"

# === KẾT NỐI MONGODB ===
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
fulltexts_collection = db["fulltexts"]
qa_collection = db["qa"]

# === MODEL TIẾNG VIỆT ===
MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu', 'trust_remote_code': True},
    encode_kwargs={'normalize_embeddings': True}
)

EMBEDDING_DIM = 768

app = FastAPI(title="Search API", description="API tìm kiếm keyword, semantic và Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === HÀM HELPER ===
def create_embedding(text: str) -> List[float]:
    """Tạo embedding từ text"""
    return embedding_model.embed_query(text)

def load_metadata(path: str) -> List[Dict[str, Any]]:
    """Load metadata từ file pkl"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[Warning] Lỗi khi load meta {e}")
        return []

def save_metadata(metadata: List[Dict[str, Any]], path: str) -> None: # <-- ĐÃ SỬA: Thêm dấu ':'
    """Save metadata ra file pkl"""
    with open(path, "wb") as f:
        pickle.dump(metadata, f) # <-- Biến được sử dụng đúng là 'metadata', không phải 'meta'

def load_qa_index_and_metadata():
    """Load FAISS Q&A index và metadata"""
    if os.path.exists(QA_FAISS_PATH) and os.path.exists(QA_METADATA_PATH):
        index = faiss.read_index(QA_FAISS_PATH)
        with open(QA_METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    return faiss.IndexFlatIP(EMBEDDING_DIM), []

def save_qa_index_and_metadata(index, metadata):
    """Save FAISS Q&A index và metadata"""
    faiss.write_index(index, QA_FAISS_PATH)
    with open(QA_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

def save_qa_from_search(question: str, answers_with_cites: List[Dict[str, Any]], type_qa: str):
    """
    Lưu cặp Q&A vào MongoDB và FAISS từ kết quả search.
    answers_with_cites: Danh sách các dict có dạng:
        {
            "content": "nội dung chunk",
            "cite": {
                "chunk_id": "...",
                "document_id": "...",
                "source_url": "...",
                "score": float (nếu có),
                "fulltexts_id": "..." // <-- fulltexts_id NẰM TRONG cite
            }
        }

    Args:
        question: Câu hỏi (query của user)
        answers_with_cites: Danh sách các answer và cite tương ứng (đã có fulltexts_id)
        type_qa: Loại search ('keyword' hoặc 'semantic')
    """
    try:
        # Lưu vào MongoDB
        # Không cần lấy fulltexts_id từ bên ngoài nữa
        # Lấy fulltexts_id từ chính phần tử đầu tiên trong answers_with_cites nếu cần để tham chiếu chung (tùy chọn)
        # primary_fulltexts_id = answers_with_cites[0]["cite"]["fulltexts_id"] if answers_with_cites else None

        qa_doc = {
            "question": question,
            "answers": answers_with_cites, # <-- Schema mới
            # "fulltexts_id": primary_fulltexts_id, # <-- BỎ DÒNG NÀY
            "type_qa": type_qa,
            "create_at": datetime.utcnow()
        }
        result = qa_collection.insert_one(qa_doc)
        print(f"[Q&A] Saved to MongoDB: {result.inserted_id}")

        # Cập nhật FAISS Q&A (chỉ lưu embedding của question gốc)
        index, metadata = load_qa_index_and_metadata()
        q_emb = create_embedding(question)
        q_emb = np.array([q_emb]).astype('float32')
        faiss.normalize_L2(q_emb)
        index.add(q_emb)
        metadata.append({
            "question": question,
            # "primary_answer": primary_answer_content, # <-- Có thể vẫn giữ nếu cần
            "answers_with_cites": answers_with_cites, # <-- Lưu vào metadata FAISS
            # "fulltexts_id": primary_fulltexts_id, # <-- BỎ DÒNG NÀY
            "type_qa": type_qa,
            "qa_id": str(result.inserted_id)
        })
        save_qa_index_and_metadata(index, metadata)
        print(f"[Q&A] Updated FAISS Q&A index, total: {index.ntotal}")
    except Exception as e:
        print(f"[Warning] Không lưu được Q&A: {e}")

# === PYDANTIC MODELS ===
class SearchKeywordRequest(BaseModel):
    query: str
    size: int = 10

class SearchSemanticRequest(BaseModel):
    query: str
    k: int = 5
    # top_k_to_save: int = 3 # <-- BỎ DÒNG NÀY

class SearchQARequest(BaseModel):
    question: str
    k: int = 1

# ========================================
# API 1: TÌM KIẾM THEO TỪ KHÓA
# ========================================
@app.post("/search-keyword")
def search_keyword(req: SearchKeywordRequest):
    """
    Tìm kiếm theo từ khóa trong Elasticsearch
    Tự động lưu Q&A với type_qa = 'keyword', lưu nhiều chunk vào answers/cites.
    """
    search_url = f"{ELASTICSEARCH_URL}/{INDEX_NAME}/_search"
    search_body = {
        "query": {"match": {"content": req.query}},
        "size": req.size
    }

    try:
        res = requests.post(
            search_url,
            json=search_body,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if res.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Lỗi khi tìm kiếm trong Elasticsearch: {res.text}"
            )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi kết nối Elasticsearch: {str(e)}"
        )

    results = res.json()
    hits = []
    for hit in results["hits"]["hits"]:
        source = hit["_source"]
        hits.append({
            "chunk_id": source["chunk_id"],
            "document_id": source["document_id"],
            "content": source["content"],
            "source_url": source["source_url"],
            "fulltexts_id": source.get("fulltexts_id"),
            "score": hit["_score"]
        })

    # --- PHẦN MỚI: Tự động lưu Q&A từ nhiều kết quả ---
    if hits:
        # Chọn top-k hits để lưu vào Q&A (có thể là toàn bộ nếu số lượng nhỏ hơn size)
        top_k_hits_to_save = hits[:req.size] # hoặc bạn có thể thêm một tham số mới nếu muốn

        answers_with_cites = []
        for hit in top_k_hits_to_save:
            answers_with_cites.append({
                "content": hit["content"],
                "cite": {
                    "chunk_id": hit["chunk_id"],
                    "document_id": hit["document_id"],
                    "source_url": hit["source_url"],
                    "score": hit["score"], # điểm relevance từ ES
                    "fulltexts_id": hit["fulltexts_id"] # <-- fulltexts_id vào cite
                }
            })

        if answers_with_cites:
            save_qa_from_search(
                question=req.query,
                answers_with_cites=answers_with_cites,
                type_qa="keyword"
            )

    return {
        "query": req.query,
        "total_hits": len(hits),
        "results": hits
    }

# ========================================
# API 2: TÌM KIẾM THEO NGỮ NGHĨA
# ========================================
@app.post("/search-semantic")
def search_semantic(req: SearchSemanticRequest):
    """
    Tìm kiếm theo ngữ nghĩa sử dụng FAISS
    Tự động lưu Q&A với type_qa = 'semantic', lưu nhiều chunk vào answers/cites.
    Bổ sung chunk_id vào kết quả.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise HTTPException(
            status_code=404,
            detail="FAISS index chưa được tạo. Vui lòng chạy process-document trước."
        )

    # Tạo embedding cho query
    query_embedding = create_embedding(req.query)
    query_embedding = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_embedding)

    # Search trong FAISS
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    D, I = faiss_index.search(query_embedding, k=req.k) # <-- Sử dụng req.k

    # Load metadata để lấy thông tin chunks
    saved_chunks = load_metadata(CHUNKS_METADATA_PATH)

    hits = []
    for i in range(len(I[0])):
        idx = I[0][i]
        if idx != -1 and idx < len(saved_chunks):
            chunk_data = saved_chunks[idx]
            # --- BỔ SUNG chunk_id ---
            chunk_id = chunk_data.get("chunk_id", f"faiss_idx_{idx}") # Fallback nếu không có
            hits.append({
                "rank": i + 1,
                "score": float(D[0][i]), # cosine similarity score từ FAISS
                "chunk_id": chunk_id, # <-- Bổ sung chunk_id
                "document_id": chunk_data["document_id"],
                "content": chunk_data["chunk"],
                "source_url": chunk_data["source_url"],
                "fulltexts_id": chunk_data.get("fulltexts_id")
            })

    # --- PHẦN MỚI: Tự động lưu Q&A từ nhiều kết quả ---
    if hits:
        # Chọn top-k hits để lưu vào Q&A (số lượng do req.k quy định - bỏ top_k_to_save)
        top_k_hits_to_save = hits[:req.k] # <-- Thay vì req.top_k_to_save

        answers_with_cites = []
        for hit in top_k_hits_to_save:
            answers_with_cites.append({
                "content": hit["content"],
                "cite": {
                    "chunk_id": hit["chunk_id"],
                    "document_id": hit["document_id"],
                    "source_url": hit["source_url"],
                    "score": hit["score"], # điểm similarity từ FAISS
                    "fulltexts_id": hit["fulltexts_id"] # <-- fulltexts_id vào cite
                }
            })

        if answers_with_cites:
            save_qa_from_search(
                question=req.query,
                answers_with_cites=answers_with_cites,
                type_qa="semantic"
            )

    return {
        "query": req.query,
        "total_hits": len(hits),
        "results": hits
    }

# ========================================
# API 3: TÌM CÂU TRẢ LỜI TỪ Q&A ĐÃ HỌC
# ========================================
@app.post("/search-answer")
def search_answer(req: SearchQARequest):
    """
    Tìm câu trả lời từ knowledge base Q&A đã học
    Sử dụng FAISS để tìm câu hỏi tương tự nhất
    Trả về theo schema mới: mảng answers và cite tương ứng.
    """
    index, metadata = load_qa_index_and_metadata()

    if len(metadata) == 0:
        raise HTTPException(
            status_code=404,
            detail="Chưa có cặp Q&A nào trong knowledge base."
        )

    # Tạo embedding cho câu hỏi
    q_emb = create_embedding(req.question)
    q_emb = np.array([q_emb]).astype('float32')
    faiss.normalize_L2(q_emb)

    # Search trong FAISS Q&A
    D, I = index.search(q_emb, k=min(req.k, len(metadata)))

    results = []
    for i in range(len(I[0])):
        idx = I[0][i]
        if idx != -1 and idx < len(metadata):
            item = metadata[idx]
            # --- TRẢ VỀ THEO SCHEMA MỚI ---
            results.append({
                "question_matched": item["question"],
                "answers": item.get("answers_with_cites", []), # <-- Trả về mảng answers và cites
                # "fulltexts_id": item["fulltexts_id"], # <-- BỎ DÒNG NÀY
                "type_qa": item.get("type_qa", "unknown"),
                "similarity_score": float(D[0][i])
            })

    return {
        "input_question": req.question,
        "results": results
    }

# ========================================
# API: HEALTH CHECK
# ========================================
@app.get("/health")
def health_check():
    """Kiểm tra trạng thái hệ thống"""
    # Check FAISS chunks
    faiss_status = "ready" if os.path.exists(FAISS_INDEX_PATH) else "not_initialized"
    faiss_size = 0
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            faiss_size = faiss_index.ntotal
        except Exception as e:
            faiss_status = f"error: {str(e)}"

    # Check Elasticsearch
    es_status = "unknown"
    try:
        res = requests.get(f"{ELASTICSEARCH_URL}/_cluster/health", timeout=5)
        if res.status_code == 200:
            es_status = res.json()["status"]
    except Exception:
        es_status = "unreachable"

    # Check Q&A
    qa_index, qa_meta = load_qa_index_and_metadata()

    return {
        "status": "healthy",
        "service": "search_api",
        "elasticsearch": {
            "url": ELASTICSEARCH_URL,
            "status": es_status,
            "index": INDEX_NAME
        },
        "mongodb": {
            "uri": MONGO_URI,
            "database": MONGO_DB_NAME
        },
        "faiss_chunks": {
            "status": faiss_status,
            "total_vectors": faiss_size,
            "path": FAISS_INDEX_PATH
        },
        "faiss_qa": {
            "total_pairs": len(qa_meta),
            "path": QA_FAISS_PATH
        },
        "model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIM
    }

# ========================================
# API: ROOT
# ========================================
@app.get("/")
def root():
    return {
        "service": "Search API",
        "version": "1.0.0",
        "endpoints": {
            "search_keyword": "POST /search-keyword",
            "search_semantic": "POST /search-semantic",
            "search_answer": "POST /search-answer",
            "health": "GET /health"
        }
    }