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
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://shared_elasticsearch:9200")
INDEX_NAME = "semantic_chunks"
MONGO_URI = os.getenv("MONGO_URI")
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

def save_metadata(metadata: List[Dict[str, Any]], path: str) -> None: 
    """Save metadata ra file pkl"""
    with open(path, "wb") as f:
        pickle.dump(metadata, f) 

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
    try:
        qa_doc = {
            "question": question,
            "answers": answers_with_cites,
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
            "answers_with_cites": answers_with_cites,
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
    k: int = 3

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
        
        # ✅ Xử lý an toàn cho cả data cũ và mới
        document_id = source.get("document_id", "unknown")
        chunk_id = source.get("chunk_id", hit.get("_id", f"legacy_{document_id}"))
        
        hits.append({
            "chunk_id": chunk_id,
            "document_id": document_id,  
            "content": source["content"],
            "source_url": source["source_url"],
            "fulltexts_id": source.get("fulltexts_id"),
            "score": hit["_score"]
        })

    if hits:
        top_k_hits_to_save = hits[:req.size] 

        answers_with_cites = []
        for hit in top_k_hits_to_save:
            answers_with_cites.append({
                "content": hit["content"],
                "cite": {
                    "chunk_id": hit["chunk_id"],
                    "document_id": hit["document_id"],
                    "source_url": hit["source_url"],
                    "score": hit["score"], # điểm relevance từ ES
                    "fulltexts_id": hit["fulltexts_id"] 
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
    D, I = faiss_index.search(query_embedding, k=req.k) 

    # Load metadata để lấy thông tin chunks
    saved_chunks = load_metadata(CHUNKS_METADATA_PATH)

    hits = []
    for i in range(len(I[0])):
        idx = I[0][i]
        if idx != -1 and idx < len(saved_chunks):
            chunk_data = saved_chunks[idx]
            # --- BỔ SUNG chunk_id ---
            chunk_id = chunk_data.get("chunk_id", f"faiss_idx_{idx}") 
            hits.append({
                "rank": i + 1,
                "score": float(D[0][i]), 
                "chunk_id": chunk_id, 
                "document_id": chunk_data["document_id"],
                "content": chunk_data["chunk"],
                "source_url": chunk_data["source_url"],
                "fulltexts_id": chunk_data.get("fulltexts_id")
            })

    if hits:
        top_k_hits_to_save = hits[:req.k] 

        answers_with_cites = []
        for hit in top_k_hits_to_save:
            answers_with_cites.append({
                "content": hit["content"],
                "cite": {
                    "chunk_id": hit["chunk_id"],
                    "document_id": hit["document_id"],
                    "source_url": hit["source_url"],
                    "score": hit["score"], # điểm similarity từ FAISS
                    "fulltexts_id": hit["fulltexts_id"] 
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
            results.append({
                "question_matched": item["question"],
                "answers": item.get("answers_with_cites", []), # <-- Trả về mảng answers và cites
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