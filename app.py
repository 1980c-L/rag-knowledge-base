"""
RAG 私有知识库 — 上传文档 → 向量化 → 智能问答
"""
import streamlit as st
import openai
import os
import json
import hashlib
import re
import time
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════
#  供应商配置
# ═══════════════════════════════════════════════════════
PROVIDERS = {
    "智谱 GLM": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "env_key": "ZHIPU_API_KEY",
        "chat_model": "GLM-4V-Flash",
        "embed_model": "embedding-3",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "chat_model": "deepseek-chat",
        "embed_model": None,  # DeepSeek 没有公开 embedding
    },
    "通义千问": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "chat_model": "qwen-plus",
        "embed_model": "text-embedding-v3",
    },
}

DB_DIR = Path(__file__).parent / "knowledge_bases"
DB_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════
#  页面配置
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="RAG 知识库",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
    footer, #MainMenu { visibility: hidden; }
    .stApp { background: linear-gradient(160deg, #080810, #0d0d20, #101028) !important; }
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-thumb { background: #2a2a45; border-radius: 3px; }
    h1 { font-size: 1.6rem; font-weight: 700; color: #e4e4e7; }
    h2 { color: #d0d0dc; font-size: 1.15rem; }
    .card {
        background: #14142b; border: 1px solid #1e1e3a;
        border-radius: 18px; padding: 20px; margin-bottom: 14px;
        box-shadow: 0 2px 16px rgba(0,0,0,0.2);
        animation: fadeIn 0.4s ease-out;
    }
    @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
    [data-testid="stChatMessage"] {
        border-radius: 16px !important; padding: 14px 18px !important;
        margin-bottom: 14px !important; background: #14142b !important;
        border: 1px solid #1e1e3a !important;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important; border: 1.5px solid #252545 !important;
        padding: 14px 18px !important; background: #12122b !important; color: #e8e8f0 !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(175deg, #0a0a18, #0f0f24) !important;
        border-right: 1px solid #1c1c35 !important;
    }
    .stButton>button { border-radius: 10px; transition: all 0.2s; }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #252545 !important; border-radius: 14px !important;
        background: rgba(99,102,241,0.03) !important;
    }
    [data-testid="stFileUploadDropzone"]:hover { border-color: #818cf8 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  文档解析
# ═══════════════════════════════════════════════════════
def parse_document(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(file_bytes))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    elif ext in (".docx", ".doc"):
        from docx import Document
        doc = Document(BytesIO(file_bytes))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext in (".txt", ".md"):
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"不支持的文件格式：{ext}")


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if len(c.strip()) > 30]


def embed_chunks(chunks: list, api_key: str, base_url: str, model: str) -> list:
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=30)
    vectors = []
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend([d.embedding for d in resp.data])
        time.sleep(0.3)
    return vectors


# ═══════════════════════════════════════════════════════
#  知识库管理
# ═══════════════════════════════════════════════════════
class KnowledgeBase:
    def __init__(self, name: str):
        self.name = name
        self.dir = DB_DIR / name
        self.dir.mkdir(exist_ok=True)
        self.index_file = self.dir / "index.json"

    def add(self, filename: str, file_bytes: bytes, api_key: str, base_url: str, embed_model: str):
        text = parse_document(file_bytes, filename)
        chunks = chunk_text(text)
        vectors = embed_chunks(chunks, api_key, base_url, embed_model)

        # 保存索引
        index = self._load_index()
        doc_id = hashlib.md5(f"{filename}{len(index)}".encode()).hexdigest()[:8]
        index[doc_id] = {
            "filename": filename,
            "chunks": chunks,
            "vectors": vectors,
            "chunk_count": len(chunks),
        }
        self._save_index(index)
        return len(chunks)

    def remove(self, doc_id: str):
        index = self._load_index()
        index.pop(doc_id, None)
        self._save_index(index)

    def search(self, query: str, api_key: str, base_url: str, embed_model: str, top_k: int = 5) -> list:
        import numpy as np
        index = self._load_index()
        if not index:
            return []

        # 收集所有 chunks 和 vectors
        all_chunks = []
        all_vectors = []
        for doc in index.values():
            all_chunks.extend(doc["chunks"])
            all_vectors.extend(doc["vectors"])

        if not all_vectors:
            return []

        # 查询向量
        qv = embed_chunks([query], api_key, base_url, embed_model)[0]
        qv_arr = np.array(qv, dtype="float32")

        # 余弦相似度
        scores = []
        for i, v in enumerate(all_vectors):
            v_arr = np.array(v, dtype="float32")
            sim = np.dot(qv_arr, v_arr) / (np.linalg.norm(qv_arr) * np.linalg.norm(v_arr) + 1e-8)
            if sim > 0.5:
                scores.append((sim, all_chunks[i]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [{"content": c, "score": round(s, 3)} for s, c in scores[:top_k]]

    def list_docs(self) -> list:
        return [
            {"id": did, "filename": doc["filename"], "chunks": doc["chunk_count"]}
            for did, doc in self._load_index().items()
        ]

    def stats(self) -> dict:
        index = self._load_index()
        return {
            "doc_count": len(index),
            "chunk_count": sum(d["chunk_count"] for d in index.values()),
        }

    def _load_index(self) -> dict:
        if self.index_file.exists():
            return json.loads(self.index_file.read_text())
        return {}

    def _save_index(self, index: dict):
        self.index_file.write_text(json.dumps(index, ensure_ascii=False))

    @staticmethod
    def list_all() -> list:
        return [d.name for d in DB_DIR.iterdir() if d.is_dir() and (d / "index.json").exists()]


# ═══════════════════════════════════════════════════════
#  侧边栏
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📚 知识库")

    provider = st.selectbox("AI 供应商", list(PROVIDERS.keys()), index=0)
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["env_key"], "")

    if api_key:
        st.success(f"Key: {api_key[:8]}…{api_key[-4:]}")
    else:
        st.error(f"未配置 {cfg['env_key']}")
        st.stop()

    st.divider()

    # 知识库选择
    existing = KnowledgeBase.list_all()
    kb_names = ["默认知识库"] + existing
    kb_name = st.selectbox("选择知识库", kb_names, index=0)
    if kb_name == "默认知识库":
        kb_name = "default"
    kb = KnowledgeBase(kb_name)

    if api_key:
        st.divider()
        st.caption("📄 上传文档（PDF/TXT/MD/DOCX）")
        uploaded = st.file_uploader(
            "上传", type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            key="file_upload",
            label_visibility="collapsed",
        )
        if uploaded:
            with st.spinner("解析 + 向量化…"):
                total = 0
                bar = st.progress(0)
                for i, f in enumerate(uploaded):
                    n = kb.add(f.name, f.read(), api_key, cfg["base_url"], cfg["embed_model"])
                    total += n
                    bar.progress((i + 1) / len(uploaded))
                bar.empty()
            st.success(f"✅ 已添加 {len(uploaded)} 个文档，{total} 个片段")
            st.rerun()

    # 文档列表
    docs = kb.list_docs()
    if docs:
        st.divider()
        st.caption(f"📋 {len(docs)} 个文档 · {sum(d['chunks'] for d in docs)} 个片段")
        for d in docs:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.caption(f"📄 {d['filename']} ({d['chunks']}块)")
            with col2:
                if st.button("🗑", key=f"del_{d['id']}", help="删除"):
                    kb.remove(d['id'])
                    st.rerun()

# ═══════════════════════════════════════════════════════
#  主界面
# ═══════════════════════════════════════════════════════
st.markdown("<h1>📚 RAG 私有知识库</h1>", unsafe_allow_html=True)

stats = kb.stats()
col_a, col_b = st.columns(2)
with col_a:
    st.metric("文档数", stats["doc_count"])
with col_b:
    st.metric("向量片段", stats["chunk_count"])

if stats["doc_count"] == 0:
    st.info("👈 左侧上传 PDF/TXT/DOCX 开始构建知识库")
    st.stop()

# ═══════════════════════════════════════════════════════
#  对话
# ═══════════════════════════════════════════════════════
if "kb_messages" not in st.session_state:
    st.session_state.kb_messages = []

for msg in st.session_state.kb_messages:
    with st.chat_message(msg["role"], avatar=msg["avatar"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("向知识库提问…"):
    st.session_state.kb_messages.append({
        "role": "user", "content": prompt, "avatar": "👤",
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("🔍 检索中…")

        # 检索
        results = kb.search(prompt, api_key, cfg["base_url"], cfg["embed_model"])
        if results:
            context = "\n\n---\n\n".join(
                f"[{r['score']:.2f}] {r['content'][:800]}"
                for r in results
            )
        else:
            context = "（知识库中未找到相关内容）"

        # 生成回答
        full_response = ""
        try:
            client = openai.OpenAI(api_key=api_key, base_url=cfg["base_url"], timeout=30)
            stream = client.chat.completions.create(
                model=cfg["chat_model"],
                messages=[{
                    "role": "system",
                    "content": f"你是一个知识库助手。严格基于以下文档内容回答，不要编造。\n\n## 参考文档\n{context}",
                }, {
                    "role": "user", "content": prompt,
                }],
                temperature=0.3, max_tokens=1024, stream=True,
            )
            for chunk in stream:
                c = chunk.choices[0].delta.content or ""
                if c:
                    full_response += c
                    placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ 生成失败：{e}"

        if not full_response.strip():
            full_response = "未找到相关答案，请换个问法。"
        placeholder.markdown(full_response)

    st.session_state.kb_messages.append({
        "role": "assistant", "content": full_response, "avatar": "🤖",
    })
    st.rerun()
