"""멀티유저 RAG 챗봇 — DB user 테이블 로그인 + 사용자별 세션/벡터 분리."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import bcrypt
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"

load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
VECTOR_BATCH_SIZE = 10
RETRIEVER_K = 10
MEMORY_MAX = 50
CHATBOT_TITLE = "재정경제부 RAG 챗봇"

ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)

    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multiusers")


logger = _setup_logging()


# ---------------------------------------------------------------------------
# Environment helpers (st.secrets 우선, 없으면 .env)
# ---------------------------------------------------------------------------
def _secret_or_env(name: str) -> str:
    try:
        if name in st.secrets:
            val = st.secrets[name]
            if val:
                return str(val).strip()
    except Exception:  # noqa: BLE001
        pass
    return os.getenv(name, "").strip()


def _get_env_keys() -> dict[str, str]:
    return {
        "openai": _secret_or_env("OPENAI_API_KEY"),
        "supabase_url": _secret_or_env("SUPABASE_URL"),
        "supabase_anon": _secret_or_env("SUPABASE_ANON_KEY"),
    }


def _missing_keys(keys: dict[str, str]) -> list[str]:
    missing: list[str] = []
    if not keys["openai"]:
        missing.append("OPENAI_API_KEY")
    if not keys["supabase_url"]:
        missing.append("SUPABASE_URL")
    if not keys["supabase_anon"]:
        missing.append("SUPABASE_ANON_KEY")
    return missing


def _get_supabase() -> Client | None:
    keys = _get_env_keys()
    if keys["supabase_url"] and keys["supabase_anon"]:
        return create_client(keys["supabase_url"], keys["supabase_anon"])
    return None


def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _format_memory_block(messages: list[dict[str, str]], max_items: int = MEMORY_MAX) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_rag_messages(
    question: str,
    context: str,
    memory_text: str,
) -> list[SystemMessage | HumanMessage]:
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def _get_llm(openai_key: str) -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_NAME, temperature=0.7, api_key=openai_key)


def _get_embeddings(openai_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=openai_key)


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# User auth (app "user" table — Supabase Auth 미사용)
# ---------------------------------------------------------------------------
def register_user(sb: Client, login_id: str, password: str) -> tuple[bool, str]:
    login_id = login_id.strip()
    if not login_id or not password:
        return False, "아이디와 비밀번호를 입력해 주세요."
    if len(login_id) < 3:
        return False, "아이디는 3자 이상이어야 합니다."
    if len(password) < 6:
        return False, "비밀번호는 6자 이상이어야 합니다."

    existing = sb.table("user").select("id").eq("login_id", login_id).execute()
    if existing.data:
        return False, "이미 사용 중인 아이디입니다."

    pw_hash = _hash_password(password)
    sb.table("user").insert(
        {"login_id": login_id, "password_hash": pw_hash}
    ).execute()
    return True, "회원가입이 완료되었습니다. 로그인해 주세요."


def login_user(sb: Client, login_id: str, password: str) -> tuple[bool, str, str | None]:
    login_id = login_id.strip()
    if not login_id or not password:
        return False, "아이디와 비밀번호를 입력해 주세요.", None

    resp = (
        sb.table("user")
        .select("id, login_id, password_hash")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return False, "아이디 또는 비밀번호가 올바르지 않습니다.", None

    row = rows[0]
    if not _verify_password(password, row["password_hash"]):
        return False, "아이디 또는 비밀번호가 올바르지 않습니다.", None

    return True, "로그인되었습니다.", row["id"]


def _generate_followup_section(llm: ChatOpenAI, user_q: str, answer: str) -> str:
    trimmed = answer[:8000]
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(out, "content", str(out)) or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Follow-up generation failed: %s", exc)
        return ""

    raw = remove_separators(str(raw))
    if not raw.strip():
        return ""
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"


def _generate_session_title(llm: ChatOpenAI, first_q: str, first_a: str) -> str:
    prompt = (
        "다음 첫 질문과 답변을 15자 이내 한국어 제목 한 줄로 요약하세요. "
        "따옴표, 설명, 부가 문장 없이 제목만 출력하세요.\n\n"
        f"[질문]\n{first_q[:500]}\n\n[답변]\n{first_a[:800]}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        title = remove_separators(str(getattr(out, "content", "") or "")).strip()
        if title:
            return title[:80]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Session title generation failed: %s", exc)
    return first_q[:40] or "새 세션"


def _current_user_id() -> str:
    uid = st.session_state.get("current_user_id")
    if not uid:
        raise RuntimeError("로그인이 필요합니다.")
    return uid


# ---------------------------------------------------------------------------
# Supabase session / message CRUD (user_id 필터)
# ---------------------------------------------------------------------------
def fetch_sessions(sb: Client, user_id: str) -> list[dict[str, Any]]:
    resp = (
        sb.table("chat_sessions")
        .select("id, title, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return resp.data or []


def fetch_messages(sb: Client, user_id: str, session_id: str) -> list[dict[str, str]]:
    resp = (
        sb.table("chat_messages")
        .select("role, content")
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .order("message_order")
        .execute()
    )
    return [{"role": r["role"], "content": r["content"]} for r in (resp.data or [])]


def fetch_vector_file_names(sb: Client, user_id: str, session_id: str) -> list[str]:
    resp = (
        sb.table("vector_documents")
        .select("file_name")
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .execute()
    )
    names = sorted({r["file_name"] for r in (resp.data or []) if r.get("file_name")})
    return names


def _upsert_session_row(sb: Client, user_id: str, session_id: str, title: str) -> None:
    existing = (
        sb.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    if existing.data:
        sb.table("chat_sessions").update({"title": title}).eq("id", session_id).eq(
            "user_id", user_id
        ).execute()
    else:
        sb.table("chat_sessions").insert(
            {"id": session_id, "user_id": user_id, "title": title}
        ).execute()


def save_messages(
    sb: Client, user_id: str, session_id: str, messages: list[dict[str, str]]
) -> None:
    sb.table("chat_messages").delete().eq("user_id", user_id).eq(
        "session_id", session_id
    ).execute()
    if not messages:
        return
    rows = [
        {
            "user_id": user_id,
            "session_id": session_id,
            "role": m["role"],
            "content": m["content"],
            "message_order": idx,
        }
        for idx, m in enumerate(messages)
    ]
    sb.table("chat_messages").insert(rows).execute()


def auto_save_session(
    sb: Client,
    llm: ChatOpenAI,
    user_id: str,
    session_id: str,
    messages: list[dict[str, str]],
) -> None:
    title = "새 세션"
    user_msgs = [m for m in messages if m["role"] == "user"]
    asst_msgs = [m for m in messages if m["role"] == "assistant"]
    if user_msgs and asst_msgs:
        title = _generate_session_title(llm, user_msgs[0]["content"], asst_msgs[0]["content"])
    elif user_msgs:
        title = user_msgs[0]["content"][:40]

    _upsert_session_row(sb, user_id, session_id, title)
    save_messages(sb, user_id, session_id, messages)
    st.session_state.session_titles[session_id] = title


def insert_new_session(
    sb: Client,
    llm: ChatOpenAI,
    user_id: str,
    messages: list[dict[str, str]],
    source_session_id: str,
) -> str:
    new_id = str(uuid.uuid4())
    title = "새 세션"
    user_msgs = [m for m in messages if m["role"] == "user"]
    asst_msgs = [m for m in messages if m["role"] == "assistant"]
    if user_msgs and asst_msgs:
        title = _generate_session_title(llm, user_msgs[0]["content"], asst_msgs[0]["content"])
    elif user_msgs:
        title = user_msgs[0]["content"][:40]

    sb.table("chat_sessions").insert(
        {"id": new_id, "user_id": user_id, "title": title}
    ).execute()
    save_messages(sb, user_id, new_id, messages)

    vec_resp = (
        sb.table("vector_documents")
        .select("file_name, content, metadata, embedding")
        .eq("user_id", user_id)
        .eq("session_id", source_session_id)
        .execute()
    )
    rows = vec_resp.data or []
    if rows:
        copy_rows = [
            {
                "user_id": user_id,
                "session_id": new_id,
                "file_name": r["file_name"],
                "content": r["content"],
                "metadata": r.get("metadata") or {},
                "embedding": r["embedding"],
            }
            for r in rows
        ]
        for i in range(0, len(copy_rows), VECTOR_BATCH_SIZE):
            sb.table("vector_documents").insert(copy_rows[i : i + VECTOR_BATCH_SIZE]).execute()

    return new_id


def delete_session(sb: Client, user_id: str, session_id: str) -> None:
    sb.table("chat_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()


def load_session_into_state(sb: Client, user_id: str, session_id: str) -> None:
    messages = fetch_messages(sb, user_id, session_id)
    file_names = fetch_vector_file_names(sb, user_id, session_id)
    st.session_state.current_session_id = session_id
    st.session_state.chat_history = messages
    st.session_state.conversation_memory = messages[-MEMORY_MAX:]
    st.session_state.processed_names = file_names
    st.session_state.sidebar_selected_id = session_id


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def _embed_texts(embeddings: OpenAIEmbeddings, texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts)


def store_pdf_vectors(
    sb: Client,
    embeddings: OpenAIEmbeddings,
    user_id: str,
    session_id: str,
    uploaded_files: list[Any],
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    processed: list[str] = []

    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            docs = PyPDFLoader(tmp_path).load()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if not docs:
            continue

        for doc in docs:
            doc.metadata["file_name"] = uf.name

        splits = splitter.split_documents(docs)
        if not splits:
            continue

        texts = [d.page_content for d in splits]
        metas = [d.metadata for d in splits]

        for i in range(0, len(texts), VECTOR_BATCH_SIZE):
            batch_texts = texts[i : i + VECTOR_BATCH_SIZE]
            batch_metas = metas[i : i + VECTOR_BATCH_SIZE]
            batch_embs = _embed_texts(embeddings, batch_texts)
            rows = [
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "file_name": m.get("file_name") or uf.name,
                    "content": t,
                    "metadata": m,
                    "embedding": emb,
                }
                for t, m, emb in zip(batch_texts, batch_metas, batch_embs, strict=True)
            ]
            sb.table("vector_documents").insert(rows).execute()

        processed.append(uf.name)

    return processed


def search_vectors(
    sb: Client,
    embeddings: OpenAIEmbeddings,
    query: str,
    user_id: str,
    session_id: str,
    k: int = RETRIEVER_K,
) -> list[Document]:
    query_emb = embeddings.embed_query(query)
    try:
        resp = sb.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_emb,
                "match_count": k,
                "filter_session_id": session_id,
                "filter_user_id": user_id,
            },
        ).execute()
        docs: list[Document] = []
        for row in resp.data or []:
            docs.append(
                Document(
                    page_content=row.get("content", ""),
                    metadata={
                        "file_name": row.get("file_name", ""),
                        **(row.get("metadata") or {}),
                    },
                )
            )
        return docs
    except Exception as exc:  # noqa: BLE001
        logger.warning("RPC match_vector_documents failed: %s", exc)
        resp = (
            sb.table("vector_documents")
            .select("content, file_name, metadata, embedding")
            .eq("user_id", user_id)
            .eq("session_id", session_id)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return []

        def _cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scored = []
        for row in rows:
            emb = row.get("embedding")
            if isinstance(emb, str):
                emb = json.loads(emb)
            if not emb:
                continue
            scored.append((_cosine(query_emb, emb), row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            Document(
                page_content=row["content"],
                metadata={"file_name": row.get("file_name", ""), **(row.get("metadata") or {})},
            )
            for _, row in scored[:k]
        ]


def _stream_llm(llm: ChatOpenAI, messages: list[Any], placeholder: Any) -> str:
    acc = ""
    for chunk in llm.stream(messages):
        piece = getattr(chunk, "content", "") or ""
        if piece:
            acc += piece
            placeholder.markdown(remove_separators(acc) + "▌")
    return remove_separators(acc)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_session() -> None:
    defaults: dict[str, Any] = {
        "logged_in": False,
        "current_user_id": None,
        "current_login_id": "",
        "chat_history": [],
        "conversation_memory": [],
        "processed_names": [],
        "current_session_id": str(uuid.uuid4()),
        "sessions_list": [],
        "session_titles": {},
        "sidebar_selected_id": None,
        "last_dropdown_id": None,
        "db_ready": False,
        "auth_mode": "login",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_chat_state() -> None:
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.processed_names = []
    st.session_state.sidebar_selected_id = None
    st.session_state.last_dropdown_id = None
    st.session_state.db_ready = False


def _logout() -> None:
    st.session_state.logged_in = False
    st.session_state.current_user_id = None
    st.session_state.current_login_id = ""
    st.session_state.sessions_list = []
    st.session_state.session_titles = {}
    _reset_chat_state()


def _ensure_db_session(sb: Client, llm: ChatOpenAI, user_id: str) -> None:
    sid = st.session_state.current_session_id
    existing = (
        sb.table("chat_sessions")
        .select("id")
        .eq("id", sid)
        .eq("user_id", user_id)
        .execute()
    )
    if not existing.data and st.session_state.chat_history:
        auto_save_session(sb, llm, user_id, sid, st.session_state.chat_history)
    elif not existing.data:
        sb.table("chat_sessions").insert(
            {"id": sid, "user_id": user_id, "title": "새 세션"}
        ).execute()


def _refresh_sessions(sb: Client, user_id: str) -> None:
    sessions = fetch_sessions(sb, user_id)
    st.session_state.sessions_list = sessions
    st.session_state.session_titles = {s["id"]: s["title"] for s in sessions}


def _render_auth_page(sb: Client | None) -> None:
    st.markdown(
        f"""
<h1 style="text-align:center; color:#1f77b4;">{CHATBOT_TITLE}</h1>
<p style="text-align:center;">로그인 또는 회원가입 후 이용해 주세요.</p>
""",
        unsafe_allow_html=True,
    )

    if sb is None:
        st.error("Supabase 연결 정보가 없어 로그인할 수 없습니다.")
        return

    mode = st.radio(
        "모드",
        ["로그인", "회원가입"],
        horizontal=True,
        key="auth_mode_radio",
    )
    st.session_state.auth_mode = "login" if mode == "로그인" else "signup"

    login_id = st.text_input("아이디 (login_id)")
    password = st.text_input("비밀번호", type="password")

    if st.button("확인", use_container_width=True):
        if st.session_state.auth_mode == "signup":
            ok, msg = register_user(sb, login_id, password)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        else:
            ok, msg, uid = login_user(sb, login_id, password)
            if ok and uid:
                st.session_state.logged_in = True
                st.session_state.current_user_id = uid
                st.session_state.current_login_id = login_id.strip()
                _reset_chat_state()
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


def _render_sidebar(sb: Client | None, llm: ChatOpenAI | None, user_id: str) -> None:
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.current_login_id}")
        if st.button("로그아웃", use_container_width=True):
            _logout()
            st.rerun()

        st.divider()
        st.markdown("### 세션 관리")

        if sb is None:
            st.warning("Supabase 연결 정보가 없어 세션 기능을 사용할 수 없습니다.")
            return

        _refresh_sessions(sb, user_id)
        sessions = st.session_state.sessions_list
        options: list[str] = []
        id_by_label: dict[str, str] = {}
        for s in sessions:
            label = f"{s['title']} ({s['id'][:8]}…)"
            options.append(label)
            id_by_label[label] = s["id"]

        selected_id: str | None = None
        if options:
            default_idx = 0
            cur = st.session_state.sidebar_selected_id
            if cur:
                for i, s in enumerate(sessions):
                    if s["id"] == cur:
                        default_idx = i
                        break
            selected_label = st.selectbox("세션 선택", options, index=default_idx)
            selected_id = id_by_label[selected_label]

            if selected_id != st.session_state.last_dropdown_id:
                load_session_into_state(sb, user_id, selected_id)
                st.session_state.last_dropdown_id = selected_id
                st.rerun()
        else:
            st.info("저장된 세션이 없습니다.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("세션저장", use_container_width=True):
                if llm is None:
                    st.error("OPENAI_API_KEY가 필요합니다.")
                elif not st.session_state.chat_history:
                    st.warning("저장할 대화가 없습니다.")
                else:
                    src = st.session_state.current_session_id
                    new_id = insert_new_session(
                        sb, llm, user_id, st.session_state.chat_history, src
                    )
                    st.session_state.current_session_id = new_id
                    st.session_state.sidebar_selected_id = new_id
                    st.session_state.last_dropdown_id = new_id
                    _refresh_sessions(sb, user_id)
                    st.success("새 세션이 저장되었습니다.")
                    st.rerun()

            if st.button("세션로드", use_container_width=True):
                if selected_id:
                    load_session_into_state(sb, user_id, selected_id)
                    st.session_state.last_dropdown_id = selected_id
                    st.success("세션을 불러왔습니다.")
                    st.rerun()
                else:
                    st.warning("불러올 세션을 선택하세요.")

        with col2:
            if st.button("세션삭제", use_container_width=True):
                target = st.session_state.sidebar_selected_id or selected_id
                if target:
                    delete_session(sb, user_id, target)
                    if target == st.session_state.current_session_id:
                        _reset_chat_state()
                    _refresh_sessions(sb, user_id)
                    st.success("세션이 삭제되었습니다.")
                    st.rerun()
                else:
                    st.warning("삭제할 세션을 선택하세요.")

            if st.button("화면초기화", use_container_width=True):
                _reset_chat_state()
                st.rerun()

        if st.button("vectordb", use_container_width=True):
            sid = st.session_state.current_session_id
            names = fetch_vector_file_names(sb, user_id, sid)
            if names:
                st.markdown("**현재 Vector DB 파일**")
                for n in names:
                    st.text(f"- {n}")
            else:
                st.info("현재 세션에 저장된 벡터 문서가 없습니다.")

        st.divider()
        uploads = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("파일 처리하기"):
            keys = _get_env_keys()
            if not keys["openai"]:
                st.error("OPENAI_API_KEY가 설정되어 있지 않습니다.")
            elif not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            else:
                try:
                    if llm:
                        _ensure_db_session(sb, llm, user_id)
                    emb = _get_embeddings(keys["openai"])
                    sid = st.session_state.current_session_id
                    names = store_pdf_vectors(sb, emb, user_id, sid, list(uploads))
                    st.session_state.processed_names = list(
                        dict.fromkeys(st.session_state.processed_names + names)
                    )
                    if llm:
                        auto_save_session(
                            sb, llm, user_id, sid, st.session_state.chat_history
                        )
                    _refresh_sessions(sb, user_id)
                    st.success("PDF 처리 및 세션 자동 저장이 완료되었습니다.")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PDF 처리 실패: %s", exc)
                    st.error(f"PDF 처리 중 오류: {exc}")

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"- {name}")

        sid = st.session_state.current_session_id
        title = st.session_state.session_titles.get(sid, "(미저장)")
        st.text(
            f"모델: {MODEL_NAME}\n"
            f"현재 세션: {title}\n"
            f"세션 ID: {sid[:8]}…\n"
            f"대화 수: {len(st.session_state.chat_history)}\n"
            f"벡터 파일 수: {len(st.session_state.processed_names)}"
        )


def _render_header() -> None:
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            f"""
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">재정경제부</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def main() -> None:
    st.set_page_config(
        page_title=CHATBOT_TITLE,
        page_icon="📚",
        layout="wide",
    )
    _init_session()

    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
""",
        unsafe_allow_html=True,
    )

    keys = _get_env_keys()
    missing = _missing_keys(keys)
    if missing:
        src = "Streamlit secrets 또는 `.env`"
        st.error(f"다음 환경 변수가 {src}에 설정되어 있지 않습니다: " + ", ".join(missing))

    sb = _get_supabase() if not missing else None
    llm = _get_llm(keys["openai"]) if keys["openai"] else None

    if not st.session_state.logged_in:
        _render_auth_page(sb)
        return

    user_id = _current_user_id()
    _render_header()

    if sb and llm and not st.session_state.db_ready:
        _refresh_sessions(sb, user_id)
        if st.session_state.sessions_list and not st.session_state.chat_history:
            latest = st.session_state.sessions_list[0]
            load_session_into_state(sb, user_id, latest["id"])
            st.session_state.last_dropdown_id = latest["id"]
        st.session_state.db_ready = True

    _render_sidebar(sb, llm, user_id)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    if missing:
        st.warning("API 키를 설정한 뒤 다시 시도해 주세요.")
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append({"role": "user", "content": user_input})
    if len(st.session_state.conversation_memory) > MEMORY_MAX:
        st.session_state.conversation_memory = st.session_state.conversation_memory[
            -MEMORY_MAX:
        ]

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            assert sb is not None and llm is not None
            _ensure_db_session(sb, llm, user_id)
            sid = st.session_state.current_session_id
            emb = _get_embeddings(keys["openai"])
            file_names = fetch_vector_file_names(sb, user_id, sid)

            if file_names:
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                docs = search_vectors(sb, emb, user_input, user_id, sid, k=RETRIEVER_K)
                context = "\n\n".join(d.page_content for d in docs)
                messages = _build_rag_messages(user_input, context, mem_txt)
                full_answer = _stream_llm(llm, messages, placeholder)
            else:
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                sys = f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n{mem_txt or '(없음)'}"
                messages = [
                    SystemMessage(content=sys),
                    HumanMessage(content=user_input),
                ]
                full_answer = _stream_llm(llm, messages, placeholder)

            placeholder.markdown(full_answer)

            if full_answer and not full_answer.lstrip().startswith("# 오류"):
                follow = _generate_followup_section(llm, user_input, full_answer)
                if follow:
                    full_answer += follow
                    placeholder.markdown(remove_separators(full_answer))

        except Exception as exc:  # noqa: BLE001
            logger.warning("답변 생성 실패: %s", exc)
            full_answer = (
                f"# 오류\n\n요청을 처리하는 중 문제가 발생했습니다.\n\n`{exc}`"
            )
            placeholder.markdown(remove_separators(full_answer))

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_answer}
        )
        st.session_state.conversation_memory.append(
            {"role": "assistant", "content": full_answer}
        )
        if len(st.session_state.conversation_memory) > MEMORY_MAX:
            st.session_state.conversation_memory = (
                st.session_state.conversation_memory[-MEMORY_MAX:]
            )

        if sb and llm:
            try:
                auto_save_session(
                    sb,
                    llm,
                    user_id,
                    st.session_state.current_session_id,
                    st.session_state.chat_history,
                )
                _refresh_sessions(sb, user_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("자동 저장 실패: %s", exc)


if __name__ == "__main__":
    main()
