"""멀티유저 / 멀티세션 RAG 챗봇 — Supabase user 테이블 기반 인증 및 세션 저장.

Supabase Authentication(auth.users)은 사용하지 않습니다.
사용자 관리는 DB의 `user` 테이블에서 앱이 직접 처리합니다.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import secrets
import ssl
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import certifi
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client


def configure_ssl_certificates() -> str:
    """SSL 검증용 CA 번들 경로를 설정한다.

    Windows에서는 Avast 등 HTTPS 검사가 시스템 루트 CA를 쓰므로,
    certifi + Windows 인증서 저장소를 합친 번들을 사용한다.
    """
    ca_path = Path(tempfile.gettempdir()) / "ktena_ssl_ca_bundle.pem"
    try:
        parts: list[bytes] = [Path(certifi.where()).read_bytes()]
        if hasattr(ssl, "enum_certificates"):
            chunks: list[bytes] = []
            for store in ("CA", "ROOT"):
                try:
                    for cert, encoding, _trust in ssl.enum_certificates(store):
                        if encoding != "x509_asn":
                            continue
                        b64 = base64.b64encode(cert).decode("ascii")
                        body = "\n".join(b64[i : i + 64] for i in range(0, len(b64), 64))
                        chunks.append(
                            (
                                "-----BEGIN CERTIFICATE-----\n"
                                f"{body}\n"
                                "-----END CERTIFICATE-----\n"
                            ).encode("ascii")
                        )
                except Exception:
                    continue
            if chunks:
                parts.append(b"\n".join(chunks))
        ca_path.write_bytes(b"\n".join(parts))
        bundle = str(ca_path)
    except Exception:
        bundle = certifi.where()

    os.environ["SSL_CERT_FILE"] = bundle
    os.environ["REQUESTS_CA_BUNDLE"] = bundle
    return bundle


configure_ssl_certificates()


# ──────────────────────────────────────────────
# 경로 및 환경 변수
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
LOGO_PATH = PROJECT_ROOT / "logo.png"

load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_BATCH_SIZE = 10
VECTOR_MATCH_COUNT = 10
PBKDF2_ITERATIONS = 100_000

ANSWER_FORMAT_INSTRUCTION = """
답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요.
주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요.
답변은 서술형으로 작성하되 존대말을 사용하세요.
완전한 문장으로 서술하세요.
구분선(---, ===, ___) 사용 금지.
취소선(~~텍스트~~) 사용 금지.
참조 표시나 출처 문구 사용 금지.
"""

RAG_SYSTEM_PROMPT = (
    "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. "
    "그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. 말투는 존대말 한글로 해줘. "
    + ANSWER_FORMAT_INSTRUCTION
)

DIRECT_LLM_SYSTEM_PROMPT = (
    "당신은 친절하고 유능한 AI 어시스턴트입니다. "
    + ANSWER_FORMAT_INSTRUCTION
)

FOLLOW_UP_SYSTEM_PROMPT = (
    "사용자와 AI의 대화를 바탕으로, 사용자가 이어서 물어볼 만한 질문 3개를 생성하세요. "
    "각 질문은 한 줄로 작성하고, 번호 없이 질문만 줄바꿈으로 구분하세요. "
    "질문만 출력하고 다른 설명은 하지 마세요."
)

TITLE_SYSTEM_PROMPT = (
    "다음 대화의 첫 질문과 답변을 바탕으로 세션 제목을 한글로 만들어 주세요. "
    "제목만 출력하고, 따옴표나 설명 없이 20자 이내로 작성하세요."
)


# ──────────────────────────────────────────────
# 시크릿 / 환경변수 로딩 (st.secrets 우선)
# ──────────────────────────────────────────────
def get_secret(key: str) -> str:
    """Streamlit Cloud secrets → .env / os.getenv 순으로 키를 읽는다."""
    try:
        if key in st.secrets and st.secrets[key]:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return (os.getenv(key) or "").strip()


OPENAI_API_KEY = ""
SUPABASE_URL = ""
SUPABASE_ANON_KEY = ""


def refresh_secrets() -> None:
    """Streamlit 런타임에서 secrets / .env 키를 다시 읽는다."""
    global OPENAI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY
    OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
    SUPABASE_URL = get_secret("SUPABASE_URL")
    SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")


# ──────────────────────────────────────────────
# 로깅
# ──────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger("multiusers")
    logger.setLevel(logging.WARNING)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "langchain",
        "langchain_openai",
        "supabase",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


LOGGER = setup_logging()


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────
def remove_separators(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"~~(.+?)~~", r"\1", text, flags=re.DOTALL)
    cleaned = re.sub(r"^[\-\=_]{3,}\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def format_memory_context(memory: list[dict[str, str]], limit: int = 50) -> str:
    recent = memory[-limit:]
    lines: list[str] = []
    for item in recent:
        role = "사용자" if item["role"] == "user" else "어시스턴트"
        lines.append(f"{role}: {item['content']}")
    return "\n".join(lines)


def append_follow_up_section(answer: str, follow_up_questions: list[str]) -> str:
    section_lines = ["### 💡 다음에 물어볼 수 있는 질문들"]
    for idx, question in enumerate(follow_up_questions[:3], start=1):
        section_lines.append(f"{idx}. {question.strip()}")
    return f"{answer.rstrip()}\n\n" + "\n".join(section_lines)


def parse_follow_up_questions(raw_text: str) -> list[str]:
    questions: list[str] = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        if cleaned:
            questions.append(cleaned)
    return questions[:3]


def missing_keys_message() -> str | None:
    missing: list[str] = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_ANON_KEY:
        missing.append("SUPABASE_ANON_KEY")
    if not missing:
        return None
    return (
        "다음 키가 설정되지 않았습니다: "
        + ", ".join(missing)
        + "\nStreamlit Cloud에서는 `st.secrets`에, 로컬에서는 "
        f"`.env`({ENV_PATH})에 설정해 주세요."
    )


def current_user_id() -> str | None:
    return st.session_state.get("user_id")


def require_user_id() -> str:
    uid = current_user_id()
    if not uid:
        raise RuntimeError("로그인이 필요합니다.")
    return uid


# ──────────────────────────────────────────────
# 비밀번호 해시 (평문 저장 금지)
# ──────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        _algo, iterations_s, salt, hash_hex = stored_hash.split("$", 3)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            int(iterations_s),
        )
        return secrets.compare_digest(digest.hex(), hash_hex)
    except Exception:
        return False


# ──────────────────────────────────────────────
# Supabase 클라이언트
# ──────────────────────────────────────────────
@st.cache_resource
def get_supabase_client() -> Client | None:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception as exc:
        LOGGER.error("Supabase 클라이언트 생성 실패: %s", exc)
        return None


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
        streaming=True,
    )


# ──────────────────────────────────────────────
# 사용자 인증 (user 테이블)
# ──────────────────────────────────────────────
def find_user_by_login_id(client: Client, login_id: str) -> dict[str, Any] | None:
    try:
        result = (
            client.table("user")
            .select("id, login_id, password_hash, created_at")
            .eq("login_id", login_id.strip())
            .limit(1)
            .execute()
        )
        if result.data:
            return result.data[0]
        return None
    except Exception as exc:
        LOGGER.error("사용자 조회 실패: %s", exc)
        return None


def register_user(client: Client, login_id: str, password: str) -> tuple[bool, str]:
    login_id = login_id.strip()
    if not login_id or not password:
        return False, "아이디와 비밀번호를 모두 입력해 주세요."
    if len(login_id) < 3:
        return False, "아이디는 3자 이상이어야 합니다."
    if len(password) < 4:
        return False, "비밀번호는 4자 이상이어야 합니다."

    existing = find_user_by_login_id(client, login_id)
    if existing:
        return False, "이미 사용 중인 아이디입니다."

    try:
        result = (
            client.table("user")
            .insert(
                {
                    "login_id": login_id,
                    "password_hash": hash_password(password),
                }
            )
            .execute()
        )
        if not result.data:
            return False, "회원가입에 실패했습니다."
        return True, "회원가입이 완료되었습니다. 로그인해 주세요."
    except Exception as exc:
        LOGGER.error("회원가입 실패: %s", exc)
        return False, f"회원가입 중 오류가 발생했습니다: {exc}"


def authenticate_user(
    client: Client, login_id: str, password: str
) -> tuple[dict[str, Any] | None, str]:
    login_id = login_id.strip()
    if not login_id or not password:
        return None, "아이디와 비밀번호를 모두 입력해 주세요."

    user = find_user_by_login_id(client, login_id)
    if not user:
        return None, "아이디 또는 비밀번호가 올바르지 않습니다."

    if not verify_password(password, user.get("password_hash") or ""):
        return None, "아이디 또는 비밀번호가 올바르지 않습니다."

    return user, "로그인 성공"


def session_belongs_to_user(client: Client, session_id: str, user_id: str) -> bool:
    try:
        result = (
            client.table("chat_sessions")
            .select("id")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        return bool(result.data)
    except Exception as exc:
        LOGGER.error("세션 소유권 확인 실패: %s", exc)
        return False


# ──────────────────────────────────────────────
# 세션 / 메시지 / 벡터 DB (항상 user_id 필터)
# ──────────────────────────────────────────────
def fetch_sessions(client: Client, user_id: str) -> list[dict[str, Any]]:
    try:
        result = (
            client.table("chat_sessions")
            .select("id, title, processed_files, created_at, updated_at, user_id")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )
        return list(result.data or [])
    except Exception as exc:
        LOGGER.error("세션 목록 조회 실패: %s", exc)
        return []


def ensure_session_row(
    client: Client,
    session_id: str,
    user_id: str,
    title: str = "새 세션",
    processed_files: list[str] | None = None,
) -> bool:
    """세션이 없으면 생성하고, 있으면 processed_files만 갱신한다."""
    files = processed_files if processed_files is not None else []
    try:
        existing = (
            client.table("chat_sessions")
            .select("id")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if existing.data:
            client.table("chat_sessions").update(
                {"processed_files": files}
            ).eq("id", session_id).eq("user_id", user_id).execute()
        else:
            client.table("chat_sessions").insert(
                {
                    "id": session_id,
                    "user_id": user_id,
                    "title": title,
                    "processed_files": files,
                }
            ).execute()
        return True
    except Exception as exc:
        LOGGER.error("세션 행 보장 실패: %s", exc)
        return False


def replace_session_messages(
    client: Client,
    session_id: str,
    user_id: str,
    chat_history: list[dict[str, str]],
) -> bool:
    try:
        (
            client.table("chat_messages")
            .delete()
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not chat_history:
            return True
        rows = [
            {
                "session_id": session_id,
                "user_id": user_id,
                "role": msg["role"],
                "content": msg["content"],
                "message_order": idx,
            }
            for idx, msg in enumerate(chat_history)
        ]
        client.table("chat_messages").insert(rows).execute()
        return True
    except Exception as exc:
        LOGGER.error("메시지 저장 실패: %s", exc)
        return False


def load_session_messages(
    client: Client, session_id: str, user_id: str
) -> list[dict[str, str]]:
    try:
        result = (
            client.table("chat_messages")
            .select("role, content, message_order")
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .order("message_order")
            .execute()
        )
        return [
            {"role": row["role"], "content": row["content"]}
            for row in (result.data or [])
        ]
    except Exception as exc:
        LOGGER.error("메시지 로드 실패: %s", exc)
        return []


def load_session_file_names(client: Client, session_id: str, user_id: str) -> list[str]:
    if not session_belongs_to_user(client, session_id, user_id):
        return []
    try:
        result = (
            client.table("vector_documents")
            .select("file_name")
            .eq("session_id", session_id)
            .execute()
        )
        names = sorted(
            {row["file_name"] for row in (result.data or []) if row.get("file_name")}
        )
        return names
    except Exception as exc:
        LOGGER.error("벡터 파일명 조회 실패: %s", exc)
        return []


def delete_session(client: Client, session_id: str, user_id: str) -> bool:
    try:
        (
            client.table("chat_sessions")
            .delete()
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )
        return True
    except Exception as exc:
        LOGGER.error("세션 삭제 실패: %s", exc)
        return False


def generate_session_title(user_query: str, answer: str) -> str:
    try:
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY,
        )
        messages = [
            SystemMessage(content=TITLE_SYSTEM_PROMPT),
            HumanMessage(
                content=f"질문: {user_query}\n\n답변: {answer[:800]}"
            ),
        ]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        title = remove_separators(str(content)).splitlines()[0].strip().strip("\"'")
        return title[:40] if title else "새 세션"
    except Exception as exc:
        LOGGER.warning("세션 제목 생성 실패: %s", exc)
        fallback = user_query.strip().replace("\n", " ")
        return (fallback[:30] + "…") if len(fallback) > 30 else (fallback or "새 세션")


def insert_vector_documents(
    client: Client,
    session_id: str,
    chunks: list[Document],
    file_name: str,
) -> str | None:
    """임베딩 후 vector_documents에 직접 INSERT (file_name NULL 방지)."""
    if not chunks:
        return None

    embeddings = get_embeddings()
    try:
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
            texts = [doc.page_content for doc in batch]
            vectors = embeddings.embed_documents(texts)
            rows = []
            for doc, vector in zip(batch, vectors):
                meta = dict(doc.metadata or {})
                meta["file_name"] = file_name
                meta["session_id"] = session_id
                rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "content": doc.page_content,
                        "metadata": meta,
                        "embedding": vector,
                        "file_name": file_name,
                    }
                )
            client.table("vector_documents").insert(rows).execute()
        return None
    except Exception as exc:
        LOGGER.error("벡터 저장 실패 (%s): %s", file_name, exc)
        return f"'{file_name}' 벡터 저장 중 오류가 발생했습니다: {exc}"


def process_pdf_files_to_supabase(
    client: Client,
    session_id: str,
    user_id: str,
    uploaded_files: list[Any],
) -> tuple[list[str], str | None]:
    """PDF를 세션별로 청크/임베딩하여 Supabase에 저장한다."""
    if not OPENAI_API_KEY:
        return [], "OPENAI_API_KEY가 설정되지 않았습니다."

    if not ensure_session_row(
        client,
        session_id,
        user_id,
        title=st.session_state.get("session_title") or "새 세션",
        processed_files=st.session_state.get("processed_files") or [],
    ):
        return [], "세션을 Supabase에 생성하지 못했습니다."

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    processed_names: list[str] = []

    for uploaded_file in uploaded_files:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata = dict(chunk.metadata or {})
                chunk.metadata["file_name"] = uploaded_file.name
                chunk.metadata["source"] = uploaded_file.name

            error = insert_vector_documents(
                client, session_id, chunks, uploaded_file.name
            )
            if error:
                return processed_names, error
            processed_names.append(uploaded_file.name)
        except Exception as exc:
            LOGGER.error("PDF 처리 실패 (%s): %s", uploaded_file.name, exc)
            return processed_names, f"'{uploaded_file.name}' 파일 처리 중 오류가 발생했습니다."
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if not processed_names:
        return [], "PDF에서 텍스트를 추출하지 못했습니다."

    merged = list(
        dict.fromkeys((st.session_state.get("processed_files") or []) + processed_names)
    )
    ensure_session_row(client, session_id, user_id, processed_files=merged)
    return merged, None


def retrieve_documents(
    client: Client,
    session_id: str,
    user_id: str,
    query: str,
    k: int = VECTOR_MATCH_COUNT,
) -> list[Document]:
    """match_vector_documents RPC로 세션 필터 검색 (소유권 확인 후)."""
    if not session_belongs_to_user(client, session_id, user_id):
        return []

    embeddings = get_embeddings()
    query_embedding = embeddings.embed_query(query)

    try:
        result = client.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_embedding,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        docs: list[Document] = []
        for row in result.data or []:
            meta = dict(row.get("metadata") or {})
            meta["file_name"] = row.get("file_name")
            meta["session_id"] = row.get("session_id")
            meta["similarity"] = row.get("similarity")
            docs.append(Document(page_content=row.get("content") or "", metadata=meta))
        return docs
    except Exception as exc:
        LOGGER.warning("RPC 검색 실패, 폴백 조회 시도: %s", exc)
        try:
            result = (
                client.table("vector_documents")
                .select("content, metadata, file_name")
                .eq("session_id", session_id)
                .limit(k)
                .execute()
            )
            docs = []
            for row in result.data or []:
                meta = dict(row.get("metadata") or {})
                meta["file_name"] = row.get("file_name")
                docs.append(
                    Document(page_content=row.get("content") or "", metadata=meta)
                )
            return docs
        except Exception as fallback_exc:
            LOGGER.error("폴백 검색도 실패: %s", fallback_exc)
            return []


def session_has_vectors(client: Client, session_id: str, user_id: str) -> bool:
    if not session_belongs_to_user(client, session_id, user_id):
        return False
    try:
        result = (
            client.table("vector_documents")
            .select("id")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )
        return bool(result.data)
    except Exception as exc:
        LOGGER.error("벡터 존재 여부 확인 실패: %s", exc)
        return False


# ──────────────────────────────────────────────
# 자동 / 수동 세션 저장
# ──────────────────────────────────────────────
def auto_save_session(client: Client, *, generate_title: bool = False) -> str | None:
    """현재 화면 상태를 Supabase에 upsert(동일 session_id 갱신)한다."""
    user_id = require_user_id()
    session_id = st.session_state.session_id
    chat_history = st.session_state.chat_history
    processed_files = st.session_state.processed_files or []

    title = st.session_state.session_title or "새 세션"
    if generate_title and len(chat_history) >= 2:
        first_user = next(
            (m["content"] for m in chat_history if m["role"] == "user"), ""
        )
        first_assistant = next(
            (m["content"] for m in chat_history if m["role"] == "assistant"), ""
        )
        if first_user and first_assistant and (
            not st.session_state.session_title
            or st.session_state.session_title == "새 세션"
        ):
            title = generate_session_title(first_user, first_assistant)
            st.session_state.session_title = title

    if not ensure_session_row(
        client, session_id, user_id, title=title, processed_files=processed_files
    ):
        return "세션 자동 저장에 실패했습니다."

    try:
        (
            client.table("chat_sessions")
            .update({"title": title, "processed_files": processed_files})
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )
    except Exception as exc:
        LOGGER.error("세션 제목 갱신 실패: %s", exc)

    if not replace_session_messages(client, session_id, user_id, chat_history):
        return "대화 메시지 자동 저장에 실패했습니다."
    return None


def manual_save_as_new_session(client: Client) -> tuple[str | None, str | None]:
    """
    세션저장 버튼: 기존 세션은 그대로 두고 INSERT로 새 세션을 만든다.
    현재 메시지/벡터도 새 session_id로 복제 저장한다.
    """
    user_id = require_user_id()
    old_session_id = st.session_state.session_id
    chat_history = st.session_state.chat_history
    processed_files = list(st.session_state.processed_files or [])

    if not chat_history and not processed_files:
        return None, "저장할 대화 또는 문서가 없습니다."

    first_user = next((m["content"] for m in chat_history if m["role"] == "user"), "")
    first_assistant = next(
        (m["content"] for m in chat_history if m["role"] == "assistant"), ""
    )
    if first_user and first_assistant:
        title = generate_session_title(first_user, first_assistant)
    else:
        title = processed_files[0] if processed_files else "새 세션"

    new_session_id = str(uuid.uuid4())
    try:
        client.table("chat_sessions").insert(
            {
                "id": new_session_id,
                "user_id": user_id,
                "title": title,
                "processed_files": processed_files,
            }
        ).execute()
    except Exception as exc:
        LOGGER.error("새 세션 INSERT 실패: %s", exc)
        return None, f"세션 저장 실패: {exc}"

    if not replace_session_messages(client, new_session_id, user_id, chat_history):
        return None, "메시지는 저장하지 못했습니다."

    # 벡터 문서 복제 (소유 세션만)
    try:
        if session_belongs_to_user(client, old_session_id, user_id):
            old_vectors = (
                client.table("vector_documents")
                .select("content, metadata, embedding, file_name")
                .eq("session_id", old_session_id)
                .execute()
            )
            rows = []
            for row in old_vectors.data or []:
                meta = dict(row.get("metadata") or {})
                meta["session_id"] = new_session_id
                rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "session_id": new_session_id,
                        "content": row["content"],
                        "metadata": meta,
                        "embedding": row.get("embedding"),
                        "file_name": row["file_name"],
                    }
                )
            for start in range(0, len(rows), EMBEDDING_BATCH_SIZE):
                batch = rows[start : start + EMBEDDING_BATCH_SIZE]
                if batch:
                    client.table("vector_documents").insert(batch).execute()
    except Exception as exc:
        LOGGER.error("벡터 복제 실패: %s", exc)
        return title, f"세션은 저장됐지만 벡터 복제에 실패했습니다: {exc}"

    st.session_state.session_id = new_session_id
    st.session_state.session_title = title
    return title, None


# ──────────────────────────────────────────────
# LLM 답변
# ──────────────────────────────────────────────
def generate_follow_up_questions(llm: Any, user_query: str, answer: str) -> list[str]:
    try:
        messages = [
            SystemMessage(content=FOLLOW_UP_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"사용자 질문:\n{user_query}\n\n"
                    f"AI 답변:\n{answer}\n\n"
                    "위 대화를 바탕으로 후속 질문 3개를 생성하세요."
                )
            ),
        ]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        questions = parse_follow_up_questions(str(content))
        while len(questions) < 3:
            questions.append("이 주제에 대해 더 자세히 설명해 주실 수 있나요?")
        return questions[:3]
    except Exception as exc:
        LOGGER.warning("후속 질문 생성 실패: %s", exc)
        return [
            "이 내용을 더 쉽게 설명해 주실 수 있나요?",
            "관련된 다른 주제도 알려 주실 수 있나요?",
            "실생활에서 어떻게 활용할 수 있나요?",
        ]


def stream_llm_response(llm: Any, messages: list[Any], placeholder: Any) -> str:
    full_response = ""
    for chunk in llm.stream(messages):
        piece = chunk.content if hasattr(chunk, "content") else str(chunk)
        if isinstance(piece, list):
            piece = "".join(
                getattr(p, "text", str(p)) if not isinstance(p, str) else p
                for p in piece
            )
        if piece:
            full_response += piece
            placeholder.markdown(remove_separators(full_response))
    return remove_separators(full_response)


def generate_direct_llm_answer(
    llm: Any,
    user_query: str,
    conversation_memory: list[dict[str, str]],
    placeholder: Any,
) -> str:
    memory_context = format_memory_context(conversation_memory)
    messages = [
        SystemMessage(content=DIRECT_LLM_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"이전 대화:\n{memory_context}\n\n"
                f"현재 질문:\n{user_query}"
            )
        ),
    ]
    answer = stream_llm_response(llm, messages, placeholder)
    follow_up = generate_follow_up_questions(llm, user_query, answer)
    final_answer = append_follow_up_section(answer, follow_up)
    placeholder.markdown(final_answer)
    return final_answer


def generate_rag_answer(
    llm: Any,
    client: Client,
    session_id: str,
    user_id: str,
    user_query: str,
    conversation_memory: list[dict[str, str]],
    placeholder: Any,
) -> str:
    docs = retrieve_documents(client, session_id, user_id, user_query)
    if not docs:
        warning = (
            "참고 문서를 찾지 못했습니다. PDF를 업로드·처리했는지, "
            "세션이 올바른지 확인해 주세요."
        )
        placeholder.warning(warning)
        return warning

    context = "\n\n".join(doc.page_content for doc in docs)
    memory_context = format_memory_context(conversation_memory)
    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"이전 대화:\n{memory_context}\n\n"
                f"참고 문서:\n{context}\n\n"
                f"질문:\n{user_query}"
            )
        ),
    ]
    answer = stream_llm_response(llm, messages, placeholder)
    follow_up = generate_follow_up_questions(llm, user_query, answer)
    final_answer = append_follow_up_section(answer, follow_up)
    placeholder.markdown(final_answer)
    return final_answer


def update_conversation_memory(
    user_query: str,
    assistant_answer: str,
    conversation_memory: list[dict[str, str]],
) -> None:
    conversation_memory.append({"role": "user", "content": user_query})
    conversation_memory.append({"role": "assistant", "content": assistant_answer})
    if len(conversation_memory) > 50:
        del conversation_memory[:-50]


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        h1 { color: #ff69b4 !important; font-size: 1.9rem !important; }
        h2 { color: #ffd700 !important; font-size: 1.6rem !important; }
        h3 { color: #1f77b4 !important; font-size: 1.35rem !important; }

        div[data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.75rem;
            font-size: 1.25rem !important;
            line-height: 1.7 !important;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] span,
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
            font-size: 1.25rem !important;
            line-height: 1.7 !important;
        }

        div[data-testid="stChatInput"] textarea {
            font-size: 1.2rem !important;
        }

        div.stButton > button {
            background-color: #ff69b4 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
        }

        div.stButton > button:hover {
            background-color: #ff85c1 !important;
            color: white !important;
        }

        .ena-header-title {
            text-align: center !important;
            font-size: 2.4rem !important;
            line-height: 1.15 !important;
            font-weight: 700 !important;
            margin: 0.5rem 0 1rem 0 !important;
        }

        .ena-header-title .ena-blue {
            color: #1f77b4 !important;
            font-size: 2.4rem !important;
        }

        .ena-header-title .ena-gold {
            color: #ffd700 !important;
            font-size: 2.4rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    left_col, center_col, right_col = st.columns([1, 2, 1])
    with left_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("## 📚")
    with center_col:
        st.markdown(
            """
            <div class="ena-header-title">
                <span class="ena-blue">ENA</span>
                <span class="ena-gold">RAG 챗봇</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        st.empty()


def reset_local_session(new_id: str | None = None) -> None:
    st.session_state.session_id = new_id or str(uuid.uuid4())
    st.session_state.session_title = "새 세션"
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.processed_files = []
    st.session_state.has_vectors = False
    st.session_state.selected_session_id = None


def clear_auth_state() -> None:
    st.session_state.user_id = None
    st.session_state.login_id = None
    reset_local_session()
    st.session_state._last_select_id = None
    st.session_state.show_vectordb = False


def init_session_state() -> None:
    defaults = {
        "user_id": None,
        "login_id": None,
        "session_id": str(uuid.uuid4()),
        "session_title": "새 세션",
        "chat_history": [],
        "conversation_memory": [],
        "processed_files": [],
        "has_vectors": False,
        "selected_session_id": None,
        "pending_load_id": None,
        "show_vectordb": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_loaded_session(client: Client, session_id: str, title: str) -> None:
    user_id = require_user_id()
    if not session_belongs_to_user(client, session_id, user_id):
        st.warning("다른 사용자의 세션에는 접근할 수 없습니다.")
        return

    messages = load_session_messages(client, session_id, user_id)
    file_names = load_session_file_names(client, session_id, user_id)

    try:
        result = (
            client.table("chat_sessions")
            .select("processed_files, title")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if result.data:
            row = result.data[0]
            title = row.get("title") or title
            stored_files = row.get("processed_files") or []
            if isinstance(stored_files, list):
                file_names = list(dict.fromkeys(list(stored_files) + file_names))
    except Exception as exc:
        LOGGER.warning("세션 메타 로드 실패: %s", exc)

    st.session_state.session_id = session_id
    st.session_state.session_title = title
    st.session_state.chat_history = messages
    st.session_state.conversation_memory = list(messages[-50:])
    st.session_state.processed_files = file_names
    st.session_state.has_vectors = session_has_vectors(client, session_id, user_id)
    st.session_state.selected_session_id = session_id


def render_chat_history() -> None:
    for message in st.session_state.chat_history:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message["content"])


def render_auth_page(client: Client | None) -> None:
    st.markdown("### 로그인 / 회원가입")
    st.caption("Supabase Auth가 아닌 앱 DB `user` 테이블로 계정을 관리합니다.")

    if client is None:
        st.error("Supabase 연결이 필요합니다. 키 설정을 확인해 주세요.")
        return

    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    with tab_login:
        with st.form("login_form"):
            login_id = st.text_input("아이디 (login_id)", key="login_login_id")
            password = st.text_input(
                "비밀번호", type="password", key="login_password"
            )
            submitted = st.form_submit_button("로그인", use_container_width=True)
        if submitted:
            user, msg = authenticate_user(client, login_id, password)
            if user:
                st.session_state.user_id = user["id"]
                st.session_state.login_id = user["login_id"]
                reset_local_session()
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        with st.form("signup_form"):
            login_id = st.text_input("아이디 (login_id)", key="signup_login_id")
            password = st.text_input(
                "비밀번호", type="password", key="signup_password"
            )
            password2 = st.text_input(
                "비밀번호 확인", type="password", key="signup_password2"
            )
            submitted = st.form_submit_button("회원가입", use_container_width=True)
        if submitted:
            if password != password2:
                st.error("비밀번호가 일치하지 않습니다.")
            else:
                ok, msg = register_user(client, login_id, password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


def render_sidebar(client: Client | None) -> str:
    user_id = current_user_id()

    st.sidebar.header("⚙️ 설정")
    st.sidebar.markdown(f"**LLM 모델:** `{MODEL_NAME}`")
    st.sidebar.markdown(f"**로그인:** `{st.session_state.login_id}`")

    if st.sidebar.button("로그아웃", use_container_width=True):
        clear_auth_state()
        st.rerun()

    rag_option = st.sidebar.radio(
        "RAG (PDF 검색) 선택",
        ["사용 안 함", "RAG 사용"],
        index=1,
    )

    uploaded_files = st.sidebar.file_uploader(
        "PDF 파일 업로드",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("파일 처리하기"):
        if client is None or not user_id:
            st.sidebar.error("로그인 및 Supabase 연결이 필요합니다.")
        elif not uploaded_files:
            st.sidebar.warning("업로드할 PDF 파일을 선택해 주세요.")
        else:
            with st.sidebar.spinner("PDF를 처리하고 Supabase에 저장하는 중..."):
                merged, error_message = process_pdf_files_to_supabase(
                    client,
                    st.session_state.session_id,
                    user_id,
                    uploaded_files,
                )
            if error_message:
                st.sidebar.error(error_message)
            else:
                st.session_state.processed_files = merged
                st.session_state.has_vectors = True
                err = auto_save_session(client)
                if err:
                    st.sidebar.warning(err)
                else:
                    st.sidebar.success(f"{len(merged)}개 PDF 처리·자동 저장 완료")

    if st.session_state.processed_files:
        st.sidebar.write("처리된 파일:")
        for file_name in st.session_state.processed_files:
            st.sidebar.write(f"- {file_name}")

    st.sidebar.divider()
    st.sidebar.subheader("📂 세션 관리")

    sessions: list[dict[str, Any]] = []
    session_labels: list[str] = ["(현재 작업 세션)"]
    label_to_id: dict[str, str | None] = {"(현재 작업 세션)": None}

    if client is not None and user_id:
        sessions = fetch_sessions(client, user_id)
        for row in sessions:
            label = f"{row.get('title') or '제목 없음'} · {str(row.get('id'))[:8]}"
            session_labels.append(label)
            label_to_id[label] = row["id"]

    current_index = 0
    if st.session_state.selected_session_id:
        for idx, label in enumerate(session_labels):
            if label_to_id.get(label) == st.session_state.selected_session_id:
                current_index = idx
                break

    selected_label = st.sidebar.selectbox(
        "세션 선택",
        session_labels,
        index=current_index,
        key="session_selectbox",
    )
    selected_id = label_to_id.get(selected_label)

    if (
        selected_id
        and selected_id != st.session_state.session_id
        and client is not None
        and user_id
    ):
        if st.session_state.get("_last_select_id") != selected_id:
            st.session_state._last_select_id = selected_id
            apply_loaded_session(client, selected_id, selected_label)
            st.rerun()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("세션저장", use_container_width=True):
            if client is None or not user_id:
                st.sidebar.error("로그인 및 Supabase 연결이 필요합니다.")
            else:
                with st.spinner("세션을 INSERT 저장하는 중..."):
                    title, err = manual_save_as_new_session(client)
                if err and not title:
                    st.sidebar.error(err)
                elif err:
                    st.sidebar.warning(f"'{title}' 저장됨 — {err}")
                else:
                    st.sidebar.success(f"새 세션 저장 완료: {title}")
                    st.rerun()

    with col2:
        if st.button("세션로드", use_container_width=True):
            if client is None or not user_id:
                st.sidebar.error("로그인 및 Supabase 연결이 필요합니다.")
            elif not selected_id:
                st.sidebar.warning("로드할 세션을 먼저 선택해 주세요.")
            else:
                apply_loaded_session(client, selected_id, selected_label)
                st.sidebar.success("세션을 불러왔습니다.")
                st.rerun()

    if st.sidebar.button("세션삭제", use_container_width=True):
        if client is None or not user_id:
            st.sidebar.error("로그인 및 Supabase 연결이 필요합니다.")
        else:
            target_id = selected_id or st.session_state.session_id
            if delete_session(client, target_id, user_id):
                st.sidebar.success("세션이 삭제되었습니다.")
                if target_id == st.session_state.session_id:
                    reset_local_session()
                st.session_state.selected_session_id = None
                st.session_state._last_select_id = None
                st.rerun()
            else:
                st.sidebar.error("세션 삭제에 실패했습니다.")

    if st.sidebar.button("화면초기화", use_container_width=True):
        reset_local_session()
        st.session_state._last_select_id = None
        st.sidebar.info("화면을 초기화했습니다. (DB 세션은 유지됩니다)")
        st.rerun()

    if st.sidebar.button("vectordb", use_container_width=True):
        st.session_state.show_vectordb = True

    if st.session_state.show_vectordb:
        st.sidebar.markdown("**Vector DB 파일명**")
        if client is None or not user_id:
            st.sidebar.warning("로그인 및 Supabase 연결이 필요합니다.")
        else:
            names = load_session_file_names(
                client, st.session_state.session_id, user_id
            )
            if not names:
                st.sidebar.write("(현재 세션에 저장된 문서 없음)")
            else:
                for name in names:
                    st.sidebar.write(f"- {name}")
        if st.sidebar.button("목록 닫기"):
            st.session_state.show_vectordb = False
            st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("현재 설정")
    st.sidebar.text(f"모델: {MODEL_NAME}")
    st.sidebar.text(f"RAG: {rag_option}")
    st.sidebar.text(f"사용자: {st.session_state.login_id}")
    st.sidebar.text(f"세션: {st.session_state.session_title}")
    st.sidebar.text(f"세션 ID: {st.session_state.session_id[:8]}…")
    st.sidebar.text(f"처리된 파일 수: {len(st.session_state.processed_files)}")
    st.sidebar.text(f"대화 기록 수: {len(st.session_state.chat_history)}")
    st.sidebar.text(f"저장된 세션 수: {len(sessions)}")

    return rag_option


def handle_user_input(
    user_query: str,
    rag_option: str,
    client: Client | None,
) -> None:
    user_id = current_user_id()
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    if not OPENAI_API_KEY:
        error_message = (
            "⚠️ OPENAI_API_KEY가 설정되지 않았습니다.\n\n"
            "Streamlit Cloud의 secrets 또는 "
            f"로컬 `.env`({ENV_PATH})에 키를 설정한 뒤 다시 시도해 주세요."
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": error_message}
        )
        update_conversation_memory(
            user_query, error_message, st.session_state.conversation_memory
        )
        with st.chat_message("assistant"):
            st.error(error_message)
        return

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            llm = get_llm()
            if rag_option == "RAG 사용":
                if client is None or not user_id:
                    warning_message = (
                        "⚠️ 로그인/Supabase 연결이 없어 RAG를 사용할 수 없습니다."
                    )
                    placeholder.warning(warning_message)
                    final_answer = warning_message
                elif not st.session_state.has_vectors and not session_has_vectors(
                    client, st.session_state.session_id, user_id
                ):
                    warning_message = (
                        "⚠️ RAG를 사용하려면 먼저 PDF 파일을 업로드하고 "
                        "'파일 처리하기' 버튼을 눌러 주세요."
                    )
                    placeholder.warning(warning_message)
                    final_answer = warning_message
                else:
                    st.session_state.has_vectors = True
                    final_answer = generate_rag_answer(
                        llm=llm,
                        client=client,
                        session_id=st.session_state.session_id,
                        user_id=user_id,
                        user_query=user_query,
                        conversation_memory=st.session_state.conversation_memory,
                        placeholder=placeholder,
                    )
            else:
                final_answer = generate_direct_llm_answer(
                    llm=llm,
                    user_query=user_query,
                    conversation_memory=st.session_state.conversation_memory,
                    placeholder=placeholder,
                )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": final_answer}
            )
            update_conversation_memory(
                user_query, final_answer, st.session_state.conversation_memory
            )

            if client is not None and user_id:
                err = auto_save_session(client, generate_title=True)
                if err:
                    LOGGER.warning(err)

        except Exception as exc:
            LOGGER.error("답변 생성 중 오류: %s", exc)
            friendly_message = (
                "답변 생성 중 오류가 발생했습니다. "
                "잠시 후 다시 시도해 주세요."
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": friendly_message}
            )
            update_conversation_memory(
                user_query, friendly_message, st.session_state.conversation_memory
            )
            placeholder.error(friendly_message)


def main() -> None:
    st.set_page_config(
        page_title="ENA RAG 챗봇",
        page_icon="📚",
        layout="wide",
    )

    refresh_secrets()
    inject_custom_css()
    init_session_state()
    render_header()

    key_msg = missing_keys_message()
    if key_msg:
        st.warning(key_msg)

    client = get_supabase_client()
    if client is None and SUPABASE_URL and SUPABASE_ANON_KEY:
        st.error("Supabase 클라이언트 초기화에 실패했습니다.")

    if not st.session_state.user_id:
        render_auth_page(client)
        return

    rag_option = render_sidebar(client)
    render_chat_history()

    user_query = st.chat_input("메시지를 입력하세요...")
    if user_query:
        handle_user_input(user_query, rag_option, client)


if __name__ == "__main__":
    main()
