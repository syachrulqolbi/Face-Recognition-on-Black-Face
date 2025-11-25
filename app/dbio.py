"""
DB I/O helpers
- Safe schema/table validation
- Parameterized queries for dynamic values
- Consistent BYTEA -> bytes conversion
"""

import re
from typing import Optional, Tuple, List

import psycopg2
from app.config import SimpleConfig

# --- Simple identifier guard for schema/table used in f-strings ---
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _validate_ident(ident: str, what: str) -> None:
    """
    Ensure schema/table look like plain SQL identifiers.
    This is a friendly guard (not full SQL injection protection).
    """
    if not _ID_RE.match(ident or ""):
        raise ValueError(
            f"Invalid {what!r}: {ident!r}. Use letters, digits, and underscores only "
            "(and start with a letter or underscore)."
        )

def _to_bytes(b) -> bytes:
    """psycopg2 may return memoryview for BYTEA; normalize to bytes."""
    return bytes(b) if isinstance(b, memoryview) else b

# ---------------------------------------------------------------------
# Public API (kept identical to your original signatures)
# ---------------------------------------------------------------------

def fetch_gallery_rows(cfg: SimpleConfig) -> List[Tuple[str, Optional[str], bytes]]:
    """
    Returns rows for building FAISS (nik, name, face_bytes).

    Output: List of (nik: str, name: Optional[str], face_bytes: bytes)
    """
    _validate_ident(cfg.DB_SCHEMA, "schema")
    _validate_ident(cfg.DB_TABLE, "table")

    sql = (
        f"SELECT nik, name, face "
        f"FROM {cfg.DB_SCHEMA}.{cfg.DB_TABLE} "
        f"WHERE face IS NOT NULL"
    )

    try:
        with psycopg2.connect(cfg.DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                out: List[Tuple[str, Optional[str], bytes]] = []
                for nik, name, face in rows:
                    out.append((nik, name, _to_bytes(face)))
                return out
    except psycopg2.Error as e:
        print(f"[db] fetch_gallery_rows failed: {e}")
        raise

def fetch_one_face_image_for_label(cfg: SimpleConfig, label: str) -> Optional[bytes]:
    """
    Try to fetch ONE sample face for a given label (your labels are typically 'name').
    Fallback to nik if name doesn't match.

    Returns: image bytes (BYTEA) or None
    """
    _validate_ident(cfg.DB_SCHEMA, "schema")
    _validate_ident(cfg.DB_TABLE, "table")

    sql_by_name = (
        f"SELECT face FROM {cfg.DB_SCHEMA}.{cfg.DB_TABLE} "
        f"WHERE name = %s AND face IS NOT NULL "
        f"LIMIT 1"
    )
    sql_by_nik = (
        f"SELECT face FROM {cfg.DB_SCHEMA}.{cfg.DB_TABLE} "
        f"WHERE nik = %s AND face IS NOT NULL "
        f"LIMIT 1"
    )

    try:
        with psycopg2.connect(cfg.DB_DSN) as conn:
            with conn.cursor() as cur:
                # 1) Exact name match
                cur.execute(sql_by_name, (label,))
                row = cur.fetchone()
                if row and row[0]:
                    return _to_bytes(row[0])

                # 2) Fallback: treat label as NIK
                cur.execute(sql_by_nik, (label,))
                row = cur.fetchone()
                if row and row[0]:
                    return _to_bytes(row[0])

                return None
    except psycopg2.Error as e:
        print(f"[db] fetch_one_face_image_for_label failed: {e}")
        raise

def fetch_nik_name_for_label(cfg: SimpleConfig, label: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (nik, name) for a label.
    We try name=label first, then nik=label. Returns (None, None) if not found.
    """
    _validate_ident(cfg.DB_SCHEMA, "schema")
    _validate_ident(cfg.DB_TABLE, "table")

    sql_name = (
        f"SELECT nik, name FROM {cfg.DB_SCHEMA}.{cfg.DB_TABLE} "
        f"WHERE name = %s LIMIT 1"
    )
    sql_nik = (
        f"SELECT nik, name FROM {cfg.DB_SCHEMA}.{cfg.DB_TABLE} "
        f"WHERE nik = %s LIMIT 1"
    )

    try:
        with psycopg2.connect(cfg.DB_DSN) as conn:
            with conn.cursor() as cur:
                # Try by name
                cur.execute(sql_name, (label,))
                row = cur.fetchone()
                if row:
                    nik = str(row[0]) if row[0] is not None else None
                    name = str(row[1]) if row[1] is not None else None
                    return nik, name

                # Fallback: try by NIK
                cur.execute(sql_nik, (label,))
                row = cur.fetchone()
                if row:
                    nik = str(row[0]) if row[0] is not None else None
                    name = str(row[1]) if row[1] is not None else None
                    return nik, name
    except psycopg2.Error as e:
        print(f"[db] fetch_nik_name_for_label failed: {e}")
        raise

    return None, None

# Optional: quick smoke test when run directly
if __name__ == "__main__":
    print("🔧 Quick DB test using SimpleConfig ...")
    cfg = SimpleConfig()
    try:
        rows = fetch_gallery_rows(cfg)
        print(f"✅ fetch_gallery_rows: got {len(rows)} rows.")
        if rows:
            nik, name, face = rows[0]
            print(f"  e.g. first row -> nik={nik!r}, name={name!r}, bytes={len(face)}")
    except Exception as e:
        print(f"❌ fetch_gallery_rows error: {e}")
