"""
Import FACES.dbf → Postgres
===============================================

What this script does
---------------------
- Opens a .DBF file (optionally using its .DBT memo sidecar).
- Reads two REQUIRED fields:
    • nik   (ID string)
    • face  (image as raw bytes OR base64/data: URL)
- Optionally reads a 3rd field if present:
    • name  (person's name)  ← if missing in the DBF, we still insert NULL
- Decodes/cleans the face bytes safely and skips tiny/invalid images.
- Inserts rows into Postgres in batches.

Why “name” even if DBF doesn’t have it?
---------------------------------------
We **create the table with a `name TEXT` column** anyway, and insert NULL when
the DBF has no name column mapped. This keeps the DB schema future-proof.

Example (PowerShell / Docker)
-----------------------------
docker run --rm --network frbf-net -v "${PWD}:/workspace" frbf-import:cpu `
  python /app/import_dbf_to_postgres.py `
  --dbf "/workspace/data/FACES.dbf" `
  --dsn "postgresql://postgres:12345678@pgvec:5432/mydb" `
  --schema public --table faces `
  --map nik=NIK face=FACE `
  --create-table

Notes
-----
- Add `name=NAMECOL` to `--map` if your DBF has a name column.
- If you don’t have FACES.dbt (memo file) and DBF requires it, you can set
  `ignore_missing_memofile=True` in the DBF() call (see code below).
"""

import argparse
import base64
import re
import sys

from dbfread import DBF, FieldParser
import psycopg2
from psycopg2.extras import execute_batch


# ---------- tolerant parser: bad dates/numbers -> None ----------
class SafeFieldParser(FieldParser):
    def parseD(self, field, data):  # Date
        try:
            return super().parseD(field, data)
        except Exception:
            return None

    def parseN(self, field, data):  # Numeric
        try:
            return super().parseN(field, data)
        except Exception:
            return None

    def parseF(self, field, data):  # Float
        try:
            return super().parseF(field, data)
        except Exception:
            return None

    def parseL(self, field, data):  # Logical
        try:
            return super().parseL(field, data)
        except Exception:
            return None


def parse_args():
    """
    Minimal CLI flags.
    """
    ap = argparse.ArgumentParser(description="Import faces from .DBF into Postgres")
    ap.add_argument("--dbf", required=True, help="Path to FACES.dbf (FACES.dbt beside it if needed)")
    ap.add_argument("--dsn", required=True, help="postgresql://user:pass@host:port/dbname")
    ap.add_argument("--schema", default="public", help="Target schema (default: public)")
    ap.add_argument("--table", default="faces", help="Target table (default: faces)")
    ap.add_argument(
        "--map",
        nargs="+",
        required=True,
        help="python_col=DBF_COL pairs; MUST include nik=... and face=... ; "
             "OPTIONAL name=... if available",
    )
    ap.add_argument("--batch", type=int, default=1000, help="Batch size for inserts (default: 1000)")
    ap.add_argument("--create-table", action="store_true", help="Create table if missing")
    return ap.parse_args()


def build_mapping(pairs):
    """
    Turn ["nik=NIK", "face=FACE", "name=NAMECOL"] into {"nik": "NIK", "face": "FACE", "name": "NAMECOL"}
    """
    m = {}
    for p in pairs:
        k, v = p.split("=", 1)
        m[k.strip().lower()] = v.strip()
    return m


def create_table_sql(schema, table):
    """
    Create a table with id, nik, **name**, face — note: we include name column even when DBF lacks it.
    """
    return f"""
    CREATE TABLE IF NOT EXISTS {schema}.{table} (
      id   BIGSERIAL PRIMARY KEY,
      nik  TEXT,
      name TEXT,
      face BYTEA
    );
    """


# ---------- base64 helpers ----------
_B64_CLEAN = re.compile(r"[^A-Za-z0-9+/=]+")


def _strip_data_url(s: str) -> str:
    """
    Remove data URL prefix like 'data:image/jpeg;base64,.....' → keep only the base64 part.
    """
    s = s.strip()
    if s.lower().startswith("data:image/"):
        parts = s.split(",", 1)
        if len(parts) == 2:
            return parts[1]
    return s


def decode_face(val):
    """
    Convert a DBF 'face' field to raw image bytes (or None if invalid/tiny).
    Accepts:
      - bytes/bytearray (already raw)
      - base64 string (optionally with data URL prefix)
    """
    if val is None:
        return None

    # Already bytes?
    if isinstance(val, (bytes, bytearray)):
        b = bytes(val)
        return b if len(b) > 100 else None  # skip tiny junk

    # Treat as string/base64
    s = str(val).strip()
    if not s:
        return None
    s = _strip_data_url(s)

    # Base64: clean, pad to multiple of 4, decode
    rem = len(s) % 4
    if rem:
        s += "=" * (4 - rem)
    try:
        raw = base64.b64decode(_B64_CLEAN.sub("", s), validate=False)
    except Exception:
        return None
    return raw if raw and len(raw) > 100 else None


def main():
    args = parse_args()
    m = build_mapping(args.map)

    # Check required mappings
    if "nik" not in m or "face" not in m:
        print("❌ Please include BOTH mappings: nik=... and face=...")
        print("   (Optional) You may also include name=... if your DBF has it.")
        sys.exit(1)

    # Setup DBF reader (tolerant parser). If you don't have .dbt but DBF expects it,
    # you may set ignore_missing_memofile=True below.
    table = DBF(
        args.dbf,
        load=False,
        ignore_missing_memofile=False,   # set True if you don't have FACES.dbt
        parserclass=SafeFieldParser,
        char_decode_errors="ignore",
    )

    # We always insert into (nik, name, face) — name can be NULL
    cols = ["nik", "name", "face"]
    placeholders = ",".join(["%s"] * len(cols))
    sql = f"INSERT INTO {args.schema}.{args.table} ({','.join(cols)}) VALUES ({placeholders})"

    total = inserted = skipped = 0

    try:
        with psycopg2.connect(args.dsn) as conn:
            conn.autocommit = False
            with conn.cursor() as cur:
                # Create table if requested
                if args.create_table:
                    cur.execute(create_table_sql(args.schema, args.table))
                    conn.commit()
                    print(f"[ok] ensured table {args.schema}.{args.table} exists (with name column)")

                batch = []
                for r in table:
                    total += 1

                    # nik (required)
                    nik = r.get(m["nik"])
                    if nik is not None:
                        nik = str(nik).strip()

                    # name (optional)
                    name = None
                    if "name" in m:
                        name_val = r.get(m["name"])
                        if name_val is not None:
                            name = str(name_val).strip() or None  # empty -> None

                    # face (required, decoded)
                    face_raw = r.get(m["face"])
                    face_bytes = decode_face(face_raw)
                    if not face_bytes:
                        skipped += 1
                        continue

                    # Add to batch (face as psycopg2.Binary)
                    batch.append((nik, name, psycopg2.Binary(face_bytes)))

                    # Flush batch
                    if len(batch) >= args.batch:
                        execute_batch(cur, sql, batch, page_size=args.batch)
                        inserted += len(batch)
                        batch.clear()
                        conn.commit()
                        print(f"[info] inserted={inserted} / scanned={total} (skipped={skipped})")

                # Final flush
                if batch:
                    execute_batch(cur, sql, batch, page_size=args.batch)
                    inserted += len(batch)
                    conn.commit()

        print(f"[done] scanned={total}, inserted={inserted}, skipped={skipped}")
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        sys.exit(2)
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        sys.exit(3)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(4)


if __name__ == "__main__":
    main()
