"""
Import images from data/gallery/** → Postgres
=================================================================

What this script does
---------------------
- Walks a folder like: data/gallery/<PERSON_NAME>/*.jpg|*.png
- For each image it inserts a row into Postgres:
    nik  = "1","2","3",...   (running number as TEXT)
    name = <PERSON_NAME>     (top-level folder name under --root)
    face = raw image bytes   (BYTEA)

It can also auto-create the table (id BIGSERIAL, nik TEXT, name TEXT, face BYTEA).

Example
-------
python -m app.import_gallery_to_postgres ^
  --root "./data/gallery" ^
  --dsn  "postgresql://postgres:12345678@localhost:5432/mydb" ^
  --schema public --table faces ^
  --create-table

Tips
------------------
- Make sure your folder structure is:
      data/gallery/Alice/img1.jpg
      data/gallery/Alice/img2.png
      data/gallery/Bob/face.jpg
- If you see "skipped", it means the file was too small or couldn't be read.
- Use --min-bytes to skip tiny corrupted files (default 100 bytes).
"""

import argparse
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_batch


# We accept common image extensions (case-insensitive)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def parse_args():
    """
    Keep CLI simple and clear.
    """
    ap = argparse.ArgumentParser(description="Import gallery images into Postgres")
    ap.add_argument("--root", required=True, help="Root folder of gallery, e.g. ./data/gallery")
    ap.add_argument("--dsn", required=True, help="postgresql://user:pass@host:port/dbname")
    ap.add_argument("--schema", default="public", help="Target schema (default: public)")
    ap.add_argument("--table", default="faces", help="Target table (default: faces)")
    ap.add_argument("--batch", type=int, default=1000, help="Insert batch size (default: 1000)")
    ap.add_argument("--min-bytes", type=int, default=100, help="Skip files smaller than this many bytes")
    ap.add_argument("--create-table", action="store_true", help="Create table if it does not exist")
    return ap.parse_args()


def create_table_sql(schema, table):
    """
    Create a simple faces table if missing.
    """
    return f"""
    CREATE TABLE IF NOT EXISTS {schema}.{table} (
      id   BIGSERIAL PRIMARY KEY,
      nik  TEXT,
      name TEXT,
      face BYTEA
    );
    """


def find_images(root: Path):
    """
    Yield (folder_name, file_path) for each image under root.
    folder_name = the first-level folder under --root (person's name).
    """
    for p in sorted(root.rglob("*")):
        # check extension (case-insensitive set already covers both)
        if p.is_file() and p.suffix in IMG_EXTS:
            try:
                parts = p.relative_to(root).parts  # e.g. ('Alice', 'img1.jpg')
                folder_name = parts[0] if len(parts) > 0 else ""
            except Exception:
                # Fallback if relative_to fails (should be rare)
                folder_name = p.parent.name
            yield folder_name, p


def main():
    args = parse_args()
    root = Path(args.root)

    if not root.exists():
        print(f"❌ [error] root not found: {root}")
        sys.exit(1)
    if not root.is_dir():
        print(f"❌ [error] root is not a directory: {root}")
        sys.exit(1)

    # We'll insert into columns (nik, name, face)
    cols = ["nik", "name", "face"]
    placeholders = ",".join(["%s"] * len(cols))
    sql_insert = f"INSERT INTO {args.schema}.{args.table} ({','.join(cols)}) VALUES ({placeholders})"

    total = inserted = skipped = 0
    nik_counter = 1  # we number rows as strings "1","2",...

    try:
        with psycopg2.connect(args.dsn) as conn:
            conn.autocommit = False  # explicit commits after each batch
            with conn.cursor() as cur:
                # Create table if asked
                if args.create_table:
                    cur.execute(create_table_sql(args.schema, args.table))
                    conn.commit()
                    print(f"[ok] ensured table {args.schema}.{args.table} exists")

                batch = []
                for person_name, img_path in find_images(root):
                    total += 1
                    try:
                        data = img_path.read_bytes()
                    except Exception as e:
                        # Could not read file (permissions? broken file?)
                        skipped += 1
                        # Show a tiny hint every now and then
                        if skipped <= 5:
                            print(f"[warn] could not read file: {img_path} ({e})")
                        continue

                    # Skip tiny/corrupt files
                    if not data or len(data) < args.min_bytes:
                        skipped += 1
                        continue

                    nik = str(nik_counter)  # "1","2","3",...
                    nik_counter += 1

                    # Use psycopg2.Binary for BYTEA fields
                    batch.append((nik, person_name, psycopg2.Binary(data)))

                    # Flush batch to DB
                    if len(batch) >= args.batch:
                        execute_batch(cur, sql_insert, batch, page_size=args.batch)
                        inserted += len(batch)
                        batch.clear()
                        conn.commit()
                        print(f"[info] inserted={inserted} / scanned={total} (skipped={skipped})")

                # Final flush (whatever is left)
                if batch:
                    execute_batch(cur, sql_insert, batch, page_size=args.batch)
                    inserted += len(batch)
                    conn.commit()

        print(f"[done] scanned={total}, inserted={inserted}, skipped={skipped}")

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
