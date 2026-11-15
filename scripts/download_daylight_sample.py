"""
download_daylight_sample.py — Project D daylight segment downloader.

Extracts only the selected HEVC member from a remote ZIP archive using HTTP
Range requests, without downloading the full archive.

Design notes
------------
- RangeReader, URL and TARGET_MEMBER are imported from inspect_remote_zip.py.
- _PrefetchReader wraps RangeReader with a 1 MiB sequential buffer so that the
  many small internal reads zipfile makes are batched into fewer Range requests.
  Without it, each ~8–64 KiB decompressor read would become a separate request,
  quickly exhausting the 64-request budget.
- Extraction streams through zf.open() in 1 MiB chunks; reading to EOF causes
  zipfile to verify the CRC32 automatically.
- The member is written to an exclusively-created staging file.  After size and
  CRC pass, os.link(STAGING_PATH, OUTPUT_PATH) creates a hard link to the
  output path — the call fails with FileExistsError if the destination already
  exists, so no pre-check is needed and no existing file is ever overwritten.
  The staging path is then unlinked.  On any failure the staging file created
  by this run is removed; the output file is never touched.

Safety limits
-------------
- 40 MiB cumulative network-body budget.
- 64 Range requests.
- 60-second per-request timeout.
- Refuses to overwrite an existing output or staging file.
- Validates compressed size, uncompressed size, compression method, and
  encryption flag before any extraction begins.
"""

import os
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Import shared components
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from inspect_remote_zip import RangeReader, URL, TARGET_MEMBER  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH  = Path("data/raw/daylight-2018-07-29-segment-7.hevc")
STAGING_PATH = Path(str(OUTPUT_PATH) + ".part")

MAX_BYTES    = 40 * (1 << 20)  # 40 MiB network-body budget
MAX_REQUESTS = 64
TIMEOUT      = 60               # seconds per request

# Verified by inspect_remote_zip.py
EXPECTED_COMPRESS_SIZE   = 37_525_770
EXPECTED_UNCOMPRESS_SIZE = 37_514_330
EXPECTED_COMPRESS_METHOD = 8    # deflate

STREAM_CHUNK = 1 << 20          # 1 MiB read chunks from ZipExtFile
FETCH_CHUNK  = 1 << 20          # 1 MiB prefetch window for RangeReader


# ---------------------------------------------------------------------------
# Sequential prefetch buffer
# ---------------------------------------------------------------------------

class _PrefetchReader:
    """
    Wraps a RangeReader with a local, lazily-compacted prefetch buffer.

    zipfile.ZipFile reads the underlying fileobj in many small increments
    (typically 8–64 KiB for the deflate decompressor).  Each such increment
    without buffering would generate a separate Range request.  This class
    batches sequential data into FETCH_CHUNK-sized Range requests, serving
    zipfile's small reads from the in-memory buffer.

    The buffer is invalidated on any non-sequential seek.  Its front is
    compacted lazily once consumed bytes exceed FETCH_CHUNK, giving amortised
    O(1) overhead per byte without unbounded memory growth.
    """

    def __init__(self, reader: RangeReader, chunk_size: int = FETCH_CHUNK) -> None:
        self._r         = reader
        self._chunk     = chunk_size
        self._buf       = bytearray()
        self._buf_start = 0     # absolute archive pos of buf[0]
        self._pos       = 0     # consumer's current absolute position

    # --- File-like interface (required by zipfile.ZipFile) -------------------

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            new_pos = offset
        elif whence == 1:
            new_pos = self._pos + offset
        elif whence == 2:
            # self._r.seek(0, 2) discovers (and caches) the archive size and
            # returns it; _r's internal position is updated but that's harmless
            # because _r is always explicitly seeked before each read().
            new_pos = self._r.seek(0, 2) + offset
        else:
            raise ValueError(f"Invalid whence: {whence!r}")

        if new_pos < 0:
            raise ValueError(
                f"Seek would produce a negative position ({new_pos})."
            )

        # Invalidate the buffer when the new position falls outside the window
        buf_end = self._buf_start + len(self._buf)
        if new_pos < self._buf_start or new_pos > buf_end:
            self._buf.clear()
            self._buf_start = new_pos

        self._pos = new_pos
        return self._pos

    def read(self, n: int = -1) -> bytes:
        rel       = self._pos - self._buf_start   # consumer's offset into buf
        available = len(self._buf) - rel           # bytes ready in the buffer

        # Unbounded read: delegate directly to RangeReader (which enforces its
        # own budget check before making any request).
        if n is None or n < 0:
            self._r.seek(self._pos)
            data = self._r.read(-1)
            self._buf.clear()
            self._buf_start = self._pos + len(data)
            self._pos = self._buf_start
            return bytes(data)

        if n == 0:
            return b""

        # Fetch the next chunk when the buffer cannot cover the request.
        if n > available:
            fetch_from = self._buf_start + len(self._buf)
            to_fetch   = max(self._chunk, n - available)
            self._r.seek(fetch_from)
            extra = self._r.read(to_fetch)  # may return < to_fetch at EOF
            self._buf.extend(extra)
            available = len(self._buf) - rel

        take   = min(n, available)
        result = bytes(self._buf[rel : rel + take])
        self._pos += take

        # Lazy compaction: drop consumed front bytes once they exceed chunk size
        consumed = self._pos - self._buf_start
        if consumed >= self._chunk:
            del self._buf[:consumed]
            self._buf_start = self._pos

        return result


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def _validate_member(info: zipfile.ZipInfo) -> str | None:
    """
    Return an error string if *info* fails any pre-extraction check,
    or None if all checks pass.
    """
    if bool(info.flag_bits & 0x1):
        return "member is encrypted"
    if info.compress_type != EXPECTED_COMPRESS_METHOD:
        return (
            f"unexpected compression method {info.compress_type}; "
            f"expected deflate ({EXPECTED_COMPRESS_METHOD})"
        )
    if info.compress_size != EXPECTED_COMPRESS_SIZE:
        return (
            f"compressed size {info.compress_size:,} B does not match "
            f"expected {EXPECTED_COMPRESS_SIZE:,} B"
        )
    if info.file_size != EXPECTED_UNCOMPRESS_SIZE:
        return (
            f"uncompressed size {info.file_size:,} B does not match "
            f"expected {EXPECTED_UNCOMPRESS_SIZE:,} B"
        )
    return None


def download() -> int:
    """Orchestrate the download; returns an exit code."""

    # --- Pre-flight: refuse to overwrite -------------------------------------
    if OUTPUT_PATH.exists():
        print(
            f"ERROR: Output already exists: '{OUTPUT_PATH}'. "
            "Remove it manually before running.",
            file=sys.stderr,
        )
        return 1

    if STAGING_PATH.exists():
        print(
            f"ERROR: Staging file already exists: '{STAGING_PATH}'. "
            "Remove it manually before running.",
            file=sys.stderr,
        )
        return 1

    # --- Open archive --------------------------------------------------------
    reader   = RangeReader(url=URL, max_bytes=MAX_BYTES,
                            max_requests=MAX_REQUESTS, timeout=TIMEOUT)
    prefetch = _PrefetchReader(reader, chunk_size=FETCH_CHUNK)

    print(f"Archive URL : {URL}")
    print(f"Target      : {TARGET_MEMBER}")
    print()

    try:
        zf = zipfile.ZipFile(prefetch)
    except zipfile.BadZipFile as exc:
        print(f"ERROR: Could not parse ZIP directory: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"ERROR (network/budget): {exc}", file=sys.stderr)
        return 3

    with zf:
        # --- Locate member ---------------------------------------------------
        matches = [
            info for info in zf.infolist()
            if info.filename == TARGET_MEMBER
            or info.filename.endswith("/" + TARGET_MEMBER)
        ]

        if len(matches) != 1:
            print(
                f"ERROR: Expected exactly 1 matching member; found {len(matches)}.",
                file=sys.stderr,
            )
            return 4

        info = matches[0]

        # --- Validate member properties before extracting --------------------
        err = _validate_member(info)
        if err:
            print(f"ERROR: Pre-extraction validation failed: {err}.", file=sys.stderr)
            return 5

        print(f"Member path               : {info.filename}")
        print(f"Compressed size (bytes)   : {info.compress_size:,}")
        print(f"Uncompressed size (bytes) : {info.file_size:,}")
        print(f"Compression method        : deflate")
        print(f"Encrypted                 : False")
        print(f"Local-header offset (abs) : {info.header_offset:,}")
        print()

        # --- Stream to staging file ------------------------------------------
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        staging_created = False
        total_written   = 0
        exit_code       = 0

        try:
            # Exclusive creation: raises FileExistsError if staging already exists
            try:
                staging_fh = open(STAGING_PATH, "xb")
                staging_created = True
            except FileExistsError:
                print(
                    f"ERROR: Staging file appeared unexpectedly: '{STAGING_PATH}'.",
                    file=sys.stderr,
                )
                return 1

            print(f"Extracting to : {STAGING_PATH} …")

            try:
                with staging_fh:
                    with zf.open(info) as member:
                        while True:
                            chunk = member.read(STREAM_CHUNK)
                            if not chunk:
                                # EOF reached — zipfile has verified CRC32
                                break
                            if total_written + len(chunk) > EXPECTED_UNCOMPRESS_SIZE:
                                raise RuntimeError("Extracted data exceeds expected size")
                            staging_fh.write(chunk)
                            total_written += len(chunk)
            except zipfile.BadZipFile as exc:
                print(
                    f"ERROR: CRC or decompression failure: {exc}",
                    file=sys.stderr,
                )
                exit_code = 6
                return exit_code
            except RuntimeError as exc:
                print(f"ERROR (network/budget): {exc}", file=sys.stderr)
                exit_code = 3
                return exit_code

            # --- Verify extracted size ---------------------------------------
            if total_written != EXPECTED_UNCOMPRESS_SIZE:
                print(
                    f"ERROR: Extracted size mismatch: "
                    f"wrote {total_written:,} B, "
                    f"expected {EXPECTED_UNCOMPRESS_SIZE:,} B.",
                    file=sys.stderr,
                )
                exit_code = 6
                return exit_code

            # --- Publish: hard-link staging → output, then unlink staging ----
            # os.link raises FileExistsError if OUTPUT_PATH already exists;
            # the call provides the overwrite guard without a separate check.
            # staging_created remains True until STAGING_PATH.unlink() succeeds
            # so that the finally block cleans up staging on any failure here.
            try:
                os.link(STAGING_PATH, OUTPUT_PATH)
            except FileExistsError:
                print(
                    f"ERROR: Output already exists at '{OUTPUT_PATH}' — "
                    "os.link refused to overwrite it.",
                    file=sys.stderr,
                )
                return 1

            # Hard link created; both paths now point to the same inode.
            # Unlink the staging path to leave only OUTPUT_PATH.
            STAGING_PATH.unlink()
            staging_created = False       # staging gone; output is live

            exit_code = 0

        finally:
            # Remove staging only if this run created it and it still exists.
            # Never touch the output file.
            if staging_created and STAGING_PATH.exists():
                print(
                    f"Cleaning up staging file after failure: {STAGING_PATH}",
                    file=sys.stderr,
                )
                STAGING_PATH.unlink(missing_ok=True)

        if exit_code != 0:
            return exit_code

    # --- Report --------------------------------------------------------------
    st = reader.stats()
    print(f"Saved                     : {OUTPUT_PATH.resolve()}")
    print(f"Extracted bytes           : {total_written:,}")
    print(f"Range requests made       : {st['requests_made']}")
    print(f"Response bytes fetched    : {st['bytes_fetched']:,}")

    return 0


def main() -> None:
    sys.exit(download())


if __name__ == "__main__":
    main()
