"""
inspect_remote_zip.py — Project D remote ZIP directory inspector.

Inspects the central directory of a remote ZIP (including ZIP64) over HTTP,
using Range requests, without downloading the archive body.

A seekable RangeReader translates every seek/read into a precisely bounded HTTP
Range request so that zipfile.ZipFile sees a normal file-like object whose
positions map 1-to-1 to absolute byte offsets in the remote archive.

The previous approach of passing only the final N bytes through BytesIO failed
because ZIP64 offsets in the central directory refer to absolute positions in
the original archive; those positions are inaccessible when the object starts
at a non-zero origin.

Safety limits
-------------
- HTTP 206 required; response body validated against Content-Range.
- Maximum 1 MiB cumulative response-body data.
- Maximum 32 Range requests.
- 60-second timeout per request.
- Unbounded reads (n = -1) are rejected.
"""

import re
import sys
import urllib.request
import zipfile

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

URL = (
    "https://huggingface.co/datasets/commaai/comma2k19"
    "/resolve/main/compression_challenge/test_videos.zip"
)

TARGET_MEMBER = "b0c9d2329ad1606b|2018-07-29--11-17-20/7/video.hevc"

MAX_BYTES    = 1_048_576   # 1 MiB cumulative response-body budget
MAX_REQUESTS = 32          # maximum Range requests
TIMEOUT      = 60          # seconds per individual request

COMPRESSION_NAMES = {
    0:  "stored (0)",
    8:  "deflate (8)",
    12: "bzip2 (12)",
    14: "lzma (14)",
}


# ---------------------------------------------------------------------------
# Seekable HTTP Range reader
# ---------------------------------------------------------------------------

class RangeReader:
    """
    A seekable, readable file-like object backed by HTTP Range requests.

    Every call to ``read(n)`` emits exactly one Range request covering the n
    bytes starting at the current logical position.  The logical position maps
    directly to the absolute byte offset in the remote file, so zipfile's
    internal offset arithmetic works without adjustment.

    Parameters
    ----------
    url : str
        URL of the remote file.  Must support Range requests.
    max_bytes : int
        Maximum cumulative response-body bytes across all requests.
    max_requests : int
        Maximum number of Range requests permitted.
    timeout : int | float
        Per-request socket timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        max_bytes: int = MAX_BYTES,
        max_requests: int = MAX_REQUESTS,
        timeout: float = TIMEOUT,
    ) -> None:
        self._url          = url
        self._max_bytes    = max_bytes
        self._max_requests = max_requests
        self._timeout      = timeout

        self._pos: int         = 0
        self._size: int | None = None   # discovered from first Content-Range
        self._bytes_fetched: int = 0
        self._requests_made: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_size(self) -> int:
        """Return the remote file size, fetching 1 byte if not yet known."""
        if self._size is not None:
            return self._size
        # Fetch a single byte purely to obtain the Content-Range total.
        self._range_request(0, 0)
        assert self._size is not None
        return self._size

    def _range_request(self, start: int, end: int) -> bytes:
        """
        Fetch bytes [start, end] inclusive via a single Range request.

        Validates HTTP 206, Content-Range header, and response body length.
        Enforces the byte and request budgets.

        Returns the fetched bytes.
        Raises RuntimeError on any protocol or budget violation.
        """
        n_requested = end - start + 1
        # Reserve n_requested + 1 bytes in the budget: the extra byte lets us
        # detect an over-long response before accepting it.
        budget_charge = n_requested + 1

        if self._requests_made >= self._max_requests:
            raise RuntimeError(
                f"Request budget exhausted: limit is {self._max_requests} requests."
            )
        if self._bytes_fetched + budget_charge > self._max_bytes:
            raise RuntimeError(
                f"Byte budget would be exceeded: already fetched "
                f"{self._bytes_fetched} B, need {budget_charge} B of headroom, "
                f"limit is {self._max_bytes} B."
            )

        req = urllib.request.Request(
            self._url,
            headers={"Range": f"bytes={start}-{end}"},
        )
        # Increment before urlopen so that failed/timed-out requests are counted.
        self._requests_made += 1
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status != 206:
                    raise RuntimeError(
                        f"Expected HTTP 206 Partial Content; got {resp.status}. "
                        "The server may not support Range requests."
                    )

                content_range = resp.headers.get("Content-Range", "")
                m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if not m:
                    raise RuntimeError(
                        f"Missing or malformed Content-Range header: {content_range!r}"
                    )

                cr_start, cr_end, cr_total = map(int, m.groups())

                if cr_start != start or cr_end != end:
                    raise RuntimeError(
                        f"Server returned range {cr_start}-{cr_end}; "
                        f"expected {start}-{end}."
                    )

                if self._size is None:
                    self._size = cr_total
                elif self._size != cr_total:
                    raise RuntimeError(
                        f"Inconsistent archive size: previously {self._size} B, "
                        f"now {cr_total} B."
                    )

                # Read exactly n_requested + 1 bytes; any more indicates an
                # over-long response and is rejected below.
                data = resp.read(n_requested + 1)

        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error during Range request: {exc}") from exc

        # Count all bytes received before deciding whether to accept them.
        self._bytes_fetched += len(data)
        if len(data) != n_requested:
            raise RuntimeError(
                f"Unexpected response body length: expected exactly {n_requested} B, "
                f"received {len(data)} B."
            )

        return data

    # ------------------------------------------------------------------
    # File-like interface required by zipfile.ZipFile
    # ------------------------------------------------------------------

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
            new_pos = self._discover_size() + offset
        else:
            raise ValueError(f"Invalid whence value: {whence!r}")
        if new_pos < 0:
            raise ValueError(
                f"Seek would produce a negative position ({new_pos}); "
                "current position is unchanged."
            )
        self._pos = new_pos
        return self._pos

    def read(self, n: int = -1) -> bytes:
        """
        Read bytes from the current position.

        If *n* is -1 or None, reads all remaining bytes to EOF — but only if
        they fit within the remaining byte budget.  If they do not fit, raises
        RuntimeError before making any request.
        """
        size = self._discover_size()
        remaining = max(0, size - self._pos)

        if n is None or n < 0:
            # Unbounded read: allow only when remaining bytes fit the budget.
            budget_left = self._max_bytes - self._bytes_fetched
            if remaining > budget_left:
                raise RuntimeError(
                    f"Unbounded read() refused: {remaining} B remaining to EOF "
                    f"exceeds remaining byte budget of {budget_left} B."
                )
            n = remaining

        if n == 0 or remaining == 0:
            return b""

        end = min(self._pos + n - 1, size - 1)
        data = self._range_request(self._pos, end)
        self._pos += len(data)
        return data

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "requests_made": self._requests_made,
            "bytes_fetched": self._bytes_fetched,
            "archive_size":  self._size,
        }


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def _compression_name(method: int) -> str:
    return COMPRESSION_NAMES.get(method, f"unknown ({method})")


def inspect_remote_zip() -> int:
    """
    Open the remote ZIP via RangeReader, scan its central directory, and
    report the matching member.  Returns an exit code.
    """
    print(f"Archive URL : {URL}")
    print(f"Target      : {TARGET_MEMBER}")
    print()

    reader = RangeReader(url=URL, max_bytes=MAX_BYTES, max_requests=MAX_REQUESTS, timeout=TIMEOUT)

    try:
        zf = zipfile.ZipFile(reader)          # reads central directory only
    except zipfile.BadZipFile as exc:
        print(f"ERROR: zipfile could not parse the archive directory: {exc}", file=sys.stderr)
        st = reader.stats()
        print(
            f"  Requests made : {st['requests_made']}\n"
            f"  Bytes fetched : {st['bytes_fetched']}\n"
            f"  Archive size  : {st['archive_size']}",
            file=sys.stderr,
        )
        return 2
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    with zf:
        entries  = zf.infolist()
        st       = reader.stats()

        print(f"Archive size           : {st['archive_size']:,} bytes")
        print(f"Central directory entries: {len(entries)}")
        print(f"Range requests made    : {st['requests_made']}")
        print(f"Response bytes fetched : {st['bytes_fetched']:,}")
        print()

        # Match exact path or path ending with /<TARGET_MEMBER>
        matches = [
            item for item in entries
            if item.filename == TARGET_MEMBER
            or item.filename.endswith("/" + TARGET_MEMBER)
        ]

        if not matches:
            print(f"No entry found matching '{TARGET_MEMBER}'.")
            print()
            print("First 10 paths in archive:")
            for item in entries[:10]:
                print(f"  {item.filename}")
            return 1

        for item in matches:
            encrypted = bool(item.flag_bits & 0x1)
            print("--- Matching entry ---")
            print(f"  Path                      : {item.filename}")
            print(f"  Compressed size (bytes)   : {item.compress_size:,}")
            print(f"  Uncompressed size (bytes) : {item.file_size:,}")
            print(f"  Compression method        : {_compression_name(item.compress_type)}")
            print(f"  Encrypted                 : {encrypted}")
            print(f"  Local-header offset (abs) : {item.header_offset:,}")

    return 0


def main() -> None:
    sys.exit(inspect_remote_zip())


if __name__ == "__main__":
    main()
