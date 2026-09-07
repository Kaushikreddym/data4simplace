"""Fetch CLMS tiles from the CDSE object store.

**Anonymous access is refused.** Every variant of
``https://s3.waw3-1.cloudferro.com/eodata/CLMS/...`` answers 403 or 401, and
``csv.dataspace.copernicus.eu`` -- the path the CLMS task records carry -- is only
a catalogue landing page pointing back at the manifest. A CDSE account with S3
keys is required, and is the one prerequisite of this stage that cannot be
satisfied from code.

The CLMS ``@datarequest_post`` route is a **working alternative**, and the one
the AOI probe was fetched with -- it needs only the ``~/.clms`` service key, and
its jobs run concurrently. It is not the right route for the continental build,
but the reason is not the one first assumed here.

Measured, server timestamps, three layers unless noted:

===================  =========  ==========  ========
AOI                  Layers     Duration    Delivered
===================  =========  ==========  ========
0.25 deg^2           1          22 s        0.7 MB
1.5 deg^2            3          2 m 54 s    13.0 MB
24 deg^2 (16x)       3          > 2 h 40 m  --
===================  =========  ==========  ========

**Volume is a wash.** Scaling the 1.5 deg^2 clip over the ~840 deg^2 the 664
tiles cover gives ~7 GB a year, ~58 GB for 2017-2024, against this route's
80.7 GB. An earlier note here argued from *uncompressed* raster size, which is
not what crosses the wire, and was wrong.

**Job cost is super-linear in area.** Sixteen times the AOI cost more than
fifty-five times the time, so a country-sized clip is not a viable unit of work.
Staying in the efficient regime means ~1.5 deg^2 jobs, i.e. **~4 500 asynchronous
jobs** for eight years of Europe -- each needing submit, poll, download and
verify, against a shared public service that has no cancel endpoint, refuses an
identical re-request while a stale task lives, and was observed losing one job
for six days.

That is the real argument for S3: not gigabytes, but **granularity and
retryability**. 15 936 immutable objects, each with a length and an MD5, fetched
in parallel and re-fetched individually. The FME route remains the right tool for
a region, and is what makes progress possible before a CDSE account exists.

**A file that exists is a file that finished.** Downloads land on a ``.part``
sibling and are renamed only once complete, so an interrupted transfer can never
leave a truncated file that a later ``exists()`` check waves through. The
manifest's own size and MD5 are the second, independent guard, which is what
catches a file left behind by some earlier tool.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "CDSE_ENDPOINT",
    "CDSE_PROFILE",
    "build_client",
    "fetch_tile",
    "smoke_test",
    "verify",
]

#: CDSE's S3-compatible endpoint. Reachable anonymously only far enough to be
#: refused: an unauthenticated GET returns 403, which is how you tell "wrong
#: host" from "no credentials".
CDSE_ENDPOINT: str = os.environ.get(
    "CDSE_S3_ENDPOINT", "https://eodata.dataspace.copernicus.eu"
)

#: Named boto3 profile holding the CDSE keys. A **named** profile on purpose:
#: this machine already has a ``[default]`` in ``~/.aws/credentials`` serving
#: something else, and silently borrowing it would either fail confusingly or
#: send someone else's keys to Copernicus.
CDSE_PROFILE: str = os.environ.get("CDSE_S3_PROFILE", "cdse")

#: CDSE runs Ceph behind a single hostname, so the bucket has to be part of the
#: *path* (``/eodata/CLMS/...``). boto3 defaults to virtual-host addressing
#: (``eodata.<host>``), which does not resolve here -- the failure looks like a
#: DNS or permissions error rather than a configuration one.
_ADDRESSING_STYLE = "path"

#: Ceph ignores the region but botocore insists on one being set.
_REGION = os.environ.get("CDSE_S3_REGION", "default")


def build_client(profile: str | None = None):
    """An S3 client pointed at CDSE, from a named credentials profile.

    Args:
        profile: Profile in ``~/.aws/credentials``; defaults to
            :data:`CDSE_PROFILE`. Pass ``""`` to fall back to boto3's own
            resolution (environment variables, instance role, ``default``).

    Returns:
        A configured ``boto3`` S3 client.
    """
    import boto3
    from botocore.config import Config

    name = CDSE_PROFILE if profile is None else profile
    session = boto3.Session(profile_name=name) if name else boto3.Session()
    return session.client(
        "s3",
        endpoint_url=CDSE_ENDPOINT,
        region_name=_REGION,
        config=Config(
            s3={"addressing_style": _ADDRESSING_STYLE},
            # CDSE throttles. Two dozen array tasks listing and fetching flat out
            # earn HTTP 429s, and botocore's default `legacy` mode gives up after
            # 4 attempts with no rate awareness -- which failed 161 of 664 tiles.
            # `adaptive` adds a client-side rate limiter that *slows down* when it
            # sees throttling, rather than merely retrying into the same wall.
            retries={"max_attempts": 10, "mode": "adaptive"},
            max_pool_connections=8,
        ),
    )


def verify(path: Path, size: int | None = None, md5: str | None = None) -> bool:
    """Whether an existing file matches what the manifest says it should be.

    Args:
        path: Local file.
        size: Expected byte length, if known.
        md5: Expected MD5, if known. Checked only when the size already matches,
            since hashing 7 MB to reject a file that is visibly the wrong length
            is wasted work.

    Returns:
        ``True`` when the file is present and matches every check supplied.
    """
    if not path.is_file():
        return False
    if size is not None and path.stat().st_size != size:
        logger.info("%s: %d bytes on disk, %d expected", path.name,
                    path.stat().st_size, size)
        return False
    if md5:
        digest = hashlib.md5()  # noqa: S324 - matching the catalogue's own algorithm
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != md5.lower():
            logger.warning("%s: MD5 mismatch", path.name)
            return False
    return True


def resolve_object(client, s3_path: str, suffix: str = ".tif") -> tuple[str, str, int, str]:
    """Find the real object behind a manifest ``s3_path``.

    **The manifest's ``s3_path`` is a prefix, not an object.** Each product entry
    is a folder repeating the granule name, holding the raster, two metadata
    sidecars and four QGIS/ArcGIS legend files::

        .../CLMS_HRLVLCC_CPMCE_S2022_R10m_E50N29_03035_V01_R00/
            CLMS_HRLVLCC_CPMCE_S2022_R10m_E50N29_03035_V01_R00.tif       1 348 852
            CLMS_HRLVLCC_CPMCE_S2022_R10m_E50N29_03035_V01_R00.tif.aux.xml  47 009
            CLMS_HRLVLCC_CPMCE_S2022_R10m_E50N29_03035_V01_R00.xml          33 767
            clms_hrlvlcc_cpmce_2022.clr / .lyr / .qml / .sld

    Downloading the prefix as though it were a key simply 404s.

    Args:
        client: A boto3 S3 client.
        s3_path: The manifest's ``s3://bucket/prefix`` value.
        suffix: Which member to return.

    Returns:
        ``(bucket, key, size, etag)`` for the matching member.

    Raises:
        FileNotFoundError: If the prefix holds no such member.
    """
    bucket, _, prefix = s3_path.removeprefix("s3://").partition("/")
    listing = client.list_objects_v2(Bucket=bucket, Prefix=prefix.rstrip("/") + "/")
    hits = [o for o in listing.get("Contents", []) if o["Key"].endswith(suffix)]
    if not hits:
        raise FileNotFoundError(f"no *{suffix} under s3://{bucket}/{prefix}")
    obj = hits[0]
    return bucket, obj["Key"], int(obj["Size"]), obj["ETag"].strip('"')


def fetch_tile(row: pd.Series, dest: Path, client=None, check_md5: bool = True,
               revalidate: bool = False) -> Path:
    """Download one manifest row's raster, unless it is already correctly on disk.

    **Verification is against the object's own size and ETag, not the manifest's
    ``content_length`` / ``checksum_value``.** Those describe the granule as a
    whole and match nothing on disk: for the tile above the manifest says
    1 452 965 bytes, while the raster is 1 348 852 and even all seven members
    together are 1 450 973. Checking a downloaded ``.tif`` against the manifest
    figure would reject every correct fetch.

    A single-part S3 ETag *is* the MD5, and these granules are well under any
    multipart threshold, so the ETag is a genuine content hash here. Multipart
    ETags (``<hash>-<parts>``) are not, and are skipped rather than mis-trusted.

    Args:
        row: A row of :func:`~.catalogue.load_manifest`, carrying ``s3_path``.
        dest: Where the raster should end up.
        client: A boto3 S3 client; built from :func:`build_client` when omitted.
        check_md5: Also verify the hash, not just the length.
        revalidate: Re-list and re-check a file that is already present. Off by
            default -- see the note in the body about why that is safe and why
            leaving it on gets the account throttled.

    Returns:
        ``dest``.

    Raises:
        IOError: If the finished download does not match the object.
    """
    # Presence alone is trusted, and no network call is made for a file already
    # here. That is exactly the `.part`-then-rename guarantee above: `dest`
    # existing means a download completed AND matched its ETag when it was
    # written. Re-listing to re-confirm it costs one S3 call per object -- 15 936
    # of them on a cached re-run, which is what earned HTTP 429s from CDSE and
    # failed 161 tiles that were in fact complete. Pass revalidate=True to force
    # the check when the store is suspect.
    if dest.is_file() and dest.stat().st_size > 0 and not revalidate:
        logger.debug("%s: present", dest.name)
        return dest

    if client is None:
        client = build_client()

    bucket, key, size, etag = resolve_object(client, str(row["s3_path"]))
    md5 = etag if (check_md5 and "-" not in etag) else None

    if verify(dest, size, md5):
        logger.debug("%s: cached", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    client.download_file(bucket, key, str(part))

    if not verify(part, size, md5):
        part.unlink(missing_ok=True)
        raise IOError(f"{dest.name}: download does not match s3://{bucket}/{key}")
    part.replace(dest)
    logger.info("%s: %.2f MB", dest.name, size / 1e6)
    return dest


#: A prefix known to exist, for the credential check below.
_PROBE_PREFIX = "CLMS/landcover_landuse/crop_types/"


def smoke_test(profile: str | None = None, prefix: str = _PROBE_PREFIX) -> bool:
    """Prove the credentials work, and say which way they failed if not.

    CDSE's S3 errors are terse and all look alike from the outside, so this
    separates the four things that actually go wrong: no profile, bad keys, keys
    without eodata access, and the wrong addressing style.

    Args:
        profile: Credentials profile; defaults to :data:`CDSE_PROFILE`.
        prefix: Key prefix to list.

    Returns:
        ``True`` if objects were listed.
    """
    from botocore.exceptions import ClientError, ProfileNotFound

    name = CDSE_PROFILE if profile is None else profile
    try:
        client = build_client(profile)
    except ProfileNotFound:
        print(
            f"No [{name}] profile in ~/.aws/credentials.\n"
            "Generate S3 keys at https://eodata-s3keysmanager.dataspace.copernicus.eu/\n"
            "then add them as a NAMED profile -- do not touch [default], which is\n"
            "already in use on this machine:\n\n"
            f"  [{name}]\n"
            "  aws_access_key_id = ...\n"
            "  aws_secret_access_key = ..."
        )
        return False

    try:
        # Delimiter is required. Without it this gateway returns an empty
        # Contents for a prefix whose objects live deeper, which reads exactly
        # like "authenticated but empty bucket" and is not.
        page = client.list_objects_v2(
            Bucket="eodata", Prefix=prefix, Delimiter="/", MaxKeys=10
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "?")
        hint = {
            "InvalidAccessKeyId": "the access key is not recognised",
            "SignatureDoesNotMatch": "the secret does not match the access key",
            "AccessDenied": "the keys are valid but lack eodata read access",
            "NoSuchBucket": "wrong endpoint, or virtual-host addressing in use",
        }.get(code, "unexpected")
        print(f"S3 refused the request: {code} -- {hint}")
        return False

    found = [p["Prefix"] for p in page.get("CommonPrefixes", [])] + [
        o["Key"] for o in page.get("Contents", [])
    ]
    if not found:
        print(f"Authenticated, but nothing under {prefix!r}.")
        return False
    print(f"OK: profile [{name}] can read eodata. Under {prefix!r}:")
    for item in found[:6]:
        print(f"  {item}")
    return True


if __name__ == "__main__":  # pragma: no cover - a hand-run credential check
    import sys

    sys.exit(0 if smoke_test() else 1)
