"""
GitHub-backed persistence for history, phenology and model artifacts.

Large but regenerable state — history.csv, phenology.csv and models/*.joblib —
is stored as compressed assets on a single GitHub release (tag: ``data``).
Releases are free, allow 2 GB per file and, unlike committing the files, do not
grow the git history on every run.

The generated forecast.json is deliberately *not* stored here: it is written to
``data/forecast.json`` and published to GitHub Pages by the workflow.

Reads are unauthenticated (the repo is public), so local development picks up
production history with no setup at all. Writes need ``GITHUB_TOKEN``, which
Actions provides automatically.
"""

import gzip
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import httpx
import pandas as pd


API_URL = "https://api.github.com"
UPLOAD_URL = "https://uploads.github.com"

DEFAULT_REPO = "renevinaya/muc-pollen-forecast"
RELEASE_TAG = os.environ.get("DATA_RELEASE_TAG", "data")

HISTORY_ASSET = "history.csv.gz"
PHENOLOGY_ASSET = "phenology.csv.gz"
MODELS_ASSET = "models.tar.gz"

_TIMEOUT = httpx.Timeout(30.0, read=600.0, write=600.0)


def _repo() -> str:
    """owner/repo holding the data release."""
    return os.environ.get("DATA_REPO") or os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO


def _token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def can_upload() -> bool:
    """True when a token is available, i.e. we may write back to the release."""
    return _token() is not None


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request(method: str, url: str, **kwargs) -> httpx.Response:
    """Issue a request, retrying transient failures with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            resp = httpx.request(method, url, timeout=_TIMEOUT, **kwargs)
            if resp.status_code < 500:
                return resp
            last_exc = httpx.HTTPStatusError(
                f"{resp.status_code} from {url}", request=resp.request, response=resp
            )
        except httpx.HTTPError as exc:
            last_exc = exc
        if attempt < 3:
            time.sleep(2**attempt)
    raise last_exc  # type: ignore[misc]


def _get_release() -> dict | None:
    """Fetch the data release, or None if it does not exist yet."""
    resp = _request(
        "GET", f"{API_URL}/repos/{_repo()}/releases/tags/{RELEASE_TAG}", headers=_headers()
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _ensure_release() -> dict:
    """Fetch the data release, creating it on first use."""
    release = _get_release()
    if release is not None:
        return release
    resp = _request(
        "POST",
        f"{API_URL}/repos/{_repo()}/releases",
        headers=_headers(),
        json={
            "tag_name": RELEASE_TAG,
            "name": "Pipeline data",
            "body": (
                "Accumulated history, phenology and trained models for the forecast "
                "pipeline. Updated automatically — not a software release."
            ),
            "make_latest": "false",
        },
    )
    resp.raise_for_status()
    print(f"Created release '{RELEASE_TAG}' in {_repo()}")
    return resp.json()


def _download_asset(name: str, dest: Path) -> bool:
    """Download a release asset verbatim to dest. Returns False if absent."""
    release = _get_release()
    if release is None:
        print(f"No release '{RELEASE_TAG}' in {_repo()} yet")
        return False
    asset = next((a for a in release.get("assets", []) if a["name"] == name), None)
    if asset is None:
        print(f"No asset '{name}' in release '{RELEASE_TAG}'")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    # browser_download_url is public and redirects to a signed CDN URL; sending
    # our token along would be rejected there, so fetch it without auth.
    with httpx.stream(
        "GET", asset["browser_download_url"], follow_redirects=True, timeout=_TIMEOUT
    ) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
    size_mb = dest.stat().st_size / 1e6
    print(f"Downloaded {name} ({size_mb:.1f} MB)")
    return True


def _delete_asset(release: dict, name: str) -> None:
    """Remove an existing asset by name so it can be replaced."""
    for asset in release.get("assets", []):
        if asset["name"] == name:
            _request(
                "DELETE",
                f"{API_URL}/repos/{_repo()}/releases/assets/{asset['id']}",
                headers=_headers(),
            )


def _upload_asset(source: Path, name: str) -> None:
    """Upload source as a release asset called name, replacing any existing one."""
    if not can_upload():
        print(f"No GITHUB_TOKEN — skipping upload of {name}")
        return

    release = _ensure_release()
    payload = source.read_bytes()

    for attempt in range(2):
        _delete_asset(release, name)
        resp = _request(
            "POST",
            f"{UPLOAD_URL}/repos/{_repo()}/releases/{release['id']}/assets",
            headers={**_headers(), "Content-Type": "application/octet-stream"},
            params={"name": name},
            content=payload,
        )
        # 422 means the asset we just deleted is still attached; re-read and retry.
        if resp.status_code == 422 and attempt == 0:
            time.sleep(2)
            release = _ensure_release()
            continue
        resp.raise_for_status()
        break

    print(f"Uploaded {name} ({len(payload) / 1e6:.1f} MB) to release '{RELEASE_TAG}'")


def upload_csv(local_path: Path, name: str) -> None:
    """Gzip a CSV and store it as a release asset."""
    with tempfile.TemporaryDirectory() as tmp:
        packed = Path(tmp) / name
        with local_path.open("rb") as src, gzip.open(packed, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        _upload_asset(packed, name)


def download_csv(name: str, local_path: Path) -> bool:
    """Fetch a gzipped CSV asset and decompress it to local_path."""
    with tempfile.TemporaryDirectory() as tmp:
        packed = Path(tmp) / name
        if not _download_asset(name, packed):
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(packed, "rb") as src, local_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    print(f"Wrote {local_path}")
    return True


def upload_models(models_dir: Path) -> None:
    """Bundle all .joblib models into a single tarball and store it."""
    models = sorted(models_dir.glob("*.joblib"))
    if not models:
        print("No models to upload")
        return
    with tempfile.TemporaryDirectory() as tmp:
        packed = Path(tmp) / MODELS_ASSET
        with tarfile.open(packed, "w:gz") as tar:
            for model in models:
                tar.add(model, arcname=model.name)
        print(f"Bundled {len(models)} models")
        _upload_asset(packed, MODELS_ASSET)


def download_models(models_dir: Path) -> int:
    """Fetch and unpack the model tarball. Returns the number of models restored."""
    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        packed = Path(tmp) / MODELS_ASSET
        if not _download_asset(MODELS_ASSET, packed):
            return 0
        with tarfile.open(packed, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".joblib")]
            for member in members:
                # Copy out by basename: an archived path can never escape models_dir.
                src = tar.extractfile(member)
                if src is None:
                    continue
                with (models_dir / Path(member.name).name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    print(f"Restored {len(members)} models to {models_dir}")
    return len(members)


def sync_historical_data(local_path: Path) -> pd.DataFrame:
    """
    Load historical data from the local CSV, falling back to the data release.
    Returns an empty DataFrame if no history exists anywhere yet.
    """
    if local_path.exists():
        df = pd.read_csv(local_path, parse_dates=["date"])
        print(f"Loaded {len(df)} rows from {local_path}")
        return df

    if download_csv(HISTORY_ASSET, local_path):
        return pd.read_csv(local_path, parse_dates=["date"])

    return pd.DataFrame()
