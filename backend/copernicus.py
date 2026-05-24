"""Copernicus Data Space OAuth2, STAC search, and download helpers."""

from __future__ import annotations

import logging
import os
import zipfile
from email.message import Message
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
DEFAULT_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
REQUEST_TIMEOUT_SECONDS = 60.0
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
BYTES_PER_MIB = 1024 * 1024

# Correct bboxes verified against live STAC API
TILE_BBOXES: dict[str, list[float]] = {
    "T43PGQ": [76.84, 12.56, 77.86, 13.56],
}


class CopernicusAuthError(Exception):
    """Raised when Copernicus OAuth2 authentication fails."""


class DownloadError(Exception):
    """Raised when a Copernicus scene download or extraction fails."""


def get_copernicus_token(config: dict[str, Any]) -> str:
    """Request a Copernicus Data Space access token.

    Args:
        config: Application configuration containing
            ``config["copernicus"]["token_url"]``.

    Returns:
        OAuth2 access token string.

    Raises:
        CopernicusAuthError: If credentials are missing, the token endpoint
            rejects the request, or the response does not contain an
            ``access_token``.
    """
    copernicus_config = config.get("copernicus", {})
    token_url = copernicus_config.get("token_url")
    client_id = os.getenv("COPERNICUS_CLIENT_ID")
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET")

    if not token_url:
        raise CopernicusAuthError("Missing config['copernicus']['token_url'].")
    if not client_id or not client_secret:
        raise CopernicusAuthError(
            "Missing COPERNICUS_CLIENT_ID or COPERNICUS_CLIENT_SECRET environment variable."
        )

    use_password = os.getenv("COPERNICUS_USERNAME") and os.getenv("COPERNICUS_PASSWORD")
    if use_password:
        payload = {
            "grant_type": "password",
            "client_id": "cdse-public",
            "username": os.getenv("COPERNICUS_USERNAME"),
            "password": os.getenv("COPERNICUS_PASSWORD"),
        }
    else:
        payload = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(token_url, data=payload)
            response.raise_for_status()
            token_payload = response.json()
    except httpx.HTTPStatusError as exc:
        message = _response_error_message(exc.response)
        raise CopernicusAuthError(f"Copernicus token request failed: {message}") from exc
    except (httpx.RequestError, ValueError) as exc:
        raise CopernicusAuthError(f"Copernicus token request failed: {exc}") from exc

    access_token = token_payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise CopernicusAuthError("Copernicus token response did not include access_token.")

    return access_token


def search_scenes(
    tile_id: str,
    date_from: str,
    date_to: str,
    max_cloud_cover: float,
    config: dict[str, Any],
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Search Sentinel-2 scenes in the Copernicus Data Space STAC API.

    Args:
        tile_id: Sentinel-2 MGRS tile ID, for example ``"T43PGQ"``.
        date_from: Start date in ``YYYY-MM-DD`` format.
        date_to: End date in ``YYYY-MM-DD`` format.
        max_cloud_cover: Maximum accepted cloud cover percentage, from 0 to 100.
        config: Application configuration passed to ``get_copernicus_token``.
        max_results: Maximum number of STAC items to request.

    Returns:
        A list of normalized scene dictionaries. Returns an empty list if the
        STAC API request fails or the response cannot be parsed.
    """
    try:
        token = get_copernicus_token(config)
        normalized_tile_id = tile_id.upper()
        mgrs_code = f"MGRS-{normalized_tile_id.lstrip('T')}"
        bbox = TILE_BBOXES.get(normalized_tile_id)

        body: dict[str, Any] = {
            "collections": ["sentinel-2-l2a"],
            "datetime": f"{date_from}T00:00:00Z/{date_to}T23:59:59Z",
            "limit": max_results,
            "filter": {
                "op": "and",
                "args": [
                    {"op": "=", "args": [{"property": "grid:code"}, mgrs_code]},
                    {"op": "<=", "args": [{"property": "eo:cloud_cover"}, max_cloud_cover]},
                ],
            },
            "filter-lang": "cql2-json",
        }
        if bbox:
            body["bbox"] = bbox

        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(
                STAC_SEARCH_URL,
                json=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            payload = response.json()

        features = payload.get("features", [])
        if not isinstance(features, list):
            logger.error("Copernicus STAC response had invalid features payload.")
            return []

        return [_scene_from_feature(feature, normalized_tile_id) for feature in features]

    except Exception as exc:
        logger.error("Copernicus STAC search failed: %s", exc)
        return []


def download_scene(
    download_id: str,
    year: str,
    config: dict[str, Any],
    progress_callback: Callable[[int], None] | None = None,
) -> str:
    """Download and extract a Copernicus scene archive.

    Args:
        download_id: Copernicus Data Space download identifier.
        year: Destination year folder below ``data/ISRO``.
        config: Application configuration passed to ``get_copernicus_token``.
        progress_callback: Optional callback receiving integer percentages.

    Returns:
        Path to the extracted ``.SAFE`` folder.

    Raises:
        DownloadError: If authentication, download, zip extraction, or
            ``.SAFE`` folder detection fails.
    """
    target_dir = Path("data") / "ISRO" / str(year)
    zip_path: Path | None = None

    try:
        token = get_copernicus_token(config)
        download_url = _download_url(config, download_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        with httpx.Client(timeout=None, follow_redirects=True) as client:
            with client.stream(
                "GET",
                download_url,
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                response.raise_for_status()
                filename = _filename_from_response(response, download_id)
                zip_path = target_dir / filename
                total_bytes = int(response.headers.get("content-length", "0") or 0)
                downloaded_bytes = 0
                next_progress = 0

                if progress_callback:
                    next_progress = _emit_progress(progress_callback, 0, next_progress)

                with zip_path.open("wb") as zip_file:
                    for chunk in response.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if not chunk:
                            continue
                        zip_file.write(chunk)
                        downloaded_bytes += len(chunk)
                        if progress_callback and total_bytes > 0:
                            percent = int(downloaded_bytes * 100 / total_bytes)
                            while percent >= next_progress and next_progress <= 100:
                                next_progress = _emit_progress(
                                    progress_callback, next_progress, next_progress
                                )

        safe_path = _extract_safe_folder(zip_path, target_dir)
        zip_path.unlink(missing_ok=True)

        if progress_callback and next_progress <= 100:
            _emit_progress(progress_callback, 100, next_progress)

        return str(safe_path)

    except Exception as exc:
        if zip_path:
            zip_path.unlink(missing_ok=True)
        logger.error("Copernicus scene download failed for %s: %s", download_id, exc)
        if isinstance(exc, DownloadError):
            raise
        raise DownloadError(f"Failed to download Copernicus scene {download_id}: {exc}") from exc


def _scene_from_feature(feature: Any, fallback_tile_id: str) -> dict[str, Any]:
    """Normalize one STAC feature into the application's scene shape."""
    if not isinstance(feature, dict):
        feature = {}

    properties = feature.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}

    # Use scene id as download_id (Copernicus zipper accepts scene id)
    scene_id = str(feature.get("id") or properties.get("id") or "")
    # in _scene_from_feature, replace download_id logic:
    assets = feature.get("assets", {})
    product_href = assets.get("Product", {}).get("href", "")
    download_id = product_href if product_href else str(feature.get("id", ""))
    title = _scene_title(properties, scene_id)
    acquisition_datetime = (
        properties.get("datetime")
        or properties.get("start_datetime")
        or properties.get("created")
        or ""
    )
    date = str(acquisition_datetime)[:10] if acquisition_datetime else ""
    cloud_cover = _as_float(properties.get("eo:cloud_cover") or properties.get("cloudCover"))

    # grid:code is "MGRS-43PGQ" → strip prefix for display
    raw_grid = properties.get("grid:code") or ""
    tile = raw_grid.replace("MGRS-", "T") if raw_grid else fallback_tile_id

    return {
        "scene_id": scene_id,
        "title": title,
        "date": date,
        "cloud_cover": cloud_cover,
        "tile": tile,
        "download_id": download_id,
        "bbox": _feature_bbox(feature),
        "size_mb": _assets_size_mb(feature.get("assets", {})),
    }


def _scene_title(properties: dict[str, Any], scene_id: str) -> str:
    """Return a compact scene title from STAC properties."""
    title = (
        properties.get("title")
        or properties.get("productIdentifier")
        or properties.get("product_id")
        or scene_id
    )
    return str(title).rstrip("/").replace("\\", "/").split("/")[-1]


def _assets_size_mb(assets: Any) -> float:
    """Calculate an approximate asset size in MiB from STAC asset metadata."""
    if not isinstance(assets, dict):
        return 0.0

    size_bytes = 0.0
    for asset in assets.values():
        if not isinstance(asset, dict):
            continue
        for key in ("file:size", "size", "content_length", "contentLength"):
            value = asset.get(key)
            if isinstance(value, (int, float)):
                size_bytes += float(value)
                break

    if size_bytes <= 0:
        return 0.0
    return round(size_bytes / BYTES_PER_MIB, 2)


def _feature_bbox(feature: dict[str, Any]) -> list[float]:
    """Return a STAC feature bbox, deriving it from geometry when needed."""
    bbox = feature.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return [float(v) for v in bbox[:4]]

    coordinates = feature.get("geometry", {}).get("coordinates")
    points = list(_iter_coordinate_pairs(coordinates))
    if not points:
        return []

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _iter_coordinate_pairs(value: Any) -> Any:
    """Yield coordinate pairs from nested GeoJSON coordinate arrays."""
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        yield float(value[0]), float(value[1])
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_coordinate_pairs(item)


def _as_float(value: Any) -> float:
    """Convert numeric API values to float, returning 0.0 when absent."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _download_url(config, download_id):
    if download_id.startswith("https://"):
        if download_id.endswith("/$value"):
            return download_id
        return download_id + "/$value"
    base_url = config.get("copernicus", {}).get("download_url") or DEFAULT_DOWNLOAD_URL
    return f"{str(base_url).rstrip('/')}/{quote(download_id, safe='')}"


def _filename_from_response(response: httpx.Response, download_id: str) -> str:
    """Derive a safe zip filename from response headers or the download ID."""
    content_disposition = response.headers.get("content-disposition")
    filename = None
    if content_disposition:
        message = Message()
        message["content-disposition"] = content_disposition
        filename = message.get_filename()

    if not filename:
        filename = download_id

    filename = _safe_filename(filename)
    if not filename.lower().endswith(".zip"):
        filename = f"{Path(filename).stem}.zip"
    return filename


def _safe_filename(filename: str) -> str:
    """Return a filesystem-safe filename without directory components."""
    cleaned = Path(str(filename).replace("\\", "/")).name.strip()
    allowed = []
    for character in cleaned:
        if character.isalnum() or character in ("-", "_", "."):
            allowed.append(character)
        else:
            allowed.append("_")
    return "".join(allowed).strip("._") or "copernicus_scene"


def _extract_safe_folder(zip_path: Path, target_dir: Path) -> Path:
    """Safely extract a zip archive and return the extracted .SAFE folder path."""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            safe_members = [
                member.split("/", 1)[0]
                for member in archive.namelist()
                if member and member.split("/", 1)[0].endswith(".SAFE")
            ]
            _safe_extract(archive, target_dir)
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"Downloaded file is not a valid zip archive: {zip_path}") from exc

    for safe_member in safe_members:
        safe_path = target_dir / safe_member
        if safe_path.is_dir():
            return safe_path

    safe_folders = sorted(target_dir.glob("*.SAFE"), key=lambda p: p.stat().st_mtime)
    if safe_folders:
        return safe_folders[-1]

    raise DownloadError(f"No .SAFE folder found after extracting {zip_path.name}.")


def _safe_extract(archive: zipfile.ZipFile, target_dir: Path) -> None:
    """Extract a zip archive while preventing path traversal."""
    target_root = target_dir.resolve()
    for member in archive.infolist():
        destination = (target_dir / member.filename).resolve()
        if target_root != destination and target_root not in destination.parents:
            raise DownloadError(f"Unsafe archive member path: {member.filename}")
    archive.extractall(target_dir)


def _emit_progress(
    progress_callback: Callable[[int], None],
    percent: int,
    next_progress: int,
) -> int:
    """Invoke a progress callback and return the next 5% threshold."""
    try:
        progress_callback(min(max(percent, 0), 100))
    except Exception as exc:
        logger.warning("Copernicus download progress callback failed: %s", exc)
    return next_progress + 5


def _response_error_message(response: httpx.Response) -> str:
    """Return a compact, non-secret HTTP error description."""
    body = response.text[:500].replace("\n", " ").strip()
    return f"HTTP {response.status_code} {response.reason_phrase}: {body}"