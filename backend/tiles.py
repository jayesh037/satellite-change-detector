import logging
from rio_tiler.io import COGReader
from fastapi.responses import Response
from pathlib import Path
import numpy as np

from backend.storage import get_signed_url, file_exists

logger = logging.getLogger(__name__)

LAYER_PATHS = {
    "tci_2019": "tiles/tci/tci_2019.tif",
    "tci_2020": "tiles/tci/tci_2020.tif",
    "tci_2021": "tiles/tci/tci_2021.tif",
    "tci_2022": "tiles/tci/tci_2022.tif",
    "tci_2023": "tiles/tci/tci_2023.tif",
    "tci_2024": "tiles/tci/tci_2024.tif",
    "tci_2025": "tiles/tci/tci_2025.tif",
    "tci_2026": "tiles/tci/tci_2026.tif",
    "ndvi_2019": "tiles/indices/ndvi_2019.tif",
    "ndvi_2020": "tiles/indices/ndvi_2020.tif",
    "ndvi_2021": "tiles/indices/ndvi_2021.tif",
    "ndvi_2022": "tiles/indices/ndvi_2022.tif",
    "ndvi_2023": "tiles/indices/ndvi_2023.tif",
    "ndvi_2024": "tiles/indices/ndvi_2024.tif",
    "ndvi_2025": "tiles/indices/ndvi_2025.tif",
    "ndvi_2026": "tiles/indices/ndvi_2026.tif",
    "ndwi_2019": "tiles/indices/ndwi_2019.tif",
    "ndwi_2020": "tiles/indices/ndwi_2020.tif",
    "ndwi_2021": "tiles/indices/ndwi_2021.tif",
    "ndwi_2022": "tiles/indices/ndwi_2022.tif",
    "ndwi_2023": "tiles/indices/ndwi_2023.tif",
    "ndwi_2024": "tiles/indices/ndwi_2024.tif",
    "ndwi_2025": "tiles/indices/ndwi_2025.tif",
    "ndwi_2026": "tiles/indices/ndwi_2026.tif",
    "ndbi_2019": "tiles/indices/ndbi_2019.tif",
    "ndbi_2020": "tiles/indices/ndbi_2020.tif",
    "ndbi_2021": "tiles/indices/ndbi_2021.tif",
    "ndbi_2022": "tiles/indices/ndbi_2022.tif",
    "ndbi_2023": "tiles/indices/ndbi_2023.tif",
    "ndbi_2024": "tiles/indices/ndbi_2024.tif",
    "ndbi_2025": "tiles/indices/ndbi_2025.tif",
    "ndbi_2026": "tiles/indices/ndbi_2026.tif",
    "tci":  "tiles/tci/tci_2023.tif",
    "ndvi": "tiles/indices/ndvi_2023.tif",
    "ndwi": "tiles/indices/ndwi_2023.tif",
    "ndbi": "tiles/indices/ndbi_2023.tif",
}

def get_tile(layer: str, z: int, x: int, y: int) -> Response:
    b2_key = LAYER_PATHS.get(layer)
    if not b2_key:
        return Response(status_code=404)
    
    # Generate pre-signed URL for B2 file
    # rio-tiler can read remote files via /vsicurl/
    if not file_exists(b2_key):
        return Response(status_code=404)
    
    signed_url = get_signed_url(b2_key, expires_in=3600)
    
    try:
        with COGReader(signed_url) as cog:
            img = cog.tile(x, y, z, tilesize=256)
            if layer.startswith("tci"):
                content = img.render(img_format="PNG")
            else:
                arr = img.data[0]
                normalized = np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)
                from PIL import Image
                import io
                im = Image.fromarray(normalized, mode='L').convert('RGBA')
                buf = io.BytesIO()
                im.save(buf, format='PNG')
                content = buf.getvalue()
        return Response(content=content, media_type="image/png")
    except Exception as e:
        logger.error(f"Tile error {layer}/{z}/{x}/{y}: {e}")
        return Response(status_code=404)