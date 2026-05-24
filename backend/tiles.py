from rio_tiler.io import COGReader
from fastapi.responses import Response
from pathlib import Path
import numpy as np

LAYER_PATHS = {
    "tci_2019": "outputs/tci/tci_2019.tif",
    "tci_2020": "outputs/tci/tci_2020.tif",
    "tci_2021": "outputs/tci/tci_2021.tif",
    "tci_2022": "outputs/tci/tci_2022.tif",
    "tci_2023": "outputs/tci/tci_2023.tif",
    "tci_2024": "outputs/tci/tci_2024.tif",
    "tci_2025": "outputs/tci/tci_2025.tif",
    "tci_2026": "outputs/tci/tci_2026.tif",
    "ndvi_2019": "outputs/indices/ndvi_2019.tif",
    "ndvi_2020": "outputs/indices/ndvi_2020.tif",
    "ndvi_2021": "outputs/indices/ndvi_2021.tif",
    "ndvi_2022": "outputs/indices/ndvi_2022.tif",
    "ndvi_2023": "outputs/indices/ndvi_2023.tif",
    "ndvi_2024": "outputs/indices/ndvi_2024.tif",
    "ndvi_2025": "outputs/indices/ndvi_2025.tif",
    "ndvi_2026": "outputs/indices/ndvi_2026.tif",
    "ndwi_2019": "outputs/indices/ndwi_2019.tif",
    "ndwi_2020": "outputs/indices/ndwi_2020.tif",
    "ndwi_2021": "outputs/indices/ndwi_2021.tif",
    "ndwi_2022": "outputs/indices/ndwi_2022.tif",
    "ndwi_2023": "outputs/indices/ndwi_2023.tif",
    "ndwi_2024": "outputs/indices/ndwi_2024.tif",
    "ndwi_2025": "outputs/indices/ndwi_2025.tif",
    "ndwi_2026": "outputs/indices/ndwi_2026.tif",
    "ndbi_2019": "outputs/indices/ndbi_2019.tif",
    "ndbi_2020": "outputs/indices/ndbi_2020.tif",
    "ndbi_2021": "outputs/indices/ndbi_2021.tif",
    "ndbi_2022": "outputs/indices/ndbi_2022.tif",
    "ndbi_2023": "outputs/indices/ndbi_2023.tif",
    "ndbi_2024": "outputs/indices/ndbi_2024.tif",
    "ndbi_2025": "outputs/indices/ndbi_2025.tif",
    "ndbi_2026": "outputs/indices/ndbi_2026.tif",
    "tci": "outputs/tci/tci_2023.tif",
    "ndvi": "outputs/indices/ndvi_2023.tif",
    "ndwi": "outputs/indices/ndwi_2023.tif",
    "ndbi": "outputs/indices/ndbi_2023.tif",
}

def get_tile(layer: str, z: int, x: int, y: int) -> Response:
    path = LAYER_PATHS.get(layer)
    if not path or not Path(path).exists():
        return Response(status_code=404)
    try:
        with COGReader(path) as cog:
            img = cog.tile(x, y, z, tilesize=256)
            if layer.startswith("tci"):
                # RGB — render as PNG directly
                content = img.render(img_format="PNG")
            else:
                # Single band index — apply colormap
                arr = img.data[0]
                # normalize -1 to 1 → 0 to 255
                normalized = np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)
                from PIL import Image
                import io
                colored = Image.fromarray(normalized, mode='L').convert('RGBA')
                buf = io.BytesIO()
                colored.save(buf, format='PNG')
                content = buf.getvalue()
        return Response(content=content, media_type="image/png")
    except Exception as e:
        return Response(status_code=404)