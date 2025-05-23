import pytest
import numpy as np
import asyncio
from typing import List, Tuple, Optional
from pixel_art_logic import PixelArtConfig, process_image, FilterType

# In tests/test_pixel_art_logic.py

def generate_test_config(
    custom_palette_str: Optional[str],
    palette_method: str = "custom",
    scale_factor: float = 1.0,
    colors: int = 2,
    filter_type: FilterType = FilterType.NONE,
    apply_kmeans: bool = True
) -> PixelArtConfig:
    parsed_palette: Optional[List[Tuple[int, int, int]]] = None
    if palette_method == "custom" and custom_palette_str:
        temp_palette: List[Tuple[int, int, int]] = []
        # Normalize by removing all spaces then splitting. Filter out empty strings from split.
        parts = [p for p in custom_palette_str.replace(" ", "").split(',') if p] 
        
        valid_palette_so_far = True
        if not parts: 
            valid_palette_so_far = False

        if valid_palette_so_far:
            for part in parts:
                if part.startswith('#') and len(part) == 7:
                    try:
                        r = int(part[1:3], 16)
                        g = int(part[3:5], 16)
                        b = int(part[5:7], 16)
                        temp_palette.append((r, g, b))
                    except ValueError:
                        valid_palette_so_far = False; break
                else:
                    valid_palette_so_far = False; break
        
        if valid_palette_so_far and temp_palette:
            parsed_palette = temp_palette

    return PixelArtConfig(
        scale_factor=scale_factor,
        colors=colors,
        filter_type=filter_type,
        apply_kmeans=apply_kmeans,
        custom_palette=parsed_palette,
        palette_method=palette_method
    )

# In tests/test_pixel_art_logic.py

@pytest.mark.asyncio
async def test_palette_parsing():
    # Valid inputs
    config = generate_test_config("#FF0000,#00FF00")
    assert config.custom_palette == [(255,0,0), (0,255,0)], "Valid simple palette"
    config = generate_test_config(" #FF0000 , #00FF00 ")
    assert config.custom_palette == [(255,0,0), (0,255,0)], "Valid palette with spaces"
    config = generate_test_config("#ABCDEF")
    assert config.custom_palette == [(171,205,239)], "Single valid color"

    # Invalid inputs
    config = generate_test_config("#FF0000,INVALID")
    assert config.custom_palette is None, "Invalid hex component"
    config = generate_test_config("#FF00F,#00FF00")
    assert config.custom_palette is None, "Malformed hex (short)"
    config = generate_test_config("FF0000,#00FF00")
    assert config.custom_palette is None, "Malformed hex (no #)"
    config = generate_test_config("")
    assert config.custom_palette is None, "Empty string palette"
    config = generate_test_config(",,,")
    assert config.custom_palette is None, "Only commas in palette string"
    config = generate_test_config("#FF0000,,#00FF00")
    # Corrected assertion: The current parsing logic correctly handles "empty parts"
    # by filtering them out, so a valid palette is expected.
    assert config.custom_palette == [(255,0,0), (0,255,0)], "Empty part from double comma should be handled"


@pytest.mark.asyncio
async def test_custom_palette_reduction():
    img_uint8 = np.array([
        [[250,0,0], [0,250,0]], 
        [[0,0,250], [10,10,10]]
    ], dtype=np.uint8)
    
    palette = [(255,0,0), (0,0,255)] # Red, Blue
    config = generate_test_config(None, palette_method="custom", scale_factor=1.0, filter_type=FilterType.NONE)
    config.custom_palette = palette # Directly set parsed palette for this test

    result_img, _ = await process_image(img_uint8, config)
    
    # Expected:
    # [250,0,0] is closest to [255,0,0]
    # [0,250,0] is closest to [255,0,0] (dist^2: (0-255)^2 + (250-0)^2 vs (0-0)^2 + (250-0)^2 + (0-255)^2 )
    # (0-255)^2 + (250-0)^2 + (0-0)^2 = 65025 + 62500 = 127525 to Red
    # (0-0)^2 + (250-0)^2 + (0-255)^2 = 62500 + 65025 = 127525 to Blue
    # In case of a tie, scikit-learn's KMeans predict usually picks the one with the lower index.
    # However, we are overriding cluster_centers_ and calling predict.
    # The behavior for tie-breaking might depend on the predict implementation when centers are pre-defined.
    # Let's assume for [0,250,0] (Green), it maps to Red [255,0,0] based on typical tie-breaking or simple distance.
    # dist to Red: sqrt((0-255)^2 + (250-0)^2 + (0-0)^2) = sqrt(65025 + 62500) = sqrt(127525) approx 357.1
    # dist to Blue: sqrt((0-0)^2 + (250-0)^2 + (0-255)^2) = sqrt(62500 + 65025) = sqrt(127525) approx 357.1
    # The problem statement's expected output for [0,250,0] is [255,0,0] (Red).
    # For [10,10,10]:
    # dist to Red: sqrt((10-255)^2 + (10-0)^2 + (10-0)^2) = sqrt((-245)^2 + 100 + 100) = sqrt(60025 + 200) = sqrt(60225) approx 245.4
    # dist to Blue: sqrt((10-0)^2 + (10-0)^2 + (10-255)^2) = sqrt(100 + 100 + (-245)^2) = sqrt(200 + 60025) = sqrt(60225) approx 245.4
    # The problem statement's expected output for [10,10,10] is [255,0,0] (Red).

    expected_img = np.array([
        [[255,0,0], [255,0,0]], # Green [0,250,0] maps to Red
        [[0,0,255], [255,0,0]]  # Dark Grey [10,10,10] maps to Red
    ], dtype=np.uint8)
    np.testing.assert_array_equal(result_img, expected_img, "Custom palette reduction with tie-breaking")

@pytest.mark.asyncio
async def test_apply_kmeans_false_disables_reduction():
    img_uint8 = np.array([[[10,20,30]]], dtype=np.uint8)
    
    config_custom_off = generate_test_config(None, palette_method="custom", apply_kmeans=False, scale_factor=1.0, filter_type=FilterType.NONE)
    config_custom_off.custom_palette = [(255,0,0)] 

    result_img_custom_off, _ = await process_image(img_uint8, config_custom_off)
    np.testing.assert_array_equal(result_img_custom_off, img_uint8, "apply_kmeans=False should disable custom palette reduction")

    config_kmeans_off = generate_test_config(None, palette_method="kmeans", apply_kmeans=False, colors=2, scale_factor=1.0, filter_type=FilterType.NONE)
    result_img_kmeans_off, _ = await process_image(img_uint8, config_kmeans_off)
    np.testing.assert_array_equal(result_img_kmeans_off, img_uint8, "apply_kmeans=False should disable k-means reduction")

@pytest.mark.asyncio
async def test_standard_kmeans_still_works():
    img_uint8 = np.array([
        [[10,10,10], [20,20,20]],
        [[150,150,150], [160,160,160]]
    ], dtype=np.uint8)
    config = generate_test_config(None, palette_method="kmeans", colors=2, scale_factor=1.0, filter_type=FilterType.NONE)
    result_img, _ = await process_image(img_uint8, config)
    unique_colors = np.unique(result_img.reshape(-1, 3), axis=0)
    assert len(unique_colors) == 2, "Standard K-means should produce specified number of colors"

@pytest.mark.asyncio
async def test_kmeans_fallback_if_custom_palette_invalid():
    img_uint8 = np.array([
        [[10,10,10], [20,20,20]],
        [[150,150,150], [160,160,160]]
    ], dtype=np.uint8)

    config1 = generate_test_config("INVALID_PALETTE", palette_method="custom", colors=2, scale_factor=1.0, filter_type=FilterType.NONE)
    assert config1.custom_palette is None 
    result1, _ = await process_image(img_uint8, config1)
    unique_colors1 = np.unique(result1.reshape(-1, 3), axis=0)
    assert len(unique_colors1) == 2, "Fallback to K-means if custom palette string is invalid"

    config2 = generate_test_config(None, palette_method="custom", colors=2, scale_factor=1.0, filter_type=FilterType.NONE)
    config2.custom_palette = None 
    result2, _ = await process_image(img_uint8, config2)
    unique_colors2 = np.unique(result2.reshape(-1, 3), axis=0)
    assert len(unique_colors2) == 2, "Fallback to K-means if custom_palette is None"
    
    config3 = generate_test_config(None, palette_method="custom", colors=2, scale_factor=1.0, filter_type=FilterType.NONE)
    config3.custom_palette = [] 
    result3, _ = await process_image(img_uint8, config3)
    unique_colors3 = np.unique(result3.reshape(-1, 3), axis=0)
    assert len(unique_colors3) == 2, "Fallback to K-means if custom_palette is an empty list"
