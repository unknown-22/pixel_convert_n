# Pixel Converter Next: Suggested Improvements and New Features

This document outlines potential improvements and new features for the Pixel Converter Next application. It includes currently planned items and additional suggestions for future development.

---
## Current Features

This project, Pixel Converter Next, allows users to convert images into pixel art through a Gradio-based web UI. Key existing features include:

- **Mosaic Processing**: Images are downscaled and then upscaled using nearest-neighbor interpolation to create a blocky, pixelated effect.
- **Color Reduction (k-means)**: The number of colors in the image is reduced by applying k-means clustering to the pixel colors. Users can specify the desired number of colors.
- **Gaussian Filter**: A Gaussian blur can be applied as a pre-processing step. Users can control the sigma value.
- **Erosion Filter**: An erosion filter can be applied as a pre-processing step. Users can control the kernel size.
- **Web UI (Gradio)**: Provides an interactive interface for uploading images and adjusting parameters.
- **Adjustable Parameters**: Users can control:
    - `scale_factor`: The degree of mosaic scaling.
    - `colors`: The number of colors for k-means reduction.
    - `filter_type`: Choice of 'None', 'Gaussian', or 'Erosion'.
    - `gaussian_sigma`: Strength of the Gaussian filter.
    - `erosion_size`: Strength of the erosion filter.
    - `apply_kmeans`: Toggle k-means color reduction on/off.

## Planned Features (from TODO)

The following features are already planned for future development, as listed in the project's README:

- **Outline Expansion**: Ability to add or enhance outlines around pixelated areas (options: none/weak/strong).
- **Saturation Adjustment**: Control over image saturation (options: none/weak/strong).
- **Contrast Adjustment**: Control over image contrast (options: none/weak/strong).
- **Dithering**: Implementation of dithering techniques to improve perceived color depth and reduce banding.

## Other Potential Improvements and New Features

Beyond the currently planned items, here are some additional ideas for enhancing Pixel Converter Next:

- **Advanced Scaling Algorithms**:
    - Explore and implement alternative image scaling algorithms for downscaling (e.g., bicubic, Lanczos variants if not already fully utilized) and upscaling (e.g., hqx, xBR) to offer different artistic styles for the pixel art.
- **Custom Color Palette**:
    - Allow users to upload or define a specific color palette to be used for the color reduction step, offering more artistic control beyond k-means generated palettes.
- **Batch Processing**:
    - Enable users to upload multiple images or specify a directory to process them all with the current settings, improving workflow for multiple images.
- **Live Preview or Parameter Combination Preview**:
    - Implement a faster, possibly lower-fidelity, live preview that updates as parameters are changed.
    - Alternatively, provide a way to generate small previews for a matrix of key parameter combinations (e.g., different scale factors and color counts).
- **Advanced Dithering Options**:
    - Expand on the planned dithering feature by offering multiple algorithms (e.g., Floyd-Steinberg, Ordered Dithering/Bayer Matrix, Atkinson).
- **Sharpening Filter**:
    - Add an optional sharpening filter (e.g., unsharp masking) that can be applied post-pixelation to enhance edges if desired.
- **UI/UX Enhancements**:
    - **Before/After Image Slider**: Implement an interactive slider over the output image to easily compare it with the original input.
    - **Parameter Presets**: Allow users to save their favorite parameter combinations as presets and load them later.
    - **Theme Options**: Simple light/dark mode for the UI.
    - **Progress Indication**: More explicit progress indication for longer processing tasks.
- **Output Options**:
    - **File Format Selection**: Allow choosing output format (e.g., PNG, GIF).
    - **Upscaling Factor for Output**: Allow users to specify a final upscaling factor for the pixel art (e.g., output at 2x, 4x the processed pixel art size, using nearest neighbor).
- **Performance Optimization**:
    - Profile existing code and optimize bottlenecks, especially in the `pixel_art_logic.py` module, for faster processing of larger images or more complex operations.
- **Error Handling and Reporting**:
    - Improve error handling for invalid inputs or processing failures, providing clearer feedback to the user via the UI.
