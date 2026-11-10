import os
import shutil
from typing import Annotated, Optional

import numpy as np
from fastmcp import FastMCP
from PIL import Image, ImageEnhance, ImageStat
from pydantic import Field

# Import the central logger
from imagesorcery_mcp.logging_config import logger

# Reuse the same path resolution the PathAccessMiddleware relies on
from imagesorcery_mcp.middlewares.path_access import resolve_path


def _build_lut(brightness: float, contrast: float, gamma: float, pivot: int) -> np.ndarray:
    """Fuse brightness, contrast and gamma into one 256-entry LUT.

    All three adjustments are point operations, so they are evaluated
    together in float64 and quantized exactly once (round-half-even),
    avoiding the repeated 8-bit round-trips of chained ImageEnhance calls.

    Order of operations (matches the legacy client pipeline):
    1. brightness: multiplicative
    2. contrast: scale around the whole-image luminance mean (PIL semantics)
    3. gamma: 255 * (i/255) ** (1/gamma)
    """
    indices = np.arange(256, dtype=np.float64)

    # 1. Brightness
    values = indices * brightness

    # 2. Contrast around the PIL-style luminance pivot
    values = pivot + (values - pivot) * contrast

    # Gamma needs normalized inputs in [0, 1]; clip out-of-range values first
    values = np.clip(values, 0.0, 255.0) / 255.0

    # 3. Gamma
    values = 255.0 * np.power(values, 1.0 / gamma)

    # Single quantization step, round-half-even (banker's rounding)
    return np.rint(values).astype(np.uint8)


def register_tool(mcp: FastMCP):
    @mcp.tool()
    def enhance_image(
        input_path: Annotated[str, Field(description="Full path to the input image (must be a full path)")],
        brightness: Annotated[
            float,
            Field(description="Brightness factor. 1.0 is neutral (no change)."),
        ] = 1.0,
        contrast: Annotated[
            float,
            Field(description="Contrast factor. 1.0 is neutral (no change). Pivot is the whole-image luminance mean, matching PIL semantics."),
        ] = 1.0,
        gamma: Annotated[
            float,
            Field(gt=0.0, description="Gamma value, must be greater than 0. 1.0 is neutral (no change)."),
        ] = 1.0,
        sharpness: Annotated[
            float,
            Field(description="Sharpness factor handed to PIL after the point operations. 1.0 is neutral (no change)."),
        ] = 1.0,
        output_path: Annotated[
            Optional[str],
            Field(
                description=(
                    "Full path to save the output image (must be a full path). "
                    "If not provided, will use input filename "
                    "with '_enhanced' suffix."
                )
            ),
        ] = None,
    ) -> str:
        """
        Enhance an image with brightness, contrast, gamma and sharpness.

        Brightness, contrast and gamma are point operations fused into a
        single 256-level LUT in float64, with exactly one round-half-even
        quantization at the end, so results stay bit-stable instead of
        accumulating 8-bit rounding across separate ImageEnhance passes.
        Sharpness is a convolution and therefore applied afterwards via PIL.

        With every parameter neutral and an unchanged output format, the
        source file is copied byte-for-byte without being decoded. A
        different output extension forces a normal decode/re-encode.

        Returns:
            Path to the enhanced image
        """
        logger.info(
            f"Enhance image tool requested for image: {input_path}, "
            f"brightness: {brightness}, contrast: {contrast}, gamma: {gamma}, "
            f"sharpness: {sharpness}"
        )

        # Check if input file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(
                f"Input file not found: {input_path}. "
                "Please provide a full path to the file."
            )

        # Generate output path if not provided
        if not output_path:
            file_name, file_ext = os.path.splitext(input_path)
            output_path = f"{file_name}_enhanced{file_ext}"
            logger.info(f"Output path not provided, generated: {output_path}")

        # Reject outputs that resolve to the source file itself
        resolved_input = resolve_path(input_path)
        resolved_output = resolve_path(output_path)
        if resolved_output == resolved_input:
            logger.error(f"Output path resolves to the same file as input: {resolved_output}")
            raise ValueError(
                "output_path must be different from input_path; "
                f"both resolve to: {resolved_output}"
            )

        is_neutral = (
            brightness == 1.0
            and contrast == 1.0
            and gamma == 1.0
            and sharpness == 1.0
        )
        source_ext = os.path.splitext(input_path)[1].lower()
        target_ext = os.path.splitext(output_path)[1].lower()
        format_changed = target_ext != source_ext

        # Fast path: nothing to do and the container format is unchanged,
        # so copy the raw bytes without ever decoding the image
        if is_neutral and not format_changed:
            logger.info(
                "All enhancement parameters are neutral and format is unchanged; "
                f"copying source bytes to: {output_path}"
            )
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            shutil.copyfile(input_path, output_path)
            logger.info(f"Image copied unchanged to: {output_path}")
            return output_path

        logger.info(f"Opening image with PIL: {input_path}")
        with Image.open(input_path) as img:
            source_format = img.format
            source_mode = img.mode

            # Palette images: keep palette transparency instead of flattening
            # transparent entries onto black via a plain convert('RGB')
            if source_mode == "P":
                if "transparency" in img.info:
                    logger.info("P-mode image has palette transparency, converting to RGBA")
                    work = img.convert("RGBA")
                else:
                    logger.info("P-mode image without transparency, converting to RGB")
                    work = img.convert("RGB")
            elif source_mode in ("RGB", "RGBA", "L"):
                work = img.copy()
            else:
                logger.info(f"Converting image from mode {source_mode} to RGB")
                work = img.convert("RGB")

            # Split alpha off so the point operations only touch color bands
            alpha = None
            if work.mode == "RGBA":
                alpha = work.getchannel("A")
                work = work.convert("RGB")

            # PIL Contrast pivot: int(mean of grayscale luminance + 0.5)
            pivot = int(ImageStat.Stat(work.convert("L")).mean[0] + 0.5)
            logger.info(f"Contrast pivot (whole-image luminance mean): {pivot}")

            lut = _build_lut(brightness, contrast, gamma, pivot)
            logger.info("Applying fused brightness/contrast/gamma LUT")

            if work.mode == "L":
                enhanced = work.point(lut.tolist())
            else:
                # One LUT per color band; no alpha band is present here
                enhanced = work.point(lut.tolist() * len(work.getbands()))

            # Sharpness is a convolution and cannot be expressed in a LUT;
            # run it only after the point operations have landed back in 8-bit
            if sharpness != 1.0:
                logger.info(f"Applying PIL sharpness factor: {sharpness}")
                enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)

            # Put the original alpha back untouched
            if alpha is not None:
                enhanced.putalpha(alpha)

            # Create directory for output if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                logger.info(f"Output directory does not exist, creating: {output_dir}")
                os.makedirs(output_dir)

            # When the extension changed, encode explicitly in the new format.
            # Let saving fail explicitly if the target format cannot hold alpha
            # (e.g. RGBA -> JPEG) instead of silently flattening transparency.
            target_format = None
            if format_changed and target_ext:
                Image.init()
                target_format = Image.registered_extensions().get(target_ext)

            logger.info(
                f"Saving enhanced image to: {output_path} "
                f"(source format: {source_format}, target format: {target_format})"
            )
            if target_format is not None:
                enhanced.save(output_path, format=target_format)
            else:
                enhanced.save(output_path)

        logger.info(f"Enhanced image saved successfully to: {output_path}")
        return output_path
