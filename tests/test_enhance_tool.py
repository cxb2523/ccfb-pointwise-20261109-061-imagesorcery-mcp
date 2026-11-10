"""Tests for the enhance_image MCP tool."""

import hashlib

import numpy as np
import pytest
from fastmcp import Client, FastMCP
from PIL import Image, ImageEnhance, ImageStat

from imagesorcery_mcp.server import mcp as image_sorcery_mcp_server
from imagesorcery_mcp.tools.enhance import _build_lut


@pytest.fixture
def mcp_server():
    # Use the existing server instance
    return image_sorcery_mcp_server


def _sha256(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def reference_lut(brightness: float, contrast: float, gamma: float, pivot: int) -> np.ndarray:
    """Independent float64 re-implementation of the fused LUT spec."""
    indices = np.arange(256, dtype=np.float64)
    values = indices * brightness
    values = pivot + (values - pivot) * contrast
    values = np.clip(values, 0.0, 255.0) / 255.0
    values = 255.0 * np.power(values, 1.0 / gamma)
    return np.rint(values).astype(np.uint8)


def pil_pivot(image: Image.Image) -> int:
    """PIL ImageEnhance.Contrast pivot: int(grayscale mean + 0.5)."""
    return int(ImageStat.Stat(image.convert("L")).mean[0] + 0.5)


@pytest.fixture
def gradient_path(tmp_path):
    """256-wide horizontal RGB gradient: column c has value c in every band."""
    path = tmp_path / "gradient.png"
    column = np.arange(256, dtype=np.uint8)
    frame = np.repeat(column[np.newaxis, :], 16, axis=0)
    rgb = np.stack([frame, frame, frame], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")
    return path


@pytest.fixture
def rgba_path(tmp_path):
    """RGBA image with a fully transparent 8x8 block in the top-left."""
    path = tmp_path / "rgba.png"
    arr = np.zeros((16, 16, 4), dtype=np.uint8)
    arr[..., 0] = 200
    arr[..., 1] = 100
    arr[..., 2] = 50
    arr[..., 3] = 255
    arr[:8, :8, 3] = 0  # transparent block
    Image.fromarray(arr, mode="RGBA").save(path, format="PNG")
    return path


@pytest.fixture
def palette_transparent_path(tmp_path, rgba_path):
    """P-mode PNG carrying palette transparency."""
    path = tmp_path / "palette_transparent.png"
    with Image.open(rgba_path) as rgba:
        quantized = rgba.quantize(colors=16, dither=Image.Dither.NONE)
        quantized.save(path, format="PNG")
    with Image.open(path) as reopened:
        assert "transparency" in reopened.info
    return path


@pytest.fixture
def palette_opaque_path(tmp_path):
    """P-mode PNG without any palette transparency."""
    path = tmp_path / "palette_opaque.png"
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[..., 0] = 180
    arr[..., 1] = 90
    arr[..., 2] = 30
    quantized = Image.fromarray(arr, mode="RGB").quantize(
        colors=16, dither=Image.Dither.NONE
    )
    quantized.save(path, format="PNG")
    with Image.open(path) as reopened:
        assert "transparency" not in reopened.info
    return path


class TestEnhanceToolDefinition:
    @pytest.mark.asyncio
    async def test_enhance_image_registered(self, mcp_server: FastMCP):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]
        assert "enhance_image" in tool_names

    @pytest.mark.asyncio
    async def test_parameter_schema(self, mcp_server: FastMCP):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        tool = next(t for t in tools if t.name == "enhance_image")
        properties = tool.inputSchema["properties"]

        for param in ["input_path", "brightness", "contrast", "gamma", "sharpness", "output_path"]:
            assert param in properties

        # gamma must be constrained to values strictly greater than 0
        assert properties["gamma"].get("exclusiveMinimum") == 0.0
        # point-operation parameters default to neutral values
        assert properties["brightness"]["default"] == 1.0
        assert properties["contrast"]["default"] == 1.0
        assert properties["gamma"]["default"] == 1.0
        assert properties["sharpness"]["default"] == 1.0


class TestNeutralFastPath:
    @pytest.mark.asyncio
    async def test_neutral_output_is_byte_identical_copy(
        self, mcp_server: FastMCP, gradient_path, tmp_path
    ):
        output_path = tmp_path / "neutral.png"
        source_hash_before = _sha256(gradient_path)

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "enhance_image",
                {"input_path": str(gradient_path), "output_path": str(output_path)},
            )

        assert result.data == str(output_path)
        assert output_path.exists()
        assert _sha256(output_path) == source_hash_before
        # The source file itself must never be modified
        assert _sha256(gradient_path) == source_hash_before

    @pytest.mark.asyncio
    async def test_neutral_output_pixel_identical(
        self, mcp_server: FastMCP, gradient_path, tmp_path
    ):
        output_path = tmp_path / "neutral_pixels.png"
        async with Client(mcp_server) as client:
            await client.call_tool(
                "enhance_image",
                {"input_path": str(gradient_path), "output_path": str(output_path)},
            )

        with Image.open(gradient_path) as src, Image.open(output_path) as out:
            assert np.array_equal(np.array(src), np.array(out))

    @pytest.mark.asyncio
    async def test_default_output_path_uses_enhanced_suffix(
        self, mcp_server: FastMCP, gradient_path
    ):
        expected = str(gradient_path).replace(".png", "_enhanced.png")
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "enhance_image", {"input_path": str(gradient_path)}
            )
        assert result.data == expected
        assert _sha256(expected) == _sha256(gradient_path)


class TestFusedLUT:
    @pytest.mark.asyncio
    async def test_brightness_only_sample_points(
        self, mcp_server: FastMCP, gradient_path, tmp_path
    ):
        output_path = tmp_path / "brightness.png"
        async with Client(mcp_server) as client:
            await client.call_tool(
                "enhance_image",
                {
                    "input_path": str(gradient_path),
                    "brightness": 1.5,
                    "output_path": str(output_path),
                },
            )

        with Image.open(gradient_path) as src:
            pivot = pil_pivot(src)
        expected = reference_lut(1.5, 1.0, 1.0, pivot)

        with Image.open(output_path) as out:
            pixels = np.array(out)
        # Gradient column c must map through the fused LUT to expected[c]
        for c in (0, 1, 3, 100, 200, 255):
            assert tuple(pixels[0, c]) == (expected[c], expected[c], expected[c])

    @pytest.mark.asyncio
    async def test_gamma_only_sample_points(
        self, mcp_server: FastMCP, gradient_path, tmp_path
    ):
        output_path = tmp_path / "gamma.png"
        async with Client(mcp_server) as client:
            await client.call_tool(
                "enhance_image",
                {
                    "input_path": str(gradient_path),
                    "gamma": 2.2,
                    "output_path": str(output_path),
                },
            )

        with Image.open(gradient_path) as src:
            pivot = pil_pivot(src)
        expected = reference_lut(1.0, 1.0, 2.2, pivot)

        with Image.open(output_path) as out:
            pixels = np.array(out)
        for c in (0, 16, 64, 128, 200, 255):
            assert tuple(pixels[0, c]) == (expected[c], expected[c], expected[c])

    @pytest.mark.asyncio
    async def test_combined_adjustments_sample_points(
        self, mcp_server: FastMCP, gradient_path, tmp_path
    ):
        output_path = tmp_path / "combined.png"
        brightness, contrast, gamma = 1.2, 0.8, 1.5
        async with Client(mcp_server) as client:
            await client.call_tool(
                "enhance_image",
                {
                    "input_path": str(gradient_path),
                    "brightness": brightness,
                    "contrast": contrast,
                    "gamma": gamma,
                    "output_path": str(output_path),
                },
            )

        with Image.open(gradient_path) as src:
            pivot = pil_pivot(src)
        expected = reference_lut(brightness, contrast, gamma, pivot)

        with Image.open(output_path) as out:
            pixels = np.array(out)
        for c in (0, 37, 100, 188, 255):
            assert tuple(pixels[0, c]) == (expected[c], expected[c], expected[c])
        # Every gradient column must match the fused LUT exactly
        for c in range(256):
            assert tuple(pixels[0, c]) == (expected[c], expected[c], expected[c])

    def test_lut_quantizes_round_half_even(self):
        # brightness 1.5 with contrast/gamma neutral: value for index i is 1.5*i
        lut = _build_lut(1.5, 1.0, 1.0, pivot=128)
        # 1.5 -> 2 (half rounds to even), 4.5 -> 4 (half rounds to even)
        assert lut[1] == 2
        assert lut[3] == 4
        # plain values are unaffected
        assert lut[0] == 0
        assert lut[2] == 3

    def test_lut_clips_and_matches_gamma_formula(self):
        # brightness 2.0 pushes high indices above 255; they must clip, not wrap
        lut = _build_lut(2.0, 1.0, 1.0, pivot=128)
        assert lut[255] == 255
        assert lut[128] == 255
        # Explicit gamma formula check for an interior index
        gamma = 1.5
        index = 100
        expected = int(np.rint(255.0 * (index / 255.0) ** (1.0 / gamma)))
        assert _build_lut(1.0, 1.0, gamma, pivot=0)[index] == expected


class TestAlphaAndPalette:
    @pytest.mark.asyncio
    async def test_rgba_alpha_preserved_untouched(
        self, mcp_server: FastMCP, rgba_path, tmp_path
    ):
        output_path = tmp_path / "rgba_enhanced.png"
        async with Client(mcp_server) as client:
            await client.call_tool(
                "enhance_image",
                {
                    "input_path": str(rgba_path),
                    "brightness": 1.3,
                    "gamma": 1.4,
                    "output_path": str(output_path),
                },
            )

        with Image.open(rgba_path) as src, Image.open(output_path) as out:
            assert out.mode == "RGBA"
            src_arr = np.array(src)
            out_arr = np.array(out)
            # Alpha channel comes back byte-for-byte
            assert np.array_equal(out_arr[..., 3], src_arr[..., 3])
            # Transparent block stays transparent, opaque block stays opaque
            assert out_arr[0, 0, 3] == 0
            assert out_arr[15, 15, 3] == 255
            # Color bands of opaque pixels follow the fused LUT
            pivot = pil_pivot(src.convert("RGB"))
            lut = reference_lut(1.3, 1.0, 1.4, pivot)
            for band in range(3):
                assert out_arr[15, 15, band] == lut[src_arr[15, 15, band]]

    @pytest.mark.asyncio
    async def test_palette_transparency_converted_to_rgba(
        self, mcp_server: FastMCP, palette_transparent_path, tmp_path
    ):
        output_path = tmp_path / "palette_out.png"
        async with Client(mcp_server) as client:
            await client.call_tool(
                "enhance_image",
                {
                    "input_path": str(palette_transparent_path),
                    "brightness": 1.2,
                    "output_path": str(output_path),
                },
            )

        with Image.open(output_path) as out:
            assert out.mode == "RGBA"
            out_arr = np.array(out)
            # The transparent palette entries must survive, not become black
            assert out_arr[0, 0, 3] == 0
            assert out_arr[15, 15, 3] == 255

    @pytest.mark.asyncio
    async def test_palette_without_transparency_converted_to_rgb(
        self, mcp_server: FastMCP, palette_opaque_path, tmp_path
    ):
        output_path = tmp_path / "palette_opaque_out.png"
        async with Client(mcp_server) as client:
            await client.call_tool(
                "enhance_image",
                {
                    "input_path": str(palette_opaque_path),
                    "contrast": 1.4,
                    "output_path": str(output_path),
                },
            )

        with Image.open(output_path) as out:
            assert out.mode == "RGB"


class TestSharpnessOrdering:
    @pytest.mark.asyncio
    async def test_sharpness_runs_after_lut_via_pil(
        self, mcp_server: FastMCP, tmp_path
    ):
        # Image with a hard edge so sharpening actually changes pixels
        path = tmp_path / "edge.png"
        arr = np.full((32, 32, 3), 40, dtype=np.uint8)
        arr[:, 16:, :] = 210
        Image.fromarray(arr, mode="RGB").save(path, format="PNG")

        output_path = tmp_path / "sharp.png"
        brightness, sharpness = 1.3, 2.0
        async with Client(mcp_server) as client:
            await client.call_tool(
                "enhance_image",
                {
                    "input_path": str(path),
                    "brightness": brightness,
                    "sharpness": sharpness,
                    "output_path": str(output_path),
                },
            )

        # Reconstruct the required ordering independently:
        # fused LUT first (back to 8-bit), PIL Sharpness only afterwards
        with Image.open(path) as src:
            pivot = pil_pivot(src)
            lut = reference_lut(brightness, 1.0, 1.0, pivot).tolist()
            after_lut = src.point(lut * 3)
            expected = ImageEnhance.Sharpness(after_lut).enhance(sharpness)

        with Image.open(output_path) as out:
            assert np.array_equal(np.array(out), np.array(expected))


class TestValidation:
    @pytest.mark.asyncio
    async def test_gamma_zero_rejected(
        self, mcp_server: FastMCP, gradient_path, tmp_path
    ):
        # The validation middleware rewrites pydantic messages into a generic
        # one on the wire; assert the call is rejected at the boundary.
        async with Client(mcp_server) as client:
            with pytest.raises(Exception) as excinfo:
                await client.call_tool(
                    "enhance_image",
                    {
                        "input_path": str(gradient_path),
                        "gamma": 0,
                        "output_path": str(tmp_path / "x.png"),
                    },
                )
        assert "enhance_image" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_invalid_gamma_pydantic_message_preserved_directly(
        self, mcp_server: FastMCP, gradient_path, tmp_path
    ):
        # Bypassing the middleware, pydantic reports the precise gt=0 constraint
        tool = await mcp_server.get_tool("enhance_image")
        for bad_gamma in (0.0, -2.0):
            with pytest.raises(Exception) as excinfo:
                await tool.run(
                    {
                        "input_path": str(gradient_path),
                        "gamma": bad_gamma,
                        "output_path": str(tmp_path / "x.png"),
                    }
                )
            assert "greater than 0" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_negative_gamma_rejected(
        self, mcp_server: FastMCP, gradient_path, tmp_path
    ):
        async with Client(mcp_server) as client:
            with pytest.raises(Exception) as excinfo:
                await client.call_tool(
                    "enhance_image",
                    {
                        "input_path": str(gradient_path),
                        "gamma": -1.5,
                        "output_path": str(tmp_path / "x.png"),
                    },
                )
        assert "enhance_image" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_output_same_as_input_rejected(
        self, mcp_server: FastMCP, gradient_path
    ):
        source_hash = _sha256(gradient_path)
        async with Client(mcp_server) as client:
            with pytest.raises(Exception) as excinfo:
                await client.call_tool(
                    "enhance_image",
                    {
                        "input_path": str(gradient_path),
                        "output_path": str(gradient_path),
                    },
                )
        assert "output_path must be different from input_path" in str(excinfo.value)
        # Rejected run must not have touched the source file
        assert _sha256(gradient_path) == source_hash

    @pytest.mark.asyncio
    async def test_missing_input_raises(self, mcp_server: FastMCP, tmp_path):
        missing = tmp_path / "does_not_exist.png"
        async with Client(mcp_server) as client:
            with pytest.raises(Exception) as excinfo:
                await client.call_tool(
                    "enhance_image",
                    {"input_path": str(missing), "output_path": str(tmp_path / "o.png")},
                )
        assert "Input file not found" in str(excinfo.value)


class TestFormatHandling:
    @pytest.mark.asyncio
    async def test_extension_change_forces_decode_reencode(
        self, mcp_server: FastMCP, gradient_path, tmp_path
    ):
        source_hash = _sha256(gradient_path)
        output_path = tmp_path / "reencoded.jpg"

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "enhance_image",
                {"input_path": str(gradient_path), "output_path": str(output_path)},
            )

        assert result.data == str(output_path)
        with Image.open(output_path) as out:
            assert out.format == "JPEG"
            assert out.size == (256, 16)
        # Even with neutral parameters, a format switch must not byte-copy
        assert _sha256(output_path) != source_hash
        assert _sha256(gradient_path) == source_hash

    @pytest.mark.asyncio
    async def test_alpha_incompatible_format_fails_explicitly(
        self, mcp_server: FastMCP, rgba_path, tmp_path
    ):
        output_path = tmp_path / "rgba_flatten_attempt.jpg"
        async with Client(mcp_server) as client:
            with pytest.raises(Exception) as excinfo:
                await client.call_tool(
                    "enhance_image",
                    {
                        "input_path": str(rgba_path),
                        "output_path": str(output_path),
                    },
                )
        message = str(excinfo.value)
        assert "RGBA" in message and "JPEG" in message
        # No silently-flattened output may be left behind
        assert not output_path.exists()
