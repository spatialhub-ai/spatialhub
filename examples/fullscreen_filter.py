"""Hardware-accelerated single-pass and multi-pass fullscreen quad shader filtering.

Executes a single-pass Sobel spatial gradient filter and an iterative multi-pass
Gaussian smoothing filter with ping-pong framebuffer swapping using FullscreenRenderer.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from spatialhub.utils.image import load_image, save_image
from spatialhub.utils.moderngl import FullscreenRenderer

logger = logging.getLogger(__name__)


def build_comparison_canvas(
    original: np.ndarray,
    gradient: np.ndarray,
    blurred: np.ndarray,
) -> np.ndarray:
    """Concatenate original, gradient magnitude, and blurred frames horizontally.

    Args:
        original: Source RGB image of shape (H, W, 3) in [0, 1].
        gradient: Gradient array of shape (H, W, 4) or (H, W, 3) in [0, 1].
        blurred: Filtered RGB image of shape (H, W, 3) in [0, 1].

    Returns:
        Side-by-side composite array of shape (H, W * 3, 3) in uint8 [0, 255].
    """
    orig_u8 = np.clip(original * 255.0, 0, 255).astype(np.uint8)

    grad_mag = gradient[..., 0] if gradient.ndim == 3 else gradient
    grad_u8 = np.clip(grad_mag * 255.0, 0, 255).astype(np.uint8)
    if grad_u8.ndim == 2:
        grad_u8 = np.repeat(grad_u8[..., np.newaxis], 3, axis=2)

    blur_u8 = np.clip(blurred * 255.0, 0, 255).astype(np.uint8)

    return np.concatenate([orig_u8, grad_u8, blur_u8], axis=1)


def load_input_frame(image_path: str | Path) -> np.ndarray:
    """Read RGB image and normalize intensities to contiguous float32 [0.0, 1.0].

    Args:
        image_path: Path to input image file.

    Returns:
        Contiguous float32 array of shape (H, W, 3) in [0.0, 1.0].
    """
    raw_rgb = load_image(image_path, color_mode="RGB")
    return np.ascontiguousarray(raw_rgb, dtype=np.float32) / 255.0


def main(
    image_path: str = "examples/assets/room_image.jpg",
    num_passes: int = 5,
    output_dir: str = "examples/output",
    update_docs: bool = False,
) -> None:
    target_dir = Path("docs/assets") if update_docs else Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # --8<-- [start:load_image]
    # Load input RGB image and normalize pixel intensity to float32 [0.0, 1.0]
    image_f32 = load_input_frame(image_path)
    height, width, channels = image_f32.shape

    print(f"Loaded image:         {image_path}")
    print(f"Image resolution:     {width}x{height} ({channels} channels)")
    # --8<-- [end:load_image]

    # --8<-- [start:vert_shader]
    # Shared vertex shader: maps NDC [-1, 1] quad vertices to [0, 1] texture coordinates
    vert_shader = """
    #version 330
    in vec2 in_position;
    out vec2 v_uv;

    void main() {
        v_uv = in_position * 0.5 + 0.5;
        gl_Position = vec4(in_position, 0.0, 1.0);
    }
    """
    # --8<-- [end:vert_shader]

    # --8<-- [start:single_pass_shader]
    # Sobel spatial gradient filter computing directional derivatives and magnitude
    sobel_frag_shader = """
    #version 330
    uniform sampler2D tex_in;
    in vec2 v_uv;
    out vec4 fragColor;

    void main() {
        vec2 texel = 1.0 / vec2(textureSize(tex_in, 0));

        // Convert RGB neighborhood to relative luminance
        vec3 w = vec3(0.299, 0.587, 0.114);
        float p00 = dot(texture(tex_in, v_uv + vec2(-texel.x, -texel.y)).rgb, w);
        float p10 = dot(texture(tex_in, v_uv + vec2( 0.0,     -texel.y)).rgb, w);
        float p20 = dot(texture(tex_in, v_uv + vec2( texel.x, -texel.y)).rgb, w);

        float p01 = dot(texture(tex_in, v_uv + vec2(-texel.x,  0.0)).rgb, w);
        float p21 = dot(texture(tex_in, v_uv + vec2( texel.x,  0.0)).rgb, w);

        float p02 = dot(texture(tex_in, v_uv + vec2(-texel.x,  texel.y)).rgb, w);
        float p12 = dot(texture(tex_in, v_uv + vec2( 0.0,      texel.y)).rgb, w);
        float p22 = dot(texture(tex_in, v_uv + vec2( texel.x,  texel.y)).rgb, w);

        float gx = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02);
        float gy = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20);
        float mag = clamp(length(vec2(gx, gy)) * 2.0, 0.0, 1.0);

        fragColor = vec4(mag, abs(gx), abs(gy), 1.0);
    }
    """
    # --8<-- [end:single_pass_shader]

    # --8<-- [start:multi_pass_shader]
    # 2D discrete Gaussian blur kernel for iterative ping-pong smoothing
    gaussian_frag_shader = """
    #version 330
    uniform sampler2D tex_in;
    in vec2 v_uv;
    out vec4 fragColor;

    void main() {
        vec2 texel = 1.0 / vec2(textureSize(tex_in, 0));

        vec4 color = vec4(0.0);
        color += texture(tex_in, v_uv + vec2(-texel.x, -texel.y)) * 0.0625;
        color += texture(tex_in, v_uv + vec2( 0.0,     -texel.y)) * 0.1250;
        color += texture(tex_in, v_uv + vec2( texel.x, -texel.y)) * 0.0625;

        color += texture(tex_in, v_uv + vec2(-texel.x,  0.0))     * 0.1250;
        color += texture(tex_in, v_uv + vec2( 0.0,      0.0))     * 0.2500;
        color += texture(tex_in, v_uv + vec2( texel.x,  0.0))     * 0.1250;

        color += texture(tex_in, v_uv + vec2(-texel.x,  texel.y)) * 0.0625;
        color += texture(tex_in, v_uv + vec2( 0.0,      texel.y)) * 0.1250;
        color += texture(tex_in, v_uv + vec2( texel.x,  texel.y)) * 0.0625;

        fragColor = color;
    }
    """
    # --8<-- [end:multi_pass_shader]

    with FullscreenRenderer() as renderer:
        # Compile GLSL shader programs
        prog_sobel = renderer.create_program(vert_shader, sobel_frag_shader)
        prog_gaussian = renderer.create_program(vert_shader, gaussian_frag_shader)

        # --8<-- [start:single_pass_execution]
        # Single-pass execution: upload, evaluate fragment shader on quad, and read back
        gradient_output = renderer.apply(prog=prog_sobel, input_data=image_f32)
        print(f"Gradient output:      {gradient_output.shape} ({gradient_output.dtype})")
        # --8<-- [end:single_pass_execution]

        # --8<-- [start:multi_pass_execution]
        # Multi-pass execution: upload once, iteratively swap ping-pong FBOs, and read back
        renderer.upload(image_f32)
        for _ in range(num_passes):
            renderer.render_pass(prog=prog_gaussian)
            renderer.swap_buffers()

        blurred_output = renderer.to_numpy()
        print(f"Blurred output:       {blurred_output.shape} ({blurred_output.dtype}) after {num_passes} passes")
        # --8<-- [end:multi_pass_execution]

        # --8<-- [start:cleanup]
        # Program resource cleanup
        prog_sobel.release()
        prog_gaussian.release()
        # --8<-- [end:cleanup]

    # --8<-- [start:export_artifacts]
    # Export individual filtered arrays and comparison canvas
    src_png_path = target_dir / "fullscreen_source_input.png"
    grad_png_path = target_dir / "fullscreen_sobel_gradient.png"
    blur_png_path = target_dir / "fullscreen_gaussian_blur.png"
    comp_png_path = target_dir / "fullscreen_filter_comparison.png"

    src_u8 = np.clip(image_f32 * 255.0, 0, 255).astype(np.uint8)
    grad_mag = np.clip(gradient_output[..., 0] * 255.0, 0, 255).astype(np.uint8)
    blur_u8 = np.clip(blurred_output[..., :3] * 255.0, 0, 255).astype(np.uint8)
    comparison_canvas = build_comparison_canvas(image_f32, gradient_output, blurred_output[..., :3])

    save_image(src_u8, src_png_path, color_mode="RGB")
    save_image(grad_mag, grad_png_path, color_mode="GRAY")
    save_image(blur_u8, blur_png_path, color_mode="RGB")
    save_image(comparison_canvas, comp_png_path, color_mode="RGB")

    print(f"Saved source frame:   {src_png_path}")
    print(f"Saved Sobel gradient: {grad_png_path}")
    print(f"Saved Gaussian blur:  {blur_png_path}")
    print(f"Saved comparison:     {comp_png_path}")
    # --8<-- [end:export_artifacts]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fullscreen Quad Shader Filtering")
    parser.add_argument(
        "--image-path",
        type=str,
        default="examples/assets/room_image.jpg",
        help="Path to input image file (default: examples/assets/room_image.jpg)",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=5,
        help="Number of ping-pong smoothing iterations (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/output",
        help="Directory for saved artifacts (default: examples/output)",
    )
    parser.add_argument(
        "--update-docs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    main(
        image_path=args.image_path,
        num_passes=args.num_passes,
        output_dir=args.output_dir,
        update_docs=args.update_docs,
    )
