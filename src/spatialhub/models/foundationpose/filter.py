"""
Depth map filtering pipeline implementing erosion and bilateral smoothing.
"""

from __future__ import annotations

import moderngl
import numpy as np

from spatialhub.utils import FullscreenShader

# ---------------------------------------------------------
# Shader Definitions
# ---------------------------------------------------------
QUAD_VS = """
    #version 330
    in vec2 in_position;
    void main() { gl_Position = vec4(in_position, 0.0, 1.0); }
"""

ERODE_FS = """
    #version 330
    out vec4 f_color;
    uniform sampler2D tex_depth;
    uniform int radius;
    uniform float depth_diff_thres;
    uniform float ratio_thres;
    uniform float zfar;
    uniform vec2 resolution;

    void main() {
        ivec2 center = ivec2(gl_FragCoord.xy);
        float d_ori = texelFetch(tex_depth, center, 0).r;
        
        if (d_ori < 0.001 || d_ori >= zfar) {
            f_color = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        
        float bad_cnt = 0.0;
        float total = 0.0;
        for (int u = -radius; u <= radius; ++u) {
            for (int v = -radius; v <= radius; ++v) {
                ivec2 p = center + ivec2(u, v);
                if (p.x < 0 || p.x >= int(resolution.x) || p.y < 0 || p.y >= int(resolution.y)) continue;
                float cur = texelFetch(tex_depth, p, 0).r;
                total += 1.0;
                if (cur < 0.001 || cur >= zfar || abs(cur - d_ori) > depth_diff_thres) {
                    bad_cnt += 1.0;
                }
            }
        }
        f_color = (bad_cnt / total > ratio_thres) ? vec4(0.0, 0.0, 0.0, 1.0) : vec4(d_ori, 0.0, 0.0, 1.0);
    }
"""

BILATERAL_FS = """
    #version 330
    out vec4 f_color;
    uniform sampler2D tex_depth;
    uniform int radius;
    uniform float zfar;
    uniform float sigmaD;
    uniform float sigmaR;
    uniform vec2 resolution;

    void main() {
        ivec2 center = ivec2(gl_FragCoord.xy);
        float d_ori = texelFetch(tex_depth, center, 0).r;
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        
        float mean_depth = 0.0;
        float num_valid = 0.0;
        
        for (int u = -radius; u <= radius; ++u) {
            for (int v = -radius; v <= radius; ++v) {
                ivec2 p = center + ivec2(u, v);
                if (p.x < 0 || p.x >= int(resolution.x) || p.y < 0 || p.y >= int(resolution.y)) continue;
                float cur = texelFetch(tex_depth, p, 0).r;
                if (cur >= 0.001 && cur < zfar) {
                    num_valid += 1.0;
                    mean_depth += cur;
                }
            }
        }
        if (num_valid == 0.0) return;
        mean_depth /= num_valid;
        
        float sum_weight = 0.0;
        float sum_val = 0.0;
        for (int u = -radius; u <= radius; ++u) {
            for (int v = -radius; v <= radius; ++v) {
                ivec2 p = center + ivec2(u, v);
                if (p.x < 0 || p.x >= int(resolution.x) || p.y < 0 || p.y >= int(resolution.y)) continue;
                float cur = texelFetch(tex_depth, p, 0).r;
                if (cur >= 0.001 && cur < zfar && abs(cur - mean_depth) < 0.01) {
                    float dist_sq = float(u*u + v*v);
                    float range_sq = (d_ori - cur)*(d_ori - cur);
                    float weight = exp(-dist_sq / (2.0*sigmaD*sigmaD) - range_sq / (2.0*sigmaR*sigmaR));
                    sum_weight += weight;
                    sum_val += weight * cur;
                }
            }
        }
        if (sum_weight > 0.0 && num_valid > 0.0) {
            f_color = vec4(sum_val / sum_weight, 0.0, 0.0, 1.0);
        }
    }
"""


class DepthFilter:
    """
    Depth map filter executing consecutive erosion and bilateral smoothing passes.
    """

    def __init__(self, glctx: moderngl.Context):
        """
        Initialize shader programs, vertex buffers, and vertex array objects.

        Args:
            glctx: Execution context instance.
        """
        self.glctx = glctx
        self.engine = FullscreenShader(glctx)
        self.quad_vbo = self.engine.create_quad_vbo()

        # Compile shader programs
        self.erode_prog = self.engine.create_program(QUAD_VS, ERODE_FS)
        self.bilateral_prog = self.engine.create_program(QUAD_VS, BILATERAL_FS)

        # Create Vertex Array Objects
        self.erode_vao = self.engine.create_vao(self.erode_prog, self.quad_vbo)
        self.bilateral_vao = self.engine.create_vao(self.bilateral_prog, self.quad_vbo)

    def apply(
        self,
        depth: np.ndarray,
        radius: int = 2,
        zfar: float = 100.0,
        depth_diff_thres: float = 0.001,
        ratio_thres: float = 0.8,
        sigmaD: float = 2.0,
        sigmaR: float = 100000.0,
    ) -> np.ndarray:
        """
        Execute depth erosion and bilateral smoothing sequentially.

        Args:
            depth: Input depth map array of shape (H, W) in float32.
            radius: Spatial filter neighborhood kernel radius in pixels (default: 2).
            zfar: Maximum far clipping plane threshold in meters (default: 100.0).
            depth_diff_thres: Neighborhood depth discontinuity threshold in meters.
            ratio_thres: Outlier ratio threshold for pixel invalidation.
            sigmaD: Spatial Gaussian domain variance parameter.
            sigmaR: Range Gaussian intensity variance parameter.

        Returns:
            Filtered depth map as a float32 array of shape (H, W).
        """
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        H, W = depth.shape[:2]

        # Phase 1: Upload input depth array
        self.engine.upload(depth)

        # Phase 2: Execute depth erosion pass
        self.erode_prog["resolution"].value = (float(W), float(H))
        self.erode_prog["radius"].value = int(radius)
        self.erode_prog["zfar"].value = float(zfar)
        self.erode_prog["depth_diff_thres"].value = float(depth_diff_thres)
        self.erode_prog["ratio_thres"].value = float(ratio_thres)

        self.engine.render(self.erode_vao, self.erode_prog, texture_uniform_name="tex_depth")

        # Swap textures for sequential filtering pass
        self.engine.swap_buffers()

        # Phase 3: Execute bilateral smoothing pass
        self.bilateral_prog["resolution"].value = (float(W), float(H))
        self.bilateral_prog["radius"].value = int(radius)
        self.bilateral_prog["zfar"].value = float(zfar)
        self.bilateral_prog["sigmaD"].value = float(sigmaD)
        self.bilateral_prog["sigmaR"].value = float(sigmaR)

        self.engine.render(self.bilateral_vao, self.bilateral_prog, texture_uniform_name="tex_depth")

        # Phase 4: Retrieve filtered output array
        return self.engine.to_numpy()

    def release(self) -> None:
        """Release compiled shader programs, vertex buffers, and framebuffer allocations."""
        self.quad_vbo.release()
        self.erode_prog.release()
        self.bilateral_prog.release()
        self.erode_vao.release()
        self.bilateral_vao.release()
        self.engine.release_fbo()
