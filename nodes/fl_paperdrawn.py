import glfw
import ctypes
import torch
import numpy as np
from PIL import Image
import OpenGL.GL as gl

from comfy.utils import ProgressBar

VERTEX_SHADER = """
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

FRAGMENT_SHADER = """
#version 330 core

out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform vec3 iResolution;
uniform float iTime;
uniform float iAngleNum;
uniform float iSampNum;
uniform float iLineWidth;
uniform float iVignette;

#define Res0 textureSize(iChannel0, 0)
#define Res1 textureSize(iChannel1, 0)
#define Res  iResolution.xy

#define randSamp iChannel1
#define colorSamp iChannel0

vec4 getRand(vec2 pos)
{
    return textureLod(iChannel1, pos / Res1 / iResolution.y * 1080., 0.0);
}

vec4 getCol(vec2 pos)
{
    vec2 uv = ((pos - Res.xy * .5) / Res.y * Res0.y) / Res0.xy + .5;
    vec4 c1 = texture(iChannel0, uv);
    vec4 e = smoothstep(vec4(-0.05), vec4(-0.0), vec4(uv, vec2(1) - uv));
    c1 = mix(vec4(1, 1, 1, 0), c1, e.x * e.y * e.z * e.w);
    float d = clamp(dot(c1.xyz, vec3(-.5, 1., -.5)), 0.0, 1.0);
    vec4 c2 = vec4(.7);
    return min(mix(c1, c2, 1.8 * d), .7);
}

vec4 getColHT(vec2 pos)
{
    return smoothstep(.95, 1.05, getCol(pos) * .8 + .2 + getRand(pos * .7));
}

float getVal(vec2 pos)
{
    vec4 c = getCol(pos);
    return pow(dot(c.xyz, vec3(.333)), 1.) * 1.;
}

vec2 getGrad(vec2 pos, float eps)
{
    vec2 d = vec2(eps, 0);
    return vec2(
        getVal(pos + d.xy) - getVal(pos - d.xy),
        getVal(pos + d.yx) - getVal(pos - d.yx)
    ) / eps / 2.;
}

#define PI2 6.28318530717959

void main()
{
    vec2 pos = TexCoord * iResolution.xy + 4.0 * sin(iTime * 1. * vec2(1, 1.7)) * iResolution.y / 400.;
    vec3 col = vec3(0);
    vec3 col2 = vec3(0);
    float sum = 0.;
    for (int i = 0; i < int(iAngleNum); i++)
    {
        float ang = PI2 / iAngleNum * (float(i) + .8);
        vec2 v = vec2(cos(ang), sin(ang));
        for (int j = 0; j < int(iSampNum); j++)
        {
            vec2 dpos = v.yx * vec2(1, -1) * float(j) * iLineWidth * iResolution.y / 400.;
            vec2 dpos2 = v.xy * float(j * j) / iSampNum * .5 * iLineWidth * iResolution.y / 400.;
            vec2 g;
            float fact;
            float fact2;

            for (float s = -1.; s <= 1.; s += 2.)
            {
                vec2 pos2 = pos + s * dpos + dpos2;
                vec2 pos3 = pos + (s * dpos + dpos2).yx * vec2(1, -1) * 2.;
                g = getGrad(pos2, .4);
                fact = dot(g, v) - .5 * abs(dot(g, v.yx * vec2(1, -1)));
                fact2 = dot(normalize(g + vec2(.0001)), v.yx * vec2(1, -1));

                fact = clamp(fact, 0., .05);
                fact2 = abs(fact2);

                fact *= 1. - float(j) / iSampNum;
                col += fact;
                col2 += fact2 * getColHT(pos3).xyz;
                sum += fact2;
            }
        }
    }
    col /= iSampNum * iAngleNum * .75 / sqrt(iResolution.y);
    col2 /= sum;
    col.x *= (.6 + .8 * getRand(pos * .7).x);
    col.x = 1. - col.x;
    col.x *= col.x * col.x;

    vec2 s = sin(pos.xy * .1 / sqrt(iResolution.y / 400.));
    vec3 karo = vec3(1);
    karo -= .5 * vec3(.25, .1, .1) * dot(exp(-s * s * 80.), vec2(1));
    float r = length(pos - iResolution.xy * .5) / iResolution.x;
    float vign = 1. - r * r * r * iVignette;
    FragColor = vec4(vec3(col.x * col2 * karo * vign), 1);
}
"""

class FL_PaperDrawn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "angle_num": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 1.0}),
                "samp_num": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 10.0, "step": 0.1}),
                "line_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "vignette": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_shader"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def apply_shader(self, image, angle_num, samp_num, line_width, vignette, fps):
        result = []
        total_images = len(image)
        frame_time = 1.0 / fps

        pbar = ProgressBar(total_images)
        for i, img in enumerate(image, start=1):
            img = self.t2p(img)
            result_img = self.process_image(img, angle_num, samp_num, line_width, vignette, i * frame_time)
            result_img = self.p2t(result_img)
            result.append(result_img)
            pbar.update_absolute(i)

        return (torch.cat(result, dim=0),)

    def process_image(self, image, angle_num, samp_num, line_width, vignette, time):
        # Convert the PIL image to a numpy array
        img_array = np.array(image).astype(np.float32) / 255.0

        # Create a white image for iChannel1
        white_image = np.ones((image.height, image.width, 3), dtype=np.float32)

        # Create a PyOpenGL context
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(image.width, image.height, "Hidden Window", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(window)

        # Compile the shader program
        vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vertex_shader, VERTEX_SHADER)
        gl.glCompileShader(vertex_shader)

        fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fragment_shader, FRAGMENT_SHADER)
        gl.glCompileShader(fragment_shader)

        shader_program = gl.glCreateProgram()
        gl.glAttachShader(shader_program, vertex_shader)
        gl.glAttachShader(shader_program, fragment_shader)
        gl.glLinkProgram(shader_program)

        gl.glUseProgram(shader_program)

        # Set up vertex buffer object (VBO) and vertex array object (VAO)
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0, 0.0,
            1.0, -1.0, 0.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 1.0, 1.0
        ], dtype=np.float32)

        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)

        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 5 * vertices.itemsize, None)
        gl.glEnableVertexAttribArray(0)

        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
        gl.glEnableVertexAttribArray(1)

        # Set up textures
        texture0 = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture0)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, image.width, image.height, 0, gl.GL_RGB, gl.GL_FLOAT, img_array)

        texture1 = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture1)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, image.width, image.height, 0, gl.GL_RGB, gl.GL_FLOAT, white_image)

        # Set shader uniforms
        gl.glUniform1i(gl.glGetUniformLocation(shader_program, "iChannel0"), 0)
        gl.glUniform1i(gl.glGetUniformLocation(shader_program, "iChannel1"), 1)
        gl.glUniform3f(gl.glGetUniformLocation(shader_program, "iResolution"), image.width, image.height, 0.0)
        gl.glUniform1f(gl.glGetUniformLocation(shader_program, "iTime"), time)
        gl.glUniform1f(gl.glGetUniformLocation(shader_program, "iAngleNum"), angle_num)
        gl.glUniform1f(gl.glGetUniformLocation(shader_program, "iSampNum"), samp_num)
        gl.glUniform1f(gl.glGetUniformLocation(shader_program, "iLineWidth"), line_width)
        gl.glUniform1f(gl.glGetUniformLocation(shader_program, "iVignette"), vignette)

        # Render the shader
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

        # Read the rendered image from the framebuffer
        img_data = gl.glReadPixels(0, 0, image.width, image.height, gl.GL_RGB, gl.GL_FLOAT)
        img_array = np.frombuffer(img_data, dtype=np.float32).reshape((image.height, image.width, 3))

        # Clean up OpenGL resources
        gl.glDeleteTextures(2, [texture0, texture1])
        gl.glDeleteBuffers(1, [vbo])
        gl.glDeleteVertexArrays(1, [vao])
        gl.glDeleteProgram(shader_program)
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)

        glfw.destroy_window(window)
        glfw.terminate()

        # Convert the processed image back to a PIL image
        processed_image = Image.fromarray((img_array * 255).astype(np.uint8))

        return processed_image

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0)
        return t