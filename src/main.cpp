#include <iostream>

#define GLEW_STATIC
#include "GL/glew.h"

#define FREEGLUT_STATIC
#include "GL/freeglut.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

#include "cudahelperlib.h"
#include "kernel.h"

using namespace std;

// Global Variables
GLuint texture_id;       // Main texture ID
GLuint vertex_buffer;    // VBO
GLuint shaders_program;  // Compiled shaders

const unsigned int width  = 512;
const unsigned int height = 512;

cudaArray * graphics_array;

cudaGraphicsResource * graphics_resource; // GL and CUDA shared resource

// Functions Declaration
void drawLoop(void);
void glErrorCheck();
GLuint loadShaders(const std::string vertex_code,
                   const std::string fragment_code);

int main(int argc, char **argv)
{

    // GLUT Setup
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(width, height);
    glutCreateWindow("OpenGL and CUDA");
    glutDisplayFunc(drawLoop);
    glutIdleFunc(drawLoop);


    // GLEW Setup
    glewExperimental = true;
    auto glewReturn = glewInit();
    if (glewReturn != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW: %s\n",
                glewGetErrorString(glewReturn));
        return -1;
    }

    // OpenGl Setup

    // There is no need for GL depth test since we will be working with
    // one texture that will occupy the whole screen
    glDisable(GL_DEPTH_TEST);

    glClearColor(0.0, 0.0, 0.0, 1.0); // Black background
    glViewport(0, 0, width, height);  // GL Screen size

    glErrorCheck();

    // Creating the VAO
    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // Plane Vertices
    static const GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
    };

    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glErrorCheck();

    // Texture Creation
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    // For cuda gl interop it is need a texture with 4, 2, or 1 floating point
    // component
    // http://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA,
                 GL_FLOAT, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    glErrorCheck();

    // Loading Shaders: These shaders only textures the plane created before
    shaders_program = loadShaders(
        // Vertex Shader Code
        "#version 330 core\n"
        "layout(location = 0) in vec3 position;"
        "out vec2 UV;"
        "void main() {"
        "    gl_Position = vec4(position, 1.0);"
        "    UV = vec2(position) * 0.5 + 0.5;"
        "}",

        // Fragment Shader Code
        "#version 330 core\n"
        "in vec2 UV;"
        "out vec3 color;"
        "uniform sampler2D textureSampler;"
        "void main() {"
        "    color = texture(textureSampler, UV).rgb;"
        "}"
    );

    glErrorCheck();


    // CUDA setup
    int device_count;
    cudaErrorCheck(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
        fprintf(stderr, "CUDA Error: No cuda device found");
    else
        cudaErrorCheck(cudaSetDevice(0));

    cudaErrorCheck(cudaGraphicsGLRegisterImage(
        &graphics_resource, texture_id, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore)
    );



    // Draw loop
    glutMainLoop();


    // Cleaning up
    glDeleteTextures(1, &texture_id);
    glDeleteBuffers(1, &vertex_buffer);
    glDeleteProgram(shaders_program);

    cudaErrorCheck(cudaGraphicsUnregisterResource(graphics_resource));

    return 0;
}


// Function Definitions

void drawLoop(void)
{
    // CUDA Loop
    cudaErrorCheck(cudaGraphicsMapResources(1, &graphics_resource, 0));

    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(
        &graphics_array, graphics_resource, 0, 0)
    );

    // Kernel Launch
    kernelCall(width, height, graphics_array);

    cudaErrorCheck(cudaGraphicsUnmapResources(1, &graphics_resource, 0));


    // OpenGL Loop
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaders_program);

    glBindTexture(GL_TEXTURE_2D, texture_id);

    // Vertex buffer at location=0
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glDrawArrays(GL_TRIANGLES, 0, 3 * 2);

    glErrorCheck();

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);

    glutSwapBuffers();
}

void glErrorCheck()
{
    GLenum err = GL_NO_ERROR;
    while((err = glGetError()) != GL_NO_ERROR)
        fprintf(stderr, "GL Error: %s\n", gluErrorString(err));
}

GLuint loadShaders(const std::string vertex_code,
                   const std::string fragment_code)
{
    auto compileShader = [](GLuint & shader_id,
                            const std::string shader_code) -> bool
    {
        char const * source_ptr = shader_code.c_str();
        glShaderSource(shader_id, 1, &source_ptr, NULL);
        glCompileShader(shader_id);

        GLint result = GL_FALSE;
        int log_length;

        glGetShaderiv(shader_id, GL_COMPILE_STATUS, &result);
        glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &log_length);

        if (log_length > 0)
        {
            GLchar * error_msg = new GLchar[log_length + 1];
            glGetShaderInfoLog(shader_id, log_length, NULL, error_msg);
            fprintf(stderr, "Shaders: %s\n", error_msg);
        }

        return result != GL_FALSE;
    };

    GLuint vertex_shader_id = glCreateShader(GL_VERTEX_SHADER);
    auto vertex_shader_status = compileShader(vertex_shader_id,
                                              vertex_code);

    GLuint frag_shader_id = glCreateShader(GL_FRAGMENT_SHADER);
    auto frag_shader_status = compileShader(frag_shader_id,
                                            fragment_code);

    if (!vertex_shader_status || !frag_shader_status)
        fprintf(stderr, "Shaders: Could not compile shaders!");

    GLuint program_id = glCreateProgram();
    glAttachShader(program_id, vertex_shader_id);
    glAttachShader(program_id, frag_shader_id);
    glLinkProgram(program_id);

    GLint result = GL_FALSE;
    int log_length;

    glGetProgramiv(program_id, GL_LINK_STATUS, &result);
    glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &log_length);

    if (log_length > 0)
    {
        GLchar * error_msg = new GLchar[log_length + 1];
        glGetProgramInfoLog(program_id, log_length, NULL, error_msg);
        fprintf(stderr, "Shaders: %s\n", error_msg);
    }

    glDetachShader(program_id, vertex_shader_id);
    glDetachShader(program_id, frag_shader_id);

    glDeleteShader(vertex_shader_id);
    glDeleteShader(frag_shader_id);

    return program_id;
}
