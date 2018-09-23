#pragma once

#include <iostream>
#include <cassert>
#include <vector>
#include "glatter/glatter.h"
#include "Reylax.h"
using namespace Reylax;


static const char* gl_vertexCode =
    R"(
        #version 330 core
        layout (location = 0) in vec3 aPos; 
        void main()
        {
            gl_Position = vec4(aPos, 1.0);
        }
    )";

static const char* gl_fragCode =
    R"(
        #version 420 core
        out vec4 FragColor;
        layout (binding = 0) uniform usamplerBuffer buffer;
        uniform int _width = 1;
        void main()
        {
            const float inv255 = 1.f/255.f;
            ivec2 t     = ivec2( gl_FragCoord.xy );
            FragColor   = (texelFetch( buffer, t.y*_width+t.x ) * inv255).bgra;
        } 
    )";

struct GLTextureBufferObject
{
    GLTextureBufferObject():
        m_initialized(false),
        m_GLbuffer(0),
        m_GLtexture(0)
    {
    }

    ~GLTextureBufferObject()
    {
        if ( !m_initialized ) return;
        glDeleteBuffers(1, &m_GLbuffer);
        glDeleteTextures(1, &m_GLtexture);
    }

    bool init(u32 width, u32 height)
    {
        assert(width!=0&&height!=0);
        if (m_initialized) return false;
        glGenBuffers(1, &m_GLbuffer);
        glBindBuffer(GL_TEXTURE_BUFFER, m_GLbuffer);
        glBufferData(GL_TEXTURE_BUFFER, width*height*sizeof(GLubyte)*4, nullptr, GL_STATIC_DRAW);
        glGenTextures(1, &m_GLtexture);
        glBindBuffer(GL_TEXTURE_BUFFER, 0);
        m_initialized  = true;
        return true;
    }

    u32 id() const { return m_GLbuffer; }

private:
    bool m_initialized;
    u32 m_GLbuffer;
    u32 m_GLtexture;

    friend struct GLTextureBufferRenderer;
};

struct GLTextureBufferRenderer
{
    GLTextureBufferRenderer():
        m_initialized(false),
        m_GLvertexShader(0),
        m_GLfragmentShader(0),
        m_GLshaderProgram(0),
        m_GLquadArray(0),
        m_GLvbo(0),
        m_activeBufferObject(nullptr)
    {
    }

    ~GLTextureBufferRenderer()
    {
        if ( !m_initialized ) return;
        glDeleteShader(m_GLfragmentShader);
        glDeleteShader(m_GLvertexShader);
        glDeleteProgram(m_GLshaderProgram);
        glDeleteBuffers(1, &m_GLvbo);
        glDeleteVertexArrays(1, &m_GLquadArray);
    }

    bool init(u32 width, u32 height)
    {
        if (m_initialized) return false;

            // Setup vertex shader
        GLint vertex_compiled;
        {
            m_GLvertexShader = glCreateShader(GL_VERTEX_SHADER);
            glShaderSource(m_GLvertexShader, 1, &gl_vertexCode, nullptr);
            glCompileShader(m_GLvertexShader);
            glGetShaderiv(m_GLvertexShader, GL_COMPILE_STATUS, &vertex_compiled);
            if ( vertex_compiled != GL_TRUE )
            {
                showCompileError(true, m_GLvertexShader);
                return false;
            }
        }

        // Setup fragment shader
        GLint fragment_compiled;
        {
            m_GLfragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
            glShaderSource(m_GLfragmentShader, 1, &gl_fragCode, nullptr);
            glCompileShader(m_GLfragmentShader);
            glGetShaderiv(m_GLfragmentShader, GL_COMPILE_STATUS, &fragment_compiled);
            if ( fragment_compiled != GL_TRUE )
            {
                showCompileError(true, m_GLfragmentShader);
                return false;
            }
        }

        // Create shader program
        {
            GLint program_linked;
            m_GLshaderProgram = glCreateProgram();
            glAttachShader(m_GLshaderProgram, m_GLvertexShader);
            glAttachShader(m_GLshaderProgram, m_GLfragmentShader);
            glLinkProgram(m_GLshaderProgram);
            glGetProgramiv(m_GLshaderProgram, GL_LINK_STATUS, &program_linked);
            if ( program_linked != GL_TRUE )
            {
                showCompileError(false, m_GLshaderProgram);
                return false;
            }
            glUseProgram(m_GLshaderProgram);
            glUniform1i(glGetUniformLocation(m_GLshaderProgram, "_width"), width);
        }

        // Create full screen quad
        {
            float quadVertices[] =
            {
                -1,  1, 0,
                1,  1, 0,
                1, -1, 0,
                -1, -1, 0,
            };
            glGenVertexArrays(1, &m_GLquadArray);
            glGenBuffers(1, &m_GLvbo);
            glBindVertexArray(m_GLquadArray);

            glBindBuffer(GL_ARRAY_BUFFER, m_GLvbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
            glEnableVertexAttribArray(0);
        }

        // As little fixed-function processing as possible
        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_ALPHA_TEST);
        glDisable(GL_BLEND);

        m_initialized=true;
        return true;
    }

    void render(const GLTextureBufferObject& t)
    {
        assert(m_initialized);
        if ( !m_initialized ) return;
        if ( m_activeBufferObject != &t )
        {
            m_activeBufferObject = &t;
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_BUFFER, t.m_GLtexture);
            glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA8UI, t.m_GLbuffer);
        }
        // No clear screen or gl render state changes necessary.
        glDrawArrays(GL_QUADS, 0, 4);
    };

    void sync()
    {
        glFinish();
    }


 private:
     void showCompileError(bool shaderOrProgramError, u32 id)
     {
         GLsizei log_length = 0;
         GLchar message[1024];
         if ( shaderOrProgramError ) glGetShaderInfoLog(id, 1024, &log_length, message);
         else glGetProgramInfoLog(id, 1024, &log_length, message);
         std::cout << message << std::endl;
         assert(false);
     }

    bool m_initialized;
    u32 m_GLvertexShader;
    u32 m_GLfragmentShader;
    u32 m_GLshaderProgram;
    u32 m_GLquadArray;
    u32 m_GLvbo;
    const GLTextureBufferObject* m_activeBufferObject;
};
