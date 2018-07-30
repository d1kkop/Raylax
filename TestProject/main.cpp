#include "Reylax.h"
#include "GLinterop.h"
#include <SDL.h>
#include <iostream>
#include <chrono>
#include "glm/common.hpp"
#include "glm/geometric.hpp"
#include "glm/vec3.hpp"
#include "glm/mat3x3.hpp"
#include "glm/mat4x4.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/transform.hpp"
#include "glm/gtx/rotate_vector.hpp"
using namespace std;
using namespace glm;
using namespace Reylax;
using namespace chrono;

#define LIB_CALL( expr, name ) \
{ \
	auto status=expr; \
	if ( status != 0 )  \
	{ \
		cout << name << " ERROR: " << (status) << endl; \
		assert( 0 ); \
		exit( 1 ); \
	} \
}

#define SDL_CALL( expr ) LIB_CALL( expr, "SDL" )

double time() { return static_cast<double>(duration_cast<duration<double, milli>>(high_resolution_clock::now().time_since_epoch()).count()); }


struct Profiler
{
    double m_start;
    vector<pair<string,double>> m_items;

    void start() { m_start = time(); }
    void stop(string name) { m_items.emplace_back(name, time()-m_start); }
    void showTimings(u32 numFrames)
    {
        printf("---- Timings ---- \n");
        for ( auto& i : m_items ) printf("%s:\t\t%.3fms\n", i.first.c_str(), i.second);
        printf("Fps:  \t\t%d\n\n", numFrames);
    }
    void clear() { m_items.clear(); }
};

struct Program
{
    bool m_done;
    float m_pan;
    float m_pitch;
    vec3 m_pos;


    void update(Profiler& pr)
    {
        SDL_Event event;
        vec3 move(0);
        float speed = .3f;
        float mspeed = 0.004f;
        static bool kds[6] ={ false, false, false, false };
        while ( SDL_PollEvent(&event) )
        {
            switch ( event.type )
            {
            case SDL_KEYDOWN:
                if ( event.key.keysym.sym == SDLK_ESCAPE ) return;
                if ( event.key.keysym.sym == SDLK_a ) kds[0]=true;
                if ( event.key.keysym.sym == SDLK_d ) kds[1]=true;
                if ( event.key.keysym.sym == SDLK_w ) kds[2]=true;
                if ( event.key.keysym.sym == SDLK_s ) kds[3]=true;
                if ( event.key.keysym.sym == SDLK_q ) kds[4]=true;
                if ( event.key.keysym.sym == SDLK_e ) kds[5]=true;
                break;

            case SDL_KEYUP:
                if ( event.key.keysym.sym == SDLK_a ) kds[0]=false;
                if ( event.key.keysym.sym == SDLK_d ) kds[1]=false;
                if ( event.key.keysym.sym == SDLK_w ) kds[2]=false;
                if ( event.key.keysym.sym == SDLK_s ) kds[3]=false;
                if ( event.key.keysym.sym == SDLK_q ) kds[4]=false;
                if ( event.key.keysym.sym == SDLK_e ) kds[5]=false;
                break;

            case SDL_MOUSEMOTION:
                m_pan += event.motion.xrel * mspeed;
                m_pitch += event.motion.yrel * mspeed;
                break;

            case SDL_QUIT:
                m_done=true;
                break;
            }
        }

        if ( kds[0] ) move.x -= speed;
        if ( kds[1] ) move.x += speed;
        if ( kds[2] ) move.z += speed;
        if ( kds[3] ) move.z -= speed;

        mat4 yaw   = glm::rotate(m_pan, vec3(0.f, 1.f, 0.f));
        mat4 pitch = glm::rotate(m_pitch, vec3(1.f, 0.f, 0.f));
        mat3 orient = (yaw * pitch);
        m_pos += orient*move;
        if ( kds[4] ) m_pos.y += speed;
        if ( kds[5] ) m_pos.y -= speed;
    }

    void render(IRenderTarget* rt, 
                GLTextureBufferRenderer& glRenderer,
                GLTextureBufferObject& glTbo,
                Profiler& pr)
    {
        u32 err=0;

        pr.start();
        err = rt->unlock();
        assert(err==0);
        pr.stop("Unlock");

        pr.start();
        err = rt->lock();
        assert(err==0);
        pr.stop("Lock");

        pr.start();
        for ( int i = 0; i<320; i++ )
        {
            err = rt->clear((i*3) | (50<<16));
            assert(err==0);
        }
        rlSyncDevice();
        pr.stop("Clear");

        // Primary rays
        pr.start();
        {
            mat4 yaw   = rotate(m_pan, vec3(0.f, 1.f, 0.f));
            mat4 pitch = rotate(m_pitch, vec3(1.f, 0.f, 0.f));
            mat3 orient = ( yaw * pitch );
  //          err = m_camera->traceScene(&m_pos.x, &orient[0][0], m_scene);
            assert(err==0);
        } 
      //  cudaDeviceSynchronize();
        pr.stop("Trace");



        // Draw fulls creen quad and copy buffer to gl render target.
        pr.start();
        //glRenderer.sync();
        glRenderer.render( glTbo );
        pr.stop("Blit");
    }
};


int main(int argc, char** argv)
{
    const char* winTitle = "ReylaxTest";
    int width  = 1920;
    int height = 1080;

    rlSetDevice( rlGetNumDevices()-1 );

    GLTextureBufferObject* glRt=new GLTextureBufferObject();
    GLTextureBufferRenderer* glRenderer=new GLTextureBufferRenderer();
    IRenderTarget* rt=nullptr;
    SDL_Window*   sdl_window=nullptr;
    SDL_Renderer* sdl_renderer;
    SDL_GLContext sdl_glContext;

    // Create sdl render window (with openGL)
    SDL_CALL(SDL_Init(SDL_INIT_VIDEO));
    u32 flags     = SDL_WINDOW_OPENGL;// | SDL_WINDOW_FULLSCREEN;
    sdl_window    = SDL_CreateWindow(winTitle, 100, 100, width, height, flags);
    sdl_renderer  = SDL_CreateRenderer(sdl_window, -1, SDL_RENDERER_ACCELERATED);
    sdl_glContext = SDL_GL_CreateContext(sdl_window);
    SDL_CALL(SDL_GL_MakeCurrent(sdl_window, sdl_glContext));

    // OpenGL interop with Reylax
    if ( !glRt->init(width, height) ) return -1;
    if ( !glRenderer->init(width, height) ) return -1;
    rt = IRenderTarget::createFromGLTBO(glRt->id(), width, height);
    if ( !rt ) return -1;

    // Update loop
    Program p;
    Profiler pr;
    memset( &p, 0, sizeof(p) );
    rt->lock();
    double tBegin = time();
    u32 numFrames = 0;
    while ( !p.m_done )
    {
        p.update( pr );
        p.render( rt, *glRenderer, *glRt, pr );
        SDL_GL_SwapWindow(sdl_window);
        if ( time() - tBegin > 1000.0 )
        {
            pr.showTimings( numFrames );
            tBegin = time();
            numFrames=0;
        }
        pr.clear();
        numFrames++;
    }

    // -- Do cleanup code ---
    delete rt;
    delete glRt;
    delete glRenderer;
    SDL_GL_DeleteContext(sdl_glContext);
    SDL_DestroyRenderer(sdl_renderer);
    SDL_DestroyWindow(sdl_window);
    SDL_Quit();

    return 0;
}