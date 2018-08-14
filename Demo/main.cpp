#include "Reylax.h"
#include "GLinterop.h"
// SDL
#include <SDL.h>
// STL
#include <cstdio>
#include <cassert>
#include <iostream>
#include <chrono>
// CDUA
#include <cuda_runtime.h>
// GLM
#include "glm/common.hpp"
#include "glm/geometric.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
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
bool loadModel(const std::string& name, vector<IMesh*>& meshes);

extern __device__ u32* buffer;
extern __device__ void TraceCallback(u32 globalId, u32 localId, const HitResult& hit, const MeshData* const* meshPtrs, float* rayOris, float* rayDirs);


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
    bool  loopDone;
    float camPan;
    float camPitch;
    vec3  camPos;

    Program()
    {
        loopDone=false;
        camPan=0;
        camPitch=0;
        camPos = vec3(0,0,-2.5f);
    }

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
                if ( event.key.keysym.sym == SDLK_ESCAPE ) loopDone=true;
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
                camPan += event.motion.xrel * mspeed;
                camPitch += event.motion.yrel * mspeed;
                break;

            case SDL_QUIT:
                loopDone=true;
                break;
            }
        }

        if ( kds[0] ) move.x -= speed;
        if ( kds[1] ) move.x += speed;
        if ( kds[2] ) move.z += speed;
        if ( kds[3] ) move.z -= speed;

        mat4 yaw   = glm::rotate(camPan, vec3(0.f, 1.f, 0.f));
        mat4 pitch = glm::rotate(camPitch, vec3(1.f, 0.f, 0.f));
        mat3 orient = (yaw * pitch);
   //     camPos += orient*move;
        if ( kds[4] ) camPos.y += speed;
        if ( kds[5] ) camPos.y -= speed;
    }

    void render(IRenderTarget* rt, 
                IGpuStaticScene* scene,
                ITraceQuery* query,
                ITraceResult* result,
                ITracer* tracer,
                GLTextureBufferRenderer& glRenderer,
                GLTextureBufferObject& glTbo,
                Profiler& pr)
    {
        u32 err=0;

        // Unlock
        {
            pr.start();
            err = rt->unlock();
            assert(err==0);
            pr.stop("Unlock");
        }

        // Lock
        {
            pr.start();
            err = rt->lock();
            assert(err==0);
            pr.stop("Lock");
        }

        // Clear
        {
            pr.start();
            for ( int i = 0; i<1; i++ )
            {
                err = rt->clear(255);
                assert(err==0);
            }
            syncDevice();
            pr.stop("Clear");
        }

        // Primary rays
        {
            pr.start();
            {
                mat4 yaw   = rotate(camPan, vec3(0.f, 1.f, 0.f));
                mat4 pitch = rotate(camPitch, vec3(1.f, 0.f, 0.f));
          //      mat3 orient = (yaw * pitch);
                mat3 orient(1);
                err = tracer->trace((const float*)&camPos, (const float*)&orient, scene, query, &result, 1, TraceCallback);
                assert(err==0);
            }
            syncDevice();
            pr.stop("Trace");
        }

        // Draw fulls creen quad and copy buffer to gl render target.
        pr.start();
        //glRenderer.sync();
        glRenderer.render( glTbo );
        pr.stop("Blit");
    }
};


// Setup camera directions in local space
vec3* createPrimaryRays(u32 width, u32 height, float left, float right, float top, float bottom, float zoom)
{
    assert(width*height!=0);
    float dx = (right-left) / width;
    float dy = (bottom-top) / height;
    float ry = top + dy*.5f;
    float z2 = zoom*zoom;
    vec3* dirs = new vec3[width*height];
    for ( u32 y=0; y<height; y++, ry += dy )
    {
        float rx = left + dx*.5f;
        for ( u32 x=0; x<width; x++, rx += dx )
        {
            float d = 1.f / sqrt(z2 + rx*rx + ry*ry);
            assert(!(::isnan(d) || d <= 0.f));
            auto addr = y*width+x;
            dirs[addr].x = rx*d;
            dirs[addr].y = ry*d;
            dirs[addr].z = zoom*d;
        }
    }
    return dirs;
}


template <class T> T _min(T a, T b) { return a<b?a:b; }
template <class T> T _max(T a, T b) { return a>b?a:b; }
template <> vec3 _min(vec3 a, vec3 b) { return vec3(_min<float>(a.x, b.x), _min<float>(a.y, b.y), _min<float>(a.z, b.z)); }
template <> vec3 _max(vec3 a, vec3 b) { return vec3(_max<float>(a.x, b.x), _max<float>(a.y, b.y), _max<float>(a.z, b.z)); }

float temp_BoxRayIntersect(const vec3& bMin, const vec3& bMax, const vec3& orig, const vec3& invDir)
{
    vec3 tMin  = (bMin - orig) * invDir;
    vec3 tMax  = (bMax - orig) * invDir;
    vec3 oMin  = _min(tMin, tMax);
    vec3 oMax  = _max(tMin, tMax);
    float dmin = _max(oMin.x, _max(oMin.y, oMin.z));
    float dmax = _min(oMax.x, _min(oMax.y, oMax.z));
    float dist = _max(0.f, dmin);
    return (dmax >= dmin ? dist : FLT_MAX);
}

int main(int argc, char** argv)
{
    vec3 bMin(-1);
    vec3 bMax(1);
    vec3 o(1.001f,0.2f,1.0f-0.01f);
    vec3 d(-1.001f,0,1);
    d = normalize(d);
    vec3 invd(1.f/d.x, 1.f/d.y, 1.f/d.z);

    float kDist = temp_BoxRayIntersect( bMin, bMax, o, invd );

    const char* winTitle = "ReylaxTest";
    int width  = 1920;
    int height = 1080;

    setDevice( getNumDevices()-1 );

    // Window management (SDL)
    SDL_Window*   sdl_window=nullptr;
    SDL_Renderer* sdl_renderer;
    SDL_GLContext sdl_glContext;
    SDL_CALL(SDL_Init(SDL_INIT_VIDEO));
    u32 flags     = SDL_WINDOW_OPENGL;// | SDL_WINDOW_FULLSCREEN;
    sdl_window    = SDL_CreateWindow(winTitle, 100, 100, width, height, flags);
    sdl_renderer  = SDL_CreateRenderer(sdl_window, -1, SDL_RENDERER_ACCELERATED);
    sdl_glContext = SDL_GL_CreateContext(sdl_window);
    SDL_CALL(SDL_GL_MakeCurrent(sdl_window, sdl_glContext));

    // GL interop with Cuda. We want our filled framebuffer to be blitted to the backbuffer without copy to host memory.
    GLTextureBufferObject* glRt=new GLTextureBufferObject();
    GLTextureBufferRenderer* glRenderer=new GLTextureBufferRenderer();
    if ( !glRt->init(width, height) ) return -1;
    if ( !glRenderer->init(width, height) ) return -1;

    IRenderTarget* rt;
    IGpuStaticScene* scene=nullptr;
    ITraceQuery* query=nullptr;
    ITraceResult* result=nullptr;
    ITracer* tracer;
    vector<IMesh*> meshes;

    // Create render target from a OpenGL texture buffer object
    rt = IRenderTarget::createFromGLTBO(glRt->id(), width, height);
    assert(rt);

    // Load test model
    if ( !loadModel(R"(D:\_Programming\2018\RaytracerCuda\Content/f16.obj)", meshes) )
    {
        cout << "Failed to load f16" << endl;
    }
 //   scene = IGpuStaticScene::create(meshes.data(), (u32)meshes.size());
 //   assert(scene);
    for ( auto& m : meshes ) delete m;

    // All primary rays only have a unique direction, set this up.
    vec3* rays = createPrimaryRays(width, height, -1, 1, 1, -1, 1);
    query = ITraceQuery::create((float*)rays, width*height);
    assert(query);
    delete[] rays;

    // Each trace has a trace result
    result = ITraceResult::create(width*height);
    assert(result);

    // Create the actual tracer
    tracer = ITracer::create();

    // Update loop
    Program p;
    Profiler pr;
    rt->lock();
    double tBegin = time();
    u32 numFrames = 0;
    while ( !p.loopDone )
    {
        p.update( pr );
        p.render( rt, scene, query, result, tracer, *glRenderer, *glRt, pr );
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
    delete result;
    delete query;
    delete scene;
    delete rt;
    delete glRt;
    delete glRenderer;
    SDL_GL_DeleteContext(sdl_glContext);
    SDL_DestroyRenderer(sdl_renderer);
    SDL_DestroyWindow(sdl_window);
    SDL_Quit();

    return 0;
}