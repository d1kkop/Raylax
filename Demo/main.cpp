#include "main.h"
using namespace std;
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

double Time() { return static_cast<double>(duration_cast<duration<double, milli>>(high_resolution_clock::now().time_since_epoch()).count()); }

// In LoadModel.cpp
extern bool LoadModel(const std::string& name, vector<IMesh*>& meshes);

// In trace.cpp
extern void UpdateTraceData( const TraceData& td, QueueRayFptr queueRayFptr );
extern HOST_OR_DEVICE void FirstRays(u32 globalId, u32 localId);
extern HOST_OR_DEVICE void TraceCallback(u32 globalId, u32 localId, u32 depth, const HitResult& hit, const MeshData* const* meshPtrs);


struct Profiler
{
    double m_start;
    vector<pair<string, double>> m_items;

    void start() { m_start = Time(); }
    void stop(string name) { m_items.emplace_back(name, Time()-m_start); }
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
    vec3  camPos = vec3(0,0,-2.5f);
    mat3  camOrient;
    float kSpeed = 1.f;
    TraceData td;

    void update(Profiler& pr)
    {
        SDL_Event event;
        vec3 move(0);
        float speed = .3f;
        float mspeed = 0.004f;
        static bool kds[6] = {};
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
                camPitch += event.motion.yrel * -mspeed;
                break;

            case SDL_MOUSEBUTTONDOWN:
                if ( event.button.button == 1 ) kSpeed *= 2;
                if ( event.button.button == 3 ) kSpeed /= 2;
                break;

            case SDL_QUIT:
                loopDone=true;
                break;
            }
        }

        speed *= kSpeed;
        if ( kds[0] ) move.x -= speed;
        if ( kds[1] ) move.x += speed;
        if ( kds[2] ) move.z += speed;
        if ( kds[3] ) move.z -= speed;

        mat4 yaw   = glm::rotate(camPan, vec3(0.f, 1.f, 0.f));
        mat4 pitch = glm::rotate(camPitch, vec3(1.f, 0.f, 0.f));
        camOrient  = (yaw * pitch);
        camPos     += camOrient*move;
        if ( kds[4] ) camPos.y += speed;
        if ( kds[5] ) camPos.y -= speed;
    }

    void render(u32 numRays, 
                IRenderTarget* rt,
                IGpuStaticScene* scene,
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
            SyncDevice();
            pr.stop("Clear");
        }

        // Primary rays
        {
            pr.start();
            {
                td.eye    = camPos;
                td.orient = camOrient;
                td.pixels = rt->buffer<u32>();
                UpdateTraceData( td, tracer->getQueueRayAddress() );
                err = tracer->trace( numRays, scene, FirstRays, TraceCallback );
                assert(err==0);
            }
            SyncDevice();
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
vec3* CreatePrimaryRays(u32 width, u32 height, float left, float right, float top, float bottom, float zoom)
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

int main(int argc, char** argv)
{
    const char* winTitle = DEMO_NAME;
    const int width  = SCREEN_WIDTH;
    const int height = SCREEN_HEIGHT;

    SetDevice( GetNumDevices()-1 );

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
    SDL_CALL(SDL_SetRelativeMouseMode(SDL_TRUE));

    // GL interop with Cuda. We want our filled framebuffer to be blitted to the backbuffer without copy to host memory.
    GLTextureBufferObject* glRt=new GLTextureBufferObject();
    GLTextureBufferRenderer* glRenderer=new GLTextureBufferRenderer();
    if ( !glRt->init(width, height) ) return -1;
    if ( !glRenderer->init(width, height) ) return -1;

    IRenderTarget* rt{};
    IDeviceBuffer* primaryRays{};
    IGpuStaticScene* scene{};
    ITracer* tracer{};
    vector<IMesh*> meshes;

    // Create render target from a OpenGL texture buffer object
    {
        rt = IRenderTarget::createFromGLTBO(glRt->id(), width, height);
        assert(rt);
    }

    // Load test model and use as 'scene'
    {
        if ( !LoadModel(R"(D:\_Programming\2018\RaytracerCuda\Content/f16.obj)", meshes) ) return -1;
        scene = IGpuStaticScene::create(meshes.data(), (u32)meshes.size());
        assert(scene);
        for ( auto& m : meshes ) delete m;  // all data is on device, safe to delete meshes in host memory

        
        //IMesh* mesh = IMesh::create();
        //vec3* vertices = new vec3[4];
        //vec3* normals  = new vec3[4];
        //vertices[0] = vec3(-1.f, -1.f, 1.56f);
        //vertices[1] = vec3(0.f, 1.f, 1.56f);
        //vertices[2] = vec3(1.f, -1.f, 1.56f);
        //vertices[3] = vec3(2.f, 1.f, 1.56f);
        //for ( int i=0; i<4; i++ ) normals[i] = vec3(0, 0, -1);
        //u32* indices = new u32[6];
        //indices[0]=0;
        //indices[1]=1;
        //indices[2]=2;
        //indices[3]=1;
        //indices[4]=2;
        //indices[5]=3;
        //vec4* colors = new vec4[4];
        //colors[0] = vec4(1.f, 0.f, 0.f, 1.f);
        //colors[1] = vec4(0.f, 1.f, 0.f, 1.f);
        //colors[2] = vec4(0.f, 0.f, 1.f, 1.f);
        //colors[3] = vec4(1.f, 1.f, 0.f, 1.f);
        //u32 err;
        //err = mesh->setIndices(indices, 6);
        //assert(err==0);
        //err = mesh->setVertexData((float*)vertices, 4, 3, VERTEX_DATA_POSITION);
        //assert(err==0);
        //err = mesh->setVertexData((float*)colors, 4, 4, VERTEX_DATA_EXTRA4);
        //assert(err==0);
        //err = mesh->setVertexData((float*)normals, 4, 3, VERTEX_DATA_NORMAL);
        //assert(err==0);
        //delete[] colors;
        //delete[] indices;
        //delete[] vertices;
        //delete[] normals;
        //scene = IGpuStaticScene::create( &mesh, 1 );
        //assert(scene);

    }

    // Set up primary rays of a pinhole camera in local space
    {
        vec3* rays  = CreatePrimaryRays(width, height, -1, 1, 1, -1, 1);
        primaryRays = IDeviceBuffer::create( sizeof(vec3)*width*height );
        assert(primaryRays);
        primaryRays->copyFrom(rays, true); // await the transfer to complete before deletion of rays in host memory
        delete[] rays;
    }

    // Create the actual tracer
    tracer = ITracer::create();

    // Update loop
    Program p{};
    Profiler pr{};
    rt->lock();
    double tBegin = Time();
    u32 numFrames = 0;
    p.td.rayDirs  = primaryRays->ptr<vec3>();
    while ( !p.loopDone )
    {
        p.update( pr );
        p.render( width*height, rt, scene, tracer, *glRenderer, *glRt, pr );
        SDL_GL_SwapWindow(sdl_window);
        if ( Time() - tBegin > 1000.0 )
        {
            pr.showTimings( numFrames );
            tBegin = Time();
            numFrames=0;
        }
        pr.clear();
        numFrames++;
    }

    // -- Do cleanup code ---
    delete scene;
    delete primaryRays;
    delete rt;
    delete glRt;
    delete glRenderer;
    SDL_GL_DeleteContext(sdl_glContext);
    SDL_DestroyRenderer(sdl_renderer);
    SDL_DestroyWindow(sdl_window);
    SDL_Quit();

    return 0;
}