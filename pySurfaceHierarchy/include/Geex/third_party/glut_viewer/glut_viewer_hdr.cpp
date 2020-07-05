/*
 *    _____   _       _   _   _____        _     _   _   _____   _          __  _____   _____   
 *   /  ___| | |     | | | | |_   _|      | |   / / | | | ____| | |        / / | ____| |  _  \  
 *   | |     | |     | | | |   | |        | |  / /  | | | |__   | |  __   / /  | |__   | |_| |  
 *   | |  _  | |     | | | |   | |        | | / /   | | |  __|  | | /  | / /   |  __|  |  _  /  
 *   | |_| | | |___  | |_| |   | |        | |/ /    | | | |___  | |/   |/ /    | |___  | | \ \  
 *   \_____/ |_____| \_____/   |_|        |___/     |_| |_____| |___/|___/     |_____| |_|  \_\
 *
 *  Version 1.0
 *  Bruno Levy, August 2009
 *  INRIA, Project ALICE
 * 
 */

#if defined(WIN32)
#  include <GLsdk/gl_stuff.h>
#else
#  include <GL/glx.h>
#  include <GL/glxext.h>
#endif

#include "glut_viewer.h"
#include <Geex/basics/file_system.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <GL/freeglut.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#ifndef M_PI
#define M_PI 3.141592
#endif

bool CreateShader(char *VShader, char *PShader, GLhandleARB *glContext) ;
char* loadShader(char *shaderName) ;

bool InitShaders() ;
bool InitFBO() ;
void InitBlur() ;

static float blur_width = -1.0 ;
static float blur_amount = 0.5 ;
static int do_vignette = 1 ;

void glut_viewer_set_blur_amount(float x) {
    blur_amount = x ;
}

void glut_viewer_set_vignette(int x) {
    do_vignette = x ;
}

unsigned char *generateBlurCodeFP(
    float s, bool vertical, bool tex2D, int img_width, int img_height
) ;

/*----------------------------------------------------------------------------------*/

// Note: this one is missing in glprocs.h/glprocs.c
// I wrap it in the NON_FATAL mode (i.e. does not
// exit on failure) because it seems to be missing 
// on ATI boards (and it is non-vital for the program).

#ifndef GL_ARB_color_buffer_float_XXX

#include <GLsdk/glwrap.h>

#define GL_RGBA_FLOAT_MODE_ARB            0x8820
#define GL_CLAMP_VERTEX_COLOR_ARB         0x891A
#define GL_CLAMP_FRAGMENT_COLOR_ARB       0x891B
#define GL_CLAMP_READ_COLOR_ARB           0x891C
#define GL_FIXED_ONLY_ARB                 0x891D

WRAP_GL_PROC_NON_FATAL(
    glClampColorARB, (GLenum what, GLenum value), (what, value)
) ;

#endif

/*----------------------------------------------------------------------------------*/

static float exposure = 0.5f;

// Tone shader data.
static GLhandleARB toneShader;
static unsigned int glslToneTexture;
static unsigned int glslToneBlurTexture;
static unsigned int glslToneExposure;
static unsigned int glslToneBlurAmount;
static unsigned int glslToneDoVignette;

static GLhandleARB reduce_shader ;
static unsigned int glsl_reduce_pixel_size ;
static unsigned int glsl_reduce_full_texture ;

// SkyBox shader.
static GLhandleARB skybox_shader ;
static unsigned int skybox_shader_light_map_id ;

// Blur shader
static GLuint blurh_fprog = 0 ;
static GLuint blurv_fprog = 0 ;

/** Wrapper class for frame buffer objects */
class FrameBuffer {
public:
    FrameBuffer() : frame_buffer_id(0), depth_buffer_id(0), offscreen_id(0), width(0), height(0) {
    }

    // Note: default uses GL_RGB_FLOAT16_ATI (else this fallbacks to a fixed point buffer)
    bool initialize(
        unsigned int width_in, unsigned int height_in, 
        bool with_depth_buffer, GLint internal_storage = GL_RGB_FLOAT16_ATI
    ) {

        width = width_in ;
        height = height_in ;

        // Generate frame buffer object then bind it.
        glGenFramebuffersEXT(1, &frame_buffer_id);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frame_buffer_id);

        if(with_depth_buffer) {
            glGenRenderbuffersEXT(1, &depth_buffer_id);
        }

        // Create the texture we will be using to render to.
        glGenTextures(1, &offscreen_id);
        glBindTexture(GL_TEXTURE_2D, offscreen_id);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTexImage2D(
            GL_TEXTURE_2D, 0, internal_storage, width, height, 0,
            GL_RGB, GL_FLOAT, 0
        ) ;
        
        // Bind the texture to the frame buffer.
        glFramebufferTexture2DEXT(
            GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
            GL_TEXTURE_2D, offscreen_id, 0
        ) ;

        // Initialize the depth buffer.
        if(with_depth_buffer) {
            glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depth_buffer_id);
            glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, width, height) ;
            glFramebufferRenderbufferEXT(
                GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                GL_RENDERBUFFER_EXT, depth_buffer_id
            ) ;
        }

        // Make sure we have not errors.
        if(glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT) != GL_FRAMEBUFFER_COMPLETE_EXT) {
            return false;
        }

        // Return out of the frame buffer.
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        return true;
    }

    void bind_as_texture(unsigned int unit = 0) {
        glActiveTextureARB(GL_TEXTURE0_ARB + unit) ;
        glBindTexture(GL_TEXTURE_2D, offscreen_id) ;
        glActiveTextureARB(GL_TEXTURE0_ARB) ;
    }

    void bind_as_framebuffer() {
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frame_buffer_id) ;
    }

    void unbind() {
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0) ;
        glBindTexture(GL_TEXTURE_2D, 0) ;
    }

    void blit() {
        bind_as_texture() ;
        glBegin(GL_QUADS);
           glTexCoord2f(0, 0); glVertex2f(-1, -1);
           glTexCoord2f(1, 0); glVertex2f(1, -1);
           glTexCoord2f(1, 1); glVertex2f(1, 1);
           glTexCoord2f(0, 1); glVertex2f(-1, 1);
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0) ;       
    }

    void setup_2D() {
        glMatrixMode(GL_MODELVIEW) ;
        glLoadIdentity();
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(-1, 1, -1, 1);
        glViewport(0, 0, width, height) ;
    }

    unsigned int frame_buffer_id ;
    unsigned int depth_buffer_id ;
    unsigned int offscreen_id ;
    unsigned int width ;
    unsigned int height ;
} ;

// FBO data.
static FrameBuffer the_framebuffer ;
static FrameBuffer the_framebuffer_shrink ;
static FrameBuffer the_framebuffer_shrink2 ;

static int WINDOW_WIDTH = 0 ;
static int WINDOW_HEIGHT = 0 ;

/*----------------------------------------------------------------------------------*/

void glut_viewer_hdr_set_exposure(float x) {
    exposure = x ;
}

void glut_viewer_hdr_init() {
    glut_viewer_get_screen_size(&WINDOW_WIDTH, &WINDOW_HEIGHT) ;
    if(!InitShaders()) {
        std::cerr << "could not init shaders" << std::endl ;
        exit(-1) ;
    }
    if(!InitFBO()) {
        std::cerr << "could not init FBO" << std::endl ;
        exit(-1) ;
    }
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FIXED_ONLY_ARB) ;
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FIXED_ONLY_ARB) ;
    InitBlur() ;
}

void glut_viewer_hdr_resize(int width, int height) {
    WINDOW_WIDTH = width ;
    WINDOW_HEIGHT = height ;
}

static void reduce() {
    the_framebuffer_shrink.bind_as_framebuffer() ;
    glUseProgramObjectARB(reduce_shader);    
    glUniform2fARB(glsl_reduce_pixel_size, 1.0 / float(the_framebuffer.width), 1.0 / float(the_framebuffer.height));
    glUniform1iARB(glsl_reduce_full_texture, 0);    
    the_framebuffer_shrink.setup_2D() ;
    the_framebuffer.blit() ;
    the_framebuffer_shrink.unbind() ;
    glUseProgramObjectARB(0);    
}

static void blur() {
    // Horizontal blur: framebuffer_shrink -> framebuffer_shrink2
    the_framebuffer_shrink2.bind_as_framebuffer() ;
    the_framebuffer_shrink2.setup_2D() ;
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, blurh_fprog);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    the_framebuffer_shrink.blit() ;
    the_framebuffer_shrink2.unbind() ;

    // Vertical blur: framebuffer_shrink2 -> framebuffer_shrink
    the_framebuffer_shrink.bind_as_framebuffer() ;
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, blurv_fprog);
    the_framebuffer_shrink2.blit() ;
    glDisable(GL_FRAGMENT_PROGRAM_ARB);
    the_framebuffer_shrink.unbind() ;
}

void glut_viewer_hdr_begin_frame() {
    glBindTexture(GL_TEXTURE_2D, 0);
    the_framebuffer.bind_as_framebuffer() ;
    glUseProgramObjectARB(0);
}

void glut_viewer_hdr_end_frame() {
    float aspect ;

    the_framebuffer.unbind() ;

    if(blur_amount != 0.0) {
        reduce() ;
        blur() ;
    }

    // Clear the buffers and reset the model view matrix.
    glClear(GL_COLOR_BUFFER_BIT);

    glViewport(0,0,WINDOW_WIDTH,WINDOW_HEIGHT) ;

    glMatrixMode(GL_MODELVIEW) ;
    glLoadIdentity();

    // Prepare to draw full screen quad.
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
   
    aspect = (float)(WINDOW_WIDTH) / (float)(WINDOW_HEIGHT) ;
    if(aspect < 1) {
        gluOrtho2D(-aspect, aspect, -1, 1);
    } else {
        gluOrtho2D(-1, 1, -1.0/aspect, 1.0/aspect);
    }

   // Set shader.
   glUseProgramObjectARB(toneShader);
   glUniform1iARB(glslToneTexture, 0);
   glUniform1fARB(glslToneExposure, exposure);
   glUniform1fARB(glslToneBlurAmount, blur_amount);
   glUniform1iARB(glslToneDoVignette, do_vignette) ;

   if(blur_amount != 0.0) {
       glUniform1iARB(glslToneBlurTexture, 1);
       the_framebuffer_shrink.bind_as_texture(1) ;
   } else {
       // If we do not blur, reuse the render texture as
       // the blur texture (in texture unit 0). Since the
       // final tone shaders does lerp the tone texture
       // and the blur texture, it will not change the
       // final result.
       glUniform1iARB(glslToneBlurTexture, 0);
   }

   // Copy framebuffer to screen
   the_framebuffer.blit() ;
}

/*----------------------------------------------------------------------------------*/

GLhandleARB glut_viewer_load_program(char* vp, char* fp) {
    GLhandleARB result ;
    if(!CreateShader(vp, fp, &result)) {
        std::cerr << "Could not create shader from " << vp << " and " << fp << std::endl ;
        abort() ;
    }
    return result ;
}

/*
  These functions are inspired from:
   High Dynamic Range Rendering - Using P Buffers
   Allen Sherrod
   Article for the Game Developers Magazine.
*/

bool InitShaders()
{
    std::string shader_path = ::Geex::FileSystem::get_project_root() + "/data/shaders/" ;

    std::string tone_vs = shader_path + "toneVS.glsl" ;
    std::string tone_ps = shader_path + "tonePS.glsl" ;

   // Create tone mapping shader.
    if(!CreateShader((char*)tone_vs.c_str(), (char*)tone_ps.c_str(), &toneShader)) {
       std::cerr << "........ problem with shader .........." << std::endl ;
        return false;
    }

   // Bind our shader variables.
   glslToneTexture = glGetUniformLocationARB(toneShader, "fullTexture");
   glslToneBlurTexture = glGetUniformLocationARB(toneShader, "blurTexture");
   glslToneExposure = glGetUniformLocationARB(toneShader, "exposure");
   glslToneBlurAmount = glGetUniformLocationARB(toneShader, "blur_amount");
   glslToneDoVignette = glGetUniformLocationARB(toneShader, "do_vignette") ;

   std::string skybox_vs = shader_path + "skybox_vp.glsl" ;
   std::string skybox_ps = shader_path + "skybox_fp.glsl" ;

   // Create skybox shader
   if(!CreateShader((char*)skybox_vs.c_str(), (char*)skybox_ps.c_str(), &skybox_shader)) {
       std::cerr << "........ problem with shader .........." << std::endl ;
       return false;
   }

    skybox_shader_light_map_id = glGetUniformLocationARB(skybox_shader, "light_map");

   std::string reduce_vs = shader_path + "reduce_vp.glsl" ;
   std::string reduce_ps = shader_path + "reduce_fp.glsl" ;

   // Create reduce shader
   if(!CreateShader((char*)reduce_vs.c_str(), (char*)reduce_ps.c_str(), &reduce_shader)) {
       std::cerr << "........ problem with shader .........." << std::endl ;
       return false;
   }

   glsl_reduce_pixel_size = glGetUniformLocationARB(reduce_shader, "pixel_size") ;
   glsl_reduce_full_texture =glGetUniformLocationARB(reduce_shader, "fullTexture") ;

   return true;
}

bool InitFBO() {
    bool result = true ;
    result = the_framebuffer.initialize(WINDOW_WIDTH, WINDOW_HEIGHT, true) ;
    if(!result) { std::cerr << "could not create the_framebuffer" << std::endl ; return false ; }
    result = the_framebuffer_shrink.initialize(WINDOW_WIDTH/2, WINDOW_HEIGHT/2, false) ;
    if(!result) { std::cerr << "could not create the_framebuffer_shrink" << std::endl ; return false ; }
    result = the_framebuffer_shrink2.initialize(WINDOW_WIDTH/2, WINDOW_HEIGHT/2, false) ;
    if(!result) { std::cerr << "could not create the_framebuffer_shrink2" << std::endl ; return false ; }
    return result ;
}

static void printInfoLog(GLhandleARB object) {
    int maxLength = 0;
    glGetObjectParameterivARB(object, GL_OBJECT_INFO_LOG_LENGTH_ARB, &maxLength);
    if(maxLength < 4096) {
       static char infoLog[4096] ;
       glGetInfoLogARB(object, maxLength, &maxLength, infoLog);
       std::cerr << infoLog ;
    } else {
       std::cerr << "Warning: infolog was too long to fit in buffer" << std::endl ;
    }
}


bool CreateShader(char *VShader, char *PShader, GLhandleARB *glContext) {

   printf("Loading vertex shader:%s\n", VShader) ;
   printf("Loading fragment shader:%s\n", PShader) ;   
   
   char error[4096];
   int result;
   GLhandleARB glVertexShader;
   GLhandleARB glPixelShader;

   // Create the program object.
   *glContext = glCreateProgramObjectARB();

   // Load the vertex shader.
   char *ShaderCode = loadShader(VShader);
   if(!ShaderCode) return false;
   glVertexShader = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
   glShaderSourceARB(glVertexShader, 1, (const char**)&ShaderCode, NULL);
   glCompileShaderARB(glVertexShader);
   printInfoLog(glVertexShader) ;
   glGetObjectParameterivARB(glVertexShader, GL_OBJECT_COMPILE_STATUS_ARB, &result);
   glAttachObjectARB(*glContext, glVertexShader);
   delete[] ShaderCode;
   ShaderCode = NULL;

   // Load the pixel shader.
   ShaderCode = loadShader(PShader);
   if(!ShaderCode) return false;
   glPixelShader = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
   glShaderSourceARB(glPixelShader, 1, (const char**)&ShaderCode, NULL);
   glCompileShaderARB(glPixelShader);
   printInfoLog(glPixelShader) ;
   glGetObjectParameterivARB(glPixelShader, GL_OBJECT_COMPILE_STATUS_ARB, &result);
   glAttachObjectARB(*glContext, glPixelShader);
   delete[] ShaderCode;
   ShaderCode = NULL;

   // Link our shaders.
   glLinkProgramARB(*glContext);
   glGetObjectParameterivARB(*glContext, GL_OBJECT_LINK_STATUS_ARB, &result);

   if(!result) {
       glGetInfoLogARB(*glContext, sizeof(error), NULL, error);
       fprintf(stderr, "Error linking shaders...") ;
       return false;
   }

   glDeleteObjectARB(glVertexShader);
   glDeleteObjectARB(glPixelShader);
   return true;
}


char* loadShader(char *filename) {
    //FILE* f = fopen(filename, "rt") ;
	FILE* f = fopen(filename, "rb");	// using binary mode, for different opreator for "enter"

    if(!f) {
        fprintf(stderr, "Cound not open file.\n") ;
        return NULL ;
    }

    /* 
     * An easy way of determining the length of a file:
     * Go to the end of the file, and ask where we are.
     */
    fseek(f, 0, SEEK_END) ;
    long size = ftell(f) ;

    /* Let's go back to the beginning of the file */
    fseek(f, 0, SEEK_SET) ;

    char* result = new char[size+1] ;
    fread(result, size, 1, f) ;
    result[size] = '\0' ;
    fclose(f) ;
/*
    std::cerr << "----------------- shader code -----------------------" << std::endl ;
    std::cerr << result << std::endl ;
    std::cerr << "-----------------------------------------------------" << std::endl ;
    std::cerr << "end " << filename << std::endl ;
*/
    return result ;
}

void glut_viewer_draw_skybox(int cube_texture) {

    glDepthMask(GL_FALSE) ;
    glDisable(GL_LIGHTING) ;
    glEnable(GL_TEXTURE_CUBE_MAP_ARB) ;

    glBindTexture(GL_TEXTURE_CUBE_MAP_ARB, cube_texture);
    glColor3f(1,1,1) ;

    glCullFace(GL_FRONT) ;
    glEnable(GL_CULL_FACE) ;

    glUseProgramObjectARB(skybox_shader);
    glUniform1iARB(cube_texture, 0);
    
    glutSolidCube(3) ;

    glUseProgramObjectARB(0);

    glDisable(GL_TEXTURE_CUBE_MAP_ARB) ;

    glDisable(GL_CULL_FACE) ;

    glDepthMask(GL_TRUE) ;
}

/*----------------------------------------------------------------------------------*/

/*
 * Gaussian Blur, from 'hdr' NVidia demo,
 * ported to ARBfp language.
 */

/* 1d Gaussian distribution, s is standard deviation */
float gaussian(float x, float s) {
  return exp(-x*x/(2*s*s)) / (s*sqrt(2*M_PI));
}

void append_sprintf(char** string, char *format, ...) {
    va_list args;
    va_start(args, format);
    vsprintf(*string, format, args);
    va_end(args);
    while(**string != '\0') {
        (*string)++ ;
    }
}

static char buffer[16384] ;
static char* buffer_ptr = buffer ;

unsigned char *generateBlurCodeARBfp(
    float s, bool vertical, bool tex2D, int img_width, int img_height
) {
    int n = (int) floor(3.0f*s)-1;
    float sum = 0;
    int i ;

    buffer_ptr = &(buffer[0]) ;

    for(i=-n; i<=n; i++) {
        float weight = gaussian(i, s);
        sum += weight;
    }

    append_sprintf(&buffer_ptr, "!!ARBfp1.0\n") ;
    append_sprintf(&buffer_ptr, "OPTION ARB_precision_hint_fastest;\n") ;
    append_sprintf(&buffer_ptr, "TEMP H0;\n") ;
    append_sprintf(&buffer_ptr, "TEMP H1;\n") ;
    append_sprintf(&buffer_ptr, "TEMP H2;\n") ;

    for(i=-n; i<=n; i++) {
        float x_offset, y_offset;
        if (vertical) {
            x_offset = 0; y_offset = i;
        } else {
            x_offset = i; y_offset = 0;
        }
        if (tex2D) {
            x_offset = x_offset / img_width;
            y_offset = y_offset / img_height;
        }
        float weight = gaussian(i, s) / sum;

        append_sprintf(&buffer_ptr, "ADD H0, fragment.texcoord[0], {%f, %f};\n", x_offset, y_offset) ;
        if (tex2D) {
            append_sprintf(&buffer_ptr, "TEX  H1, H0, texture[0], 2D;\n") ;
        } else {
            append_sprintf(&buffer_ptr, "TEX  H1, H0, texture[0], RECT;\n") ;
        }

        if (i==-n) {
            append_sprintf(&buffer_ptr, "MUL H2, H1, {%f}.x;\n", weight) ;
        } else {
            append_sprintf(&buffer_ptr, "MAD H2, H1, {%f}.x, H2;\n", weight) ;
        }
    }
    
    append_sprintf(&buffer_ptr, "MOV result.color.xyz, H2;\n") ;
    append_sprintf(&buffer_ptr, "END\n") ;
    
    return (unsigned char*)buffer ;
}

GLuint loadProgramARB(GLenum target, const GLubyte *code) {
    GLuint program_id ;
    glGenProgramsARB(1, &program_id) ;
    glBindProgramARB(target,program_id) ;
    glProgramStringARB(
        GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen((const char*)code), (const char*)code
    ) ;
    GLint errpos ;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &errpos) ;
    bool ok = ( errpos == -1) ;
    if(!ok) {
        const char* s = (const char*)(glGetString(GL_PROGRAM_ERROR_STRING_ARB)) ;
        std::cerr << "pixel_program_ARB error"
                  << ":" << errpos << ": " << s << std::endl ;
    }
    return program_id ;
}


//-----------------------------------------------------------------------------------------------------------------

void glut_viewer_set_blur_width(float w) {

    if(!glut_viewer_is_enabled(GLUT_VIEWER_HDR)) { return ; }

    if(blur_width == w) { return ; }
	
    blur_width = w ;
    int blur_w = the_framebuffer_shrink.width ;
    int blur_h = the_framebuffer_shrink.height ;

    // delete old programs
    if (blurh_fprog != 0) { glDeleteProgramsARB(1, &blurh_fprog); }
    if (blurv_fprog != 0) { glDeleteProgramsARB(1, &blurv_fprog); }

    unsigned char* blur_code = generateBlurCodeARBfp(blur_width, false, true, blur_w, blur_h) ;
    blurh_fprog = loadProgramARB(GL_FRAGMENT_PROGRAM_ARB, blur_code) ;
    blur_code = generateBlurCodeARBfp(blur_width, true, true, blur_w, blur_h);
    blurv_fprog = loadProgramARB(GL_FRAGMENT_PROGRAM_ARB, blur_code);
}

void InitBlur() {
    glut_viewer_set_blur_width(3.0) ;
}

/*----------------------------------------------------------------------------------*/


