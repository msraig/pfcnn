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


#ifndef __GLUT_VIEWER__
#define __GLUT_VIEWER__

#define WITH_HDR
#define WITH_ANTTWEAKBAR
#define WITH_PNG
#define WITH_GEEX

#include <assert.h>

#ifdef WITH_GEEX
#include <misc/linkage.h>
#include <GLsdk/gl_stuff.h>
#define GLUT_VIEWER_API GEEX_API
#else
#include <GL/gl.h>
#define GLUT_VIEWER_API
#endif

#ifdef WITH_HDR
#include "glut_viewer_hdr.h"
#endif

#include <malloc.h>

#ifdef __cplusplus
#include <string>
#endif

#ifdef __cplusplus
extern "C" {
#endif

    typedef void (*GlutViewerDisplayFunc)() ;
    typedef int (*GlutViewerKeyboardFunc)(char key) ;
    typedef void (*GlutViewerKeyFunc)() ;
    typedef void (*GlutViewerInitFunc)() ;

	// CJ
	typedef void(*GlutViewerSpecialKeyFunc)();

    enum GlutViewerEvent { GLUT_VIEWER_DOWN, GLUT_VIEWER_MOVE, GLUT_VIEWER_UP } ;

    typedef GLboolean (*GlutViewerMouseFunc)(float x, float y, int button, enum GlutViewerEvent event) ;

#define GLUT_VIEWER_IDLE_REDRAW 0
#define GLUT_VIEWER_DRAW_SCENE 1
#define GLUT_VIEWER_SHOW_HELP 2
#define GLUT_VIEWER_BACKGROUND 3
#define GLUT_VIEWER_HDR 4
#define GLUT_VIEWER_ROTATE_LIGHT 5   
#define GLUT_VIEWER_3D 6
#define GLUT_VIEWER_TWEAKBARS 7
   
    extern GLUT_VIEWER_API void glut_viewer_enable(int cap) ;
    extern GLUT_VIEWER_API void glut_viewer_disable(int cap) ;
    extern GLUT_VIEWER_API GLboolean  glut_viewer_is_enabled(int cap) ;
    extern GLUT_VIEWER_API GLboolean* glut_viewer_is_enabled_ptr(int cap) ;
   
    extern GLUT_VIEWER_API void glut_viewer_main_loop(int argc, char** argv) ;
    extern GLUT_VIEWER_API void glut_viewer_exit_main_loop() ;
    extern GLUT_VIEWER_API void glut_viewer_set_window_title(char* title) ;
    extern GLUT_VIEWER_API void glut_viewer_set_display_func(GlutViewerDisplayFunc f) ;
    extern GLUT_VIEWER_API void glut_viewer_set_overlay_func(GlutViewerDisplayFunc f) ;
    extern GLUT_VIEWER_API void glut_viewer_set_keyboard_func(GlutViewerKeyboardFunc f) ;
    extern GLUT_VIEWER_API void glut_viewer_set_mouse_func(GlutViewerMouseFunc f) ;
    extern GLUT_VIEWER_API void glut_viewer_set_init_func(GlutViewerInitFunc f) ;
    extern GLUT_VIEWER_API void glut_viewer_add_toggle(char key, GLboolean* pointer, const char* description) ;
    extern GLUT_VIEWER_API void glut_viewer_add_key_func(char key, GlutViewerKeyFunc f, const char* description) ;
    extern GLUT_VIEWER_API void glut_viewer_unbind_key(char key) ;
    extern GLUT_VIEWER_API void glut_viewer_set_region_of_interest(
        float xmin, float ymin, float zmin, float xmax, float ymax, float zmax
		);
	extern GLUT_VIEWER_API void glut_viewer_get_region_of_interest(
		float *xin, float *yin, float *zin, float *xax, float *yax, float *zax
		);
    extern GLUT_VIEWER_API void glut_viewer_set_screen_size(int w, int h) ;
    extern GLUT_VIEWER_API void glut_viewer_get_screen_size(int* w, int* h) ;

	// add special key
	extern GLUT_VIEWER_API void glut_viewer_add_special_key_func(char key, GlutViewerKeyFunc f);
	extern GLUT_VIEWER_API void glut_viewer_add_special_key_up_func(char key, GlutViewerKeyFunc f);

	// Get exact screen size(CJ)
	extern GLUT_VIEWER_API void glut_viewer_get_m_screen_size(int* w, int* h);

    extern GLUT_VIEWER_API void glut_viewer_clear_text() ;    
    extern GLUT_VIEWER_API void glut_viewer_printf(char *format,...) ;
    extern GLUT_VIEWER_API void glut_viewer_set_skybox(int cube_texture)  ;
    extern GLUT_VIEWER_API int  glut_viewer_get_skybox() ;
    extern GLUT_VIEWER_API void glut_viewer_set_background_color(GLfloat r, GLfloat g, GLfloat b) ;
    extern GLUT_VIEWER_API void glut_viewer_set_background_color2(GLfloat r, GLfloat g, GLfloat b) ;   
    extern GLUT_VIEWER_API GLfloat* glut_viewer_get_background_color() ;
    extern GLUT_VIEWER_API GLfloat* glut_viewer_get_background_color2() ;


    extern GLUT_VIEWER_API float* glut_viewer_get_light_matrix() ;
    extern GLUT_VIEWER_API float* glut_viewer_get_scene_quaternion() ;
    extern GLUT_VIEWER_API float* glut_viewer_get_light_quaternion() ;

    extern GLUT_VIEWER_API void glTexImage2DXPM(const char** xpm_data) ;
    extern GLUT_VIEWER_API void glTexImage2Dfile(const char* filename) ;

    extern GLUT_VIEWER_API GLboolean glut_viewer_load_image(
        const char* filename, GLuint* width, GLuint* height, GLuint* bpp, GLvoid** pixels
    ) ;

    extern GLUT_VIEWER_API int glut_viewer_fps() ;
   
    extern GLUT_VIEWER_API void glut_viewer_redraw() ;

	// get saved transformation
	extern GLUT_VIEWER_API void glut_viewer_get_saved_transform(float* rotation, float* xlat, float* scale);
	extern GLUT_VIEWER_API void glut_viewer_set_saved_transform(float* rotation, float* xlat, float scale);
	extern GLUT_VIEWER_API void glut_viewer_get_saved_scale(float* scale);
	extern GLUT_VIEWER_API void glut_viewer_get_saved_rotation(float* rotation);
	extern GLUT_VIEWER_API void glut_viewer_save_transform_for_picking() ;
    extern GLUT_VIEWER_API void glut_viewer_get_picked_ray(GLdouble* p, GLdouble* v) ;
    extern GLUT_VIEWER_API void glut_viewer_get_picked_point(GLdouble* p, GLboolean* hit_background) ;
	extern GLUT_VIEWER_API void glut_viewer_get_picked_pos2d(GLdouble* pos);

	//set matrix from other place
	extern GLUT_VIEWER_API void glut_viewer_screen_to_space(int x_in, int y_in, GLdouble* pt, GLdouble* ray);
	extern GLUT_VIEWER_API void glut_viewer_space_to_screen(GLdouble* pt, GLdouble* pt_s);


	//<<<<<
	//extern GlutViewerKeyFunc my_idle_func;
	extern GLUT_VIEWER_API void glut_viewer_set_idle_func(GlutViewerDisplayFunc f) ;

	extern GLUT_VIEWER_API void save_scene_geex();
	extern GLUT_VIEWER_API void load_scene_geex();
	//>>>>>

	// set rotation matrix
	extern GLUT_VIEWER_API void glut_viewer_set_rotation(float a[3], float theta, float m[4][4]);
	extern GLUT_VIEWER_API void glut_viewer_set_rotation_using_quaternion(float delta_rot[4], float m[4][4]);
	extern GLUT_VIEWER_API void glut_viewer_set_matrix(float a[3], float theta, float b[3], float new_quat[4], float m[4][4]);
	extern GLUT_VIEWER_API void glut_viewer_get_rotation_quat(int x0, int y0, int x1, int y1, float *cur_rot);

	// set camera position
	extern GLUT_VIEWER_API void glut_viewer_set_camera(float eye[3], float up[3]);
	extern GLUT_VIEWER_API void glut_viewer_get_saved_matrix(float* modelViewMatrix, float* projectionMatrix);
	extern GLUT_VIEWER_API void glut_viewer_get_input_file_name(char *fn);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
    inline void glut_viewer_printf(const std::string& s) { glut_viewer_printf((char*)s.c_str()) ; }
#endif

#endif

