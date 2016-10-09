# -- FreeGLUT 3 --
set(FREEGLUT_BUILD_SHARED_LIBS OFF)
add_subdirectory(freeglut3)
set(FreeGLUT3_INCLUDE_DIRS "deps/freeglut3/include" PARENT_SCOPE)

# -- GLEW 2 --
add_subdirectory("glew2")
