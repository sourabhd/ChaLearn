# set the binaries that have to be built
TARGETS := DenseTrack Video

# set the build configuration set 
BUILD := release
#BUILD := debug

# set bin and build dirs
BUILDDIR := .build_$(BUILD)
BINDIR := $(BUILD)

CC=gcc
CXX=g++
DYLD_LIBRARY_PATH := /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib:$(DYLD_LIBRARY_PATH)
# libraries 
LDLIBS = $(addprefix -l, $(LIBS) $(LIBS_$(notdir $*)))
LIBS := \
	opencv_core opencv_highgui opencv_video opencv_imgproc \
	avformat avdevice avutil avcodec swscale

#LIBS += "/Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_calib3d.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_contrib.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_core.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_features2d.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_flann.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_gpu.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_highgui.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_imgproc.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_legacy.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_ml.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_nonfree.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_objdetect.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_ocl.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_photo.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_stitching.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_superres.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_ts.a /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_video.dylib /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib/libopencv_videostab.dylib"

# set some flags and compiler/linker specific commands
CXXFLAGS = -pipe -D __STDC_CONSTANT_MACROS -D STD=std -Wall $(CXXFLAGS_$(BUILD)) -I. -I/usr/local/include -I/Users/sourabhdaptardar/Projects/opencv-2.4.8-build/include/opencv -I/Users/sourabhdaptardar/Projects/opencv-2.4.8-build/include
CXXFLAGS_debug := -ggdb
CXXFLAGS_release := -O3 -DNDEBUG -ggdb
#LDFLAGS = -L/opt/lib -pipe -Wall $(LDFLAGS_$(BUILD))
LDFLAGS = -L/usr/local/lib -L /Users/sourabhdaptardar/Projects/opencv-2.4.8-build/lib -pipe -Wall $(LDFLAGS_$(BUILD))
LDFLAGS_debug := -ggdb
LDFLAGS_release := -O3 -ggdb

include make/generic.mk
