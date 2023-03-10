
CXX=g++
CUDACC=nvcc
HOSTLINKER=g++

PREFIX = /usr/local

CUB_INCDIR = ./dependencies/cub-1.17.0
THRUST_INCDIR = ./dependencies/thrust-1.17.0
WARPCORE_INCDIR = ./dependencies/warpcore/include
RMM_INCDIR = ./dependencies/rmm-22.06.01/include
SPDLOG_INCDIR = ./dependencies/spdlog-1.10.0/include

WARPCORE_FLAGS = -DCARE_HAS_WARPCORE -I$(WARPCORE_INCDIR)

CXXFLAGS = 

COMPILER_WARNINGS = -Wall -Wextra 
COMPILER_DISABLED_WARNING = -Wno-terminate -Wno-class-memaccess

CFLAGS_BASIC = $(COMPILER_WARNINGS) $(COMPILER_DISABLED_WARNING) -fopenmp -Iinclude -O3 -g -march=native -I$(THRUST_INCDIR)
CFLAGS_DEBUG_BASIC = $(COMPILER_WARNINGS) $(COMPILER_DISABLED_WARNING) -fopenmp -g -Iinclude -O0 -march=native -I$(THRUST_INCDIR)

CFLAGS_CPU = $(CFLAGS_BASIC) -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
CFLAGS_CPU_DEBUG = $(CFLAGS_DEBUG_BASIC) -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP

NVCCFLAGS = -x cu -lineinfo --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) -I$(CUB_INCDIR) $(WARPCORE_FLAGS) -I$(RMM_INCDIR) -I$(SPDLOG_INCDIR)
NVCCFLAGS_DEBUG = -x cu --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) -I$(CUB_INCDIR) $(WARPCORE_FLAGS) -I$(RMM_INCDIR) -I$(SPDLOG_INCDIR)

LDFLAGSGPU = -lpthread -lgomp -lstdc++fs -lnvToolsExt -lz -ldl
LDFLAGSCPU = -lpthread -lgomp -lstdc++fs -lz -ldl


SOURCES_EXTEND_CPU = \
    src/cpu_alignment.cpp \
    src/cpuminhasherconstruction.cpp \
    src/dispatch_care_extend_cpu.cpp \
    src/extensionresultoutput.cpp \
    src/main_extend_cpu.cpp \
    src/msa.cpp \
    src/options.cpp \
    src/readextension.cpp \
    src/readlibraryio.cpp \
    src/threadpool.cpp

SOURCES_EXTEND_GPU = \
    src/extensionresultoutput.cpp \
    src/options.cpp \
    src/readlibraryio.cpp \
    src/threadpool.cpp \
    src/gpu/alignmentkernels.cu \
    src/gpu/dispatch_care_extend_gpu.cu \
    src/gpu/gpuminhasherconstruction.cu \
    src/gpu/main_extend_gpu.cu \
    src/gpu/msakernels.cu \
    src/gpu/readextension_gpu.cu \
    src/gpu/sequenceconversionkernels.cu \
	src/gpu/util_gpu.cu


EXECUTABLE_EXTEND_CPU = carex-cpu
EXECUTABLE_EXTEND_GPU = carex-gpu

BUILDDIR_EXTEND_CPU = build_cpu
BUILDDIR_EXTEND_GPU = build_gpu

SOURCES_EXTEND_CPU_NODIR = $(notdir $(SOURCES_EXTEND_CPU))
SOURCES_EXTEND_GPU_NODIR = $(notdir $(SOURCES_EXTEND_GPU))

OBJECTS_EXTEND_CPU_NODIR = $(SOURCES_EXTEND_CPU_NODIR:%.cpp=%.o)
OBJECTS_EXTEND_GPU_NODIR_TMP = $(SOURCES_EXTEND_GPU_NODIR:%.cpp=%.o)
OBJECTS_EXTEND_GPU_NODIR = $(OBJECTS_EXTEND_GPU_NODIR_TMP:%.cu=%.o)

OBJECTS_EXTEND_CPU = $(OBJECTS_EXTEND_CPU_NODIR:%=$(BUILDDIR_EXTEND_CPU)/%)
OBJECTS_EXTEND_GPU = $(OBJECTS_EXTEND_GPU_NODIR:%=$(BUILDDIR_EXTEND_GPU)/%)

.PHONY: cpu gpu extendcpu extendgpu install clean
cpu: extend_cpu_release
gpu: extend_gpu_release

findgpus: findgpus.cu
	@$(CUDACC) findgpus.cu -o findgpus

.PHONY: gpuarchs.txt
gpuarchs.txt : findgpus
	$(shell ./findgpus > gpuarchs.txt) 

extend_cpu_release:
	@$(MAKE) extend_cpu_release_dummy DIR=$(BUILDDIR_EXTEND_CPU) CXXFLAGS="-std=c++17"

extend_gpu_release: gpuarchs.txt
	@$(MAKE) extend_gpu_release_dummy DIR=$(BUILDDIR_EXTEND_GPU) CXXFLAGS="-std=c++17" CUDA_ARCH="$(shell cat gpuarchs.txt)"

extend_cpu_release_dummy: $(BUILDDIR_EXTEND_CPU) $(OBJECTS_EXTEND_CPU) 
	@echo Linking $(EXECUTABLE_EXTEND_CPU)
	@$(HOSTLINKER) $(OBJECTS_EXTEND_CPU) $(LDFLAGSCPU) -o $(EXECUTABLE_EXTEND_CPU)
	@echo Linked $(EXECUTABLE_EXTEND_CPU)

extend_gpu_release_dummy: $(BUILDDIR_EXTEND_GPU) $(OBJECTS_EXTEND_GPU) 
	@echo Linking $(EXECUTABLE_EXTEND_GPU)
	@$(CUDACC) $(CUDA_ARCH) $(OBJECTS_EXTEND_GPU) $(LDFLAGSGPU) -o $(EXECUTABLE_EXTEND_GPU)
	@echo Linked $(EXECUTABLE_EXTEND_GPU)


COMPILE = @echo "Compiling $< to $@" ; $(CXX) $(CXXFLAGS) $(CFLAGS_CPU) -c $< -o $@
CUDA_COMPILE = @echo "Compiling $< to $@" ; $(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS_BASIC)" -c $< -o $@


install: 
	@echo "Installing to directory $(PREFIX)/bin"
	mkdir -p $(PREFIX)/bin
ifneq ("$(wildcard $(EXECUTABLE_EXTEND_CPU))","")
	cp $(EXECUTABLE_EXTEND_CPU) $(PREFIX)/bin/$(EXECUTABLE_EXTEND_CPU)
endif	
ifneq ("$(wildcard $(EXECUTABLE_EXTEND_GPU))","")
	cp $(EXECUTABLE_EXTEND_GPU) $(PREFIX)/bin/$(EXECUTABLE_EXTEND_GPU)
endif


clean : 
	@rm -rf build_*
	@rm -f $(EXECUTABLE_EXTEND_CPU)
	@rm -f $(EXECUTABLE_EXTEND_GPU)


$(DIR):
	mkdir $(DIR)

$(DIR)/cpu_alignment.o : src/cpu_alignment.cpp
	$(COMPILE)

$(DIR)/cpuminhasherconstruction.o : src/cpuminhasherconstruction.cpp
	$(COMPILE)

$(DIR)/dispatch_care_extend_cpu.o : src/dispatch_care_extend_cpu.cpp
	$(COMPILE)

$(DIR)/extensionresultoutput.o : src/extensionresultoutput.cpp
	$(COMPILE)

$(DIR)/main_extend_cpu.o : src/main_extend_cpu.cpp
	$(COMPILE)

$(DIR)/msa.o : src/msa.cpp
	$(COMPILE)
	
$(DIR)/options.o : src/options.cpp
	$(COMPILE)

$(DIR)/readextension.o : src/readextension.cpp
	$(COMPILE)

$(DIR)/readlibraryio.o : src/readlibraryio.cpp
	$(COMPILE)

$(DIR)/threadpool.o : src/threadpool.cpp
	$(COMPILE)

$(DIR)/alignmentkernels.o : src/gpu/alignmentkernels.cu
	$(CUDA_COMPILE)

$(DIR)/dispatch_care_extend_gpu.o : src/gpu/dispatch_care_extend_gpu.cu
	$(CUDA_COMPILE)

$(DIR)/gpuminhasherconstruction.o : src/gpu/gpuminhasherconstruction.cu
	$(CUDA_COMPILE)

$(DIR)/main_extend_gpu.o : src/gpu/main_extend_gpu.cu
	$(CUDA_COMPILE)

$(DIR)/msakernels.o : src/gpu/msakernels.cu
	$(CUDA_COMPILE)

$(DIR)/readextension_gpu.o : src/gpu/readextension_gpu.cu
	$(CUDA_COMPILE)

$(DIR)/sequenceconversionkernels.o : src/gpu/sequenceconversionkernels.cu
	$(CUDA_COMPILE)

$(DIR)/util_gpu.o : src/gpu/util_gpu.cu
	$(CUDA_COMPILE)
	


