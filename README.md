# Unified Memory & CUDA Stream Optimization for Histogram by Manish Chugani
  
Steps to execute the code:
  
1: Clone the repository into a local system that has a GPU Device.  
2: Run the make command.  
3: Run the "./histogram" command with command line arguments for number of elements and number of bins for the histogram.  
4: The code implements Unified Memory using CUDAHostAlloc, CUDAMallocManaged and CUDA Streams using CUDAMemCpyAsync function calls for histograms. The tests pass showing that the project is complete in its functionality.
