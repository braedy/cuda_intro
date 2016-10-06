#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  //create a mapping from the 2D block and grid locations
  //to an absolute 2D location in the image, then use that to
  //calculate a 1D offset
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;  

  //convert from color to greyscale (ignore alpha)
  //mapping from components of a uchar4 to RGBA are:
  //.x -> R ; .y -> G ; .z -> B ; .w -> A;
  
  uchar4 rgba = rgbaImage[thread_1D_pos];
  //The output (greyImage) at each pixel should be the result of
  //applying the formula: output = .299f * R + .587f * G + .114f * B;
  float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
  greyImage[thread_1D_pos] = channelSum;
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage,
                            uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage,
                            size_t numRows,
                            size_t numCols)
{
  const dim3 blockSize(16, 16);
  const dim3 gridSize((numCols + blockSize.x - 1)/blockSize.x, 
                      (numRows + blockSize.y - 1)/blockSize.y);
  //launch kernel
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage,
                                             d_greyImage,
                                             numRows,
                                             numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}