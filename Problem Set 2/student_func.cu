#include "reference_calc.cpp"
#include "utils.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  // compute any intermediate results in floating point
  float result = 0.f;
  // get positions
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  // handle OOBs
  if((thread_2D_pos.x < numCols) && (thread_2D_pos.y < numRows))
    {
        // square filter
        const int filter_offset = filterWidth/2;
        // sum the window pixels
        for(int i = -filter_offset; i <= filter_offset; i++)
        {
            for(int j = -filter_offset; j <= filter_offset; j++)
            {
                // clamp for out of range access
                int clamped_i = min(max(thread_2D_pos.y + i, 0), static_cast<int>(numRows - 1));
                int clamped_j = min(max(thread_2D_pos.x + j, 0), static_cast<int>(numCols - 1));
                //clamp(i, 0, numRows-1); clamp(j, 0, numCols-1); //Cg
                // result += tex2D(tex8u,thread_2D_pos.y + i, thread_2D_pos.x + j); // tex2D option
                float image_value = static_cast<float>(inputChannel[clamped_i * numCols + clamped_j]);
                float filter_value = filter[(i + filter_offset) * filterWidth + j + filter_offset];
                result += image_value * filter_value;     
            }
        }
    // store final result as unsigned char
    outputChannel[thread_1D_pos] = static_cast<unsigned char>(result);
    }
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{ 
  //position
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
  //OOB
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;
  
  redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
  greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
  blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);
  //store in output image
  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //allocate memory for the filter on the GPU
  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));
  
  //Copy the filter on the host (h_filter) to the memory just allocated
  checkCudaErrors(cudaMemcpy(d_filter,
               h_filter,
               sizeof(float) * filterWidth * filterWidth,
               cudaMemcpyHostToDevice
               )
         );
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  const dim3 blockSize(16, 16);

  //Compute grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize((numCols + blockSize.x - 1)/blockSize.x,
                      (numRows + blockSize.y - 1)/blockSize.y);

  //TODO: Launch a kernel for separating the RGBA image into different color channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                      numRows,
                      numCols,
                      d_red,
                      d_green,
                      d_blue);
  
  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //call convolution kernel here 3 times, once for each color channel.
  gaussian_blur<<<gridSize, blockSize>>>(d_red,
                     d_redBlurred,
                     numRows,
                     numCols,
                     d_filter,
                     filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_green,
                     d_greenBlurred,
                     numRows,
                     numCols,
                     d_filter,
                     filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_blue,
                     d_blueBlurred,
                     numRows,
                     numCols,
                     d_filter,
                     filterWidth);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now recombine results. We take care of launching this kernel for you.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

//Free all the memory allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  
  checkCudaErrors(cudaFree(d_filter));
}