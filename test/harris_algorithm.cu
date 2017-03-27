#include <iostream>
#include <thrust/device_vector.h>
#include "../EasyBMP/EasyBMP.h"
#include <cstdint>
#include <cstddef>

__device__ float meanOfRGPU(int * j_, int * i_, std::uint8_t * picturePixels, int * width_, int * height_, float * threshold_)
{
	float a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, Ix = 0.0f, Iy = 0.0f;
	float temp;

	//---local var set
	int i = *i_;
	int j = *j_;
	int width = *width_;
	int height = *height_;
	float threshold = *threshold_;
 
	if (i > 0) {
		Ix += 2.0f * picturePixels[i-1+(j)*width];
		if (j > 0){
			Ix += picturePixels[i-1+(j-1)*width];
			Iy += picturePixels[i-1+(j-1)*width];
		}
		else{
			Ix += picturePixels[i-1+(j)*width];
			Iy += picturePixels[i-1+(j)*width];
		}
		if (j<height-1){
			Ix += picturePixels[i-1+(j+1)*width];
			Iy -= picturePixels[i-1+(j+1)*width];
		}
		else{
			Ix += picturePixels[i-1+(j)*width];
			Iy -= picturePixels[i-1+(j)*width];
		}
	}
	else{
		Ix += 2.0f * picturePixels[i+(j)*width];
		if (j > 0){
			Ix += picturePixels[i+(j-1)*width];
			Iy += picturePixels[i+(j-1)*width];
		}
		else{
			Ix += picturePixels[i+(j)*width];
			Iy += picturePixels[i+(j)*width];
		}
		if (j < height-1){
			Ix += picturePixels[i+(j+1)*width];
			Iy -= picturePixels[i+(j+1)*width];
		}
		else{
			Ix += picturePixels[i+(j)*width];
			Iy -= picturePixels[i+(j)*width];
		}
	}
 
	if (j > 0)
		Iy += 2.0f * picturePixels[i+(j-1)*width];
	else
		Iy += 2.0f * picturePixels[i+(j)*width];
	if (i < width - 1) {
		Ix -= 2.0f * picturePixels[i+1+(j)*width];
		if (j > 0) {
			Ix -= picturePixels[i+1+(j-1)*width];
			Iy += picturePixels[i+1+(j-1)*width];
		}
		else{
			Ix -= picturePixels[i+1+(j)*width];
			Iy += picturePixels[i+1+(j)*width];
		}
  		if (j < height-1){
			Ix -= picturePixels[i+1+(j+1)*width];
			Iy -= picturePixels[i+1+(j+1)*width];
		}
		else{
			Ix -= picturePixels[i+1+(j)*width];
			Iy -= picturePixels[i+1+(j)*width];
		}
	}
	else{
		Ix -= 2.0f*picturePixels[i+(j)*width];
		if ( j>0 ) {
			Ix -= picturePixels[i+(j-1)*width];
			Iy += picturePixels[i+(j-1)*width];
		}
		else{
			Ix -= picturePixels[i+(j)*width];
			Iy += picturePixels[i+(j)*width];
		}
		if (j < height - 1) {
			Ix -= picturePixels[i+(j+1)*width];
			Iy -= picturePixels[i+(j+1)*width];
		}
		else{
			Ix -= picturePixels[i+(j)*width];
			Iy -= picturePixels[i+(j)*width];
		}
	}
	if (j < height - 1)
		Iy -= 2.0f * picturePixels[i+(j+1)*width];
	else
		Iy -= 2.0f * picturePixels[i+(j)*width];
	a1 = Ix * Ix;
	a2 = Ix * Iy;
	a3 = Ix * Iy;
	a4 = Iy * Iy;
 
	temp = ((a1 + a4 - a3 - a2) + (0.04f * (a1 + a4) * (a1 + a4)));
	if (temp > threshold)
		return temp;
	return 0;
}

__global__ void fillPictMean(std::uint8_t * picturePixels, int * width_, int * height_, float * threshold_, float * pictureMeans)
{
	//--Calc thread ID & local variables
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < (*height_) && j < (*width_))
			pictureMeans[j + i * (* width_)] = meanOfRGPU(&i, &j, picturePixels, width_, height_, threshold_);
}


//Harris detector on CUDA. Calculating R, comparing with threshold, finding local maxima
__global__ void kernel(std::uint8_t * picturePixels, int * width_, int * height_, float * pictureMeans)
{
	//--Calc thread ID & local variables
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
	int width = * width_;
	int height = * height_;

	if(i < height && j < width) 
	{
		bool localMax = 1;
		for (int indexI = -1; indexI < 2 ; indexI++)
			for (int indexJ = -1; indexJ < 2 ; indexJ++)
				if ((indexI+i>=0)&&(indexJ+j>=0)&&(indexI+i<height)&&(indexJ+j<width)&&((indexI!=0)||(indexJ!=0))){
					if ((pictureMeans[i*width+j]<pictureMeans[(i+indexI)*width+j+indexJ])||(pictureMeans[i*width+j]==0))
							localMax=0;
				}

		if (localMax == 1)
			picturePixels[i*width+j] = 1;
		else
			picturePixels[i*width+j] = 0;
	}
}


//Funtcion to organize CUDA calls
void organizeCUDAcall(std::uint8_t *picturePixels, int *width, int *height, float * threshold)
{
	//Alloc GPU memory
	const int imageSize = (* width) * (* height);
	const int threadCount = 512;
	const dim3 blockSize = (*(width)) * (*(height)) / (1 * threadCount);

	std::uint8_t * picturePixelsGPU = NULL;
	float * pictureMeansG = NULL;
	int * widthGPU = NULL;
	int * heightGPU = NULL;
	float * thresholdGPU = NULL;

	cudaMalloc(&picturePixelsGPU, imageSize * sizeof(std::uint8_t));
	cudaMalloc(&widthGPU, sizeof(int));
	cudaMalloc(&heightGPU, sizeof(int));
	cudaMalloc(&thresholdGPU, sizeof(float));
	cudaMalloc(&pictureMeansG, imageSize * sizeof(float));

	//Copy data from host to device 
	cudaMemcpy(picturePixelsGPU, picturePixels, imageSize * sizeof(std::uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(widthGPU, width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(heightGPU, height, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(thresholdGPU, threshold, sizeof(float), cudaMemcpyHostToDevice);

	//Call kernel
	fillPictMean<<<blockSize, threadCount>>> (picturePixelsGPU, widthGPU, heightGPU, thresholdGPU, pictureMeansG);
	cudaDeviceSynchronize();
	kernel<<<blockSize, threadCount>>> (picturePixelsGPU, widthGPU, heightGPU, pictureMeansG);
	cudaDeviceSynchronize();

	cudaDeviceSynchronize();

	//Copy data from device to host
	cudaMemcpy(picturePixels, picturePixelsGPU, imageSize * sizeof(std::uint8_t), cudaMemcpyDeviceToHost); 

	//Free memory
	cudaFree(thresholdGPU);
	cudaFree(heightGPU);
	cudaFree(widthGPU);
	cudaFree(picturePixelsGPU);
}


//Harris detector on CPU. Calculating R, comparing with threshold
float MeanOfR (int j, int i, std::uint8_t  * picturePixels, int width, int height, float threshold)
{
	float a1=0.0f,a2=0.0f,a3=0.0f,a4=0.0f,Ix=0.0f,Iy=0.0f;
	float temp;
 
	if (i>0){
		Ix+=2.0f*picturePixels[i-1+(j)*width];
		if (j>0){
			Ix+=picturePixels[i-1+(j-1)*width];
			Iy+=picturePixels[i-1+(j-1)*width];
		}
		else{
			Ix+=picturePixels[i-1+(j)*width];
			Iy+=picturePixels[i-1+(j)*width];
		}
		if (j<height-1){
			Ix+=picturePixels[i-1+(j+1)*width];
			Iy-=picturePixels[i-1+(j+1)*width];
		}
		else{
			Ix+=picturePixels[i-1+(j)*width];
			Iy-=picturePixels[i-1+(j)*width];
		}
	}
	else{
		Ix+=2.0f*picturePixels[i+(j)*width];
		if (j>0){
			Ix+=picturePixels[i+(j-1)*width];
			Iy+=picturePixels[i+(j-1)*width];
		}
		else{
			Ix+=picturePixels[i+(j)*width];
			Iy+=picturePixels[i+(j)*width];
		}
		if (j<height-1){
			Ix+=picturePixels[i+(j+1)*width];
			Iy-=picturePixels[i+(j+1)*width];
		}
		else{
			Ix+=picturePixels[i+(j)*width];
			Iy-=picturePixels[i+(j)*width];
		}
	}
 
	if (j>0)
		Iy+=2.0f*picturePixels[i+(j-1)*width];
	else
		Iy+=2.0f*picturePixels[i+(j)*width];
	if (i<width-1){
		Ix-=2.0f*picturePixels[i+1+(j)*width];
		if (j>0){
			Ix-=picturePixels[i+1+(j-1)*width];
			Iy+=picturePixels[i+1+(j-1)*width];
		}
		else{
			Ix-=picturePixels[i+1+(j)*width];
			Iy+=picturePixels[i+1+(j)*width];
		}
  		if (j<height-1){
			Ix-=picturePixels[i+1+(j+1)*width];
			Iy-=picturePixels[i+1+(j+1)*width];
		}
		else{
			Ix-=picturePixels[i+1+(j)*width];
			Iy-=picturePixels[i+1+(j)*width];
		}
	}
	else{
		Ix-=2.0f*picturePixels[i+(j)*width];
		if (j>0){
			Ix-=picturePixels[i+(j-1)*width];
			Iy+=picturePixels[i+(j-1)*width];
		}
		else{
			Ix-=picturePixels[i+(j)*width];
			Iy+=picturePixels[i+(j)*width];
		}
		if (j<height-1){
			Ix-=picturePixels[i+(j+1)*width];
			Iy-=picturePixels[i+(j+1)*width];
		}
		else{
			Ix-=picturePixels[i+(j)*width];
			Iy-=picturePixels[i+(j)*width];
		}
	}
	if (j<height-1)
		Iy-=2.0f*picturePixels[i+(j+1)*width];
	else
		Iy-=2.0f*picturePixels[i+(j)*width];
	a1=Ix*Ix;
	a2=Ix*Iy;
	a3=Ix*Iy;
	a4=Iy*Iy;
 
	temp=((a1+a4-a3-a2)+(0.04f*(a1+a4)*(a1+a4)));
	if (temp>threshold)
		return temp;
	return 0;
}


//Harris detector on CPU. Finding local maxima
void harris (std::uint8_t  *picturePixels, int width, int height, float threshold)
{
	float * pictureMeans;
	pictureMeans = (float*)malloc(sizeof(float)*width*height);
	for (int i=0; i<height;i++)
		for (int j=0;j<width;j++)
			pictureMeans[j+i*width]=MeanOfR(i,j, picturePixels, width, height, threshold);
	for (int i=0; i<height;i++)
		for (int j=0;j<width;j++){
			bool localMax=1;
			for (int indexI=-1; indexI<2 ; indexI++)
				for (int indexJ=-1; indexJ<2 ; indexJ++)
					if ((indexI+i>=0)&&(indexJ+j>=0)&&(indexI+i<height)&&(indexJ+j<width)&&((indexI!=0)||(indexJ!=0))){
						if ((pictureMeans[i*width+j]<pictureMeans[(i+indexI)*width+j+indexJ])||(pictureMeans[i*width+j]==0))
							localMax=0;
					}
			if (localMax==1)
				picturePixels[i*width+j]=1;
			else
				picturePixels[i*width+j]=0;
		}
}


//Comparing CPU and GPU results
bool areTheResultsEqual(int height, int width, std::uint8_t  * picturePixelsGPU, std::uint8_t * picturePixelsCPU)
{
	for (int i=0;i<height;i++)
		for (int j=0;j<width;j++)
			if (picturePixelsGPU[i*width+j]!=picturePixelsCPU[i*width+j])
				return false;
	return true;
}

int main(int argc, char *argv[]) {

	//--Check args
	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " <filename>\t"<< "<threshold>" << std::endl;
		return 0;
	}

	char *fileName = argv[1];
	std::uint8_t  * picturePixelsCPU, * picturePixelsGPU;
	float threshold=atof(argv[2]);
	bool equalResults=1;
	BMP AnImage;
	
	//loading grayscale image from BMP24 format (using only red channel)
	AnImage.ReadFromFile(fileName);
	int width=AnImage.TellWidth();
	int height=AnImage.TellHeight();
	int n=width*height;
	picturePixelsCPU = (std::uint8_t *)malloc(sizeof(std::uint8_t )*n);
	picturePixelsGPU = (std::uint8_t *)malloc(sizeof(std::uint8_t )*n);
	for (int i=0;i<height;i++)
		for (int j=0;j<width;j++){
			picturePixelsCPU[i*width+j]=AnImage.GetPixel(j,i).Red;
		}

	memcpy(&picturePixelsGPU[0],&picturePixelsCPU[0],n*sizeof(std::uint8_t ));
	
	//TODO measure time using CUDA events

	//CPU call
	//harris(picturePixelsCPU, width, height, threshold);

	//Saving the resulting CPU image
	RGBApixel redDot;
	redDot.Red=255;
	redDot.Blue=0;
	redDot.Green=0;
	redDot.Alpha=0;

	for (int i=0;i<height;i++)
		for (int j=0;j<width;j++)
			if (picturePixelsCPU[i*width+j]==1)
				for (int indexI=-0;indexI<1;indexI++)
					for (int indexJ=-0;indexJ<1;indexJ++)
						if ((indexI+i>=0)&&(indexJ+j>=0)&&(indexI+i<height)&&(indexJ+j<width))
							AnImage.SetPixel(j, i, redDot);
	AnImage.WriteToFile("out.bmp");


	//GPU call
	organizeCUDAcall(&picturePixelsGPU[0], &width, &height, &threshold);
	
	//--Save GPU-generated image
	for (int i=0;i<height;i++)
		for (int j=0;j<width;j++)
			if (picturePixelsGPU[i*width+j]==1)
				for (int indexI=-0;indexI<1;indexI++)
					for (int indexJ=-0;indexJ<1;indexJ++)
						if ((indexI+i>=0)&&(indexJ+j>=0)&&(indexI+i<height)&&(indexJ+j<width))
							AnImage.SetPixel(j,i,redDot);
	AnImage.WriteToFile("out_gpu.bmp");



	//checking the results 
	if (!areTheResultsEqual(height, width, picturePixelsGPU, picturePixelsCPU))
		equalResults = 0; 

	//TODO print out CPU and GPU time
	return 0;
}
