#include <iostream>
#include <cmath>
#include <IL/il.h>
#include <cuda.h>
#include <cuda_runtime.h>

void normaliserNoyauGaussien(float noyauGaussien[5][5]) {
    float somme = 0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            somme += noyauGaussien[i][j];
        }
    }
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            noyauGaussien[i][j] /= somme;
        }
    }
}

__global__ void applyGaussianBlur(const unsigned char *donneesSrc, unsigned char *donneesDst, int largeur, int hauteur, int bpp, const float *noyauGaussien) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 2 || y < 2 || x + 2 >= largeur || y + 2 >= hauteur) {
        for (int c = 0; c < bpp; ++c) {
            donneesDst[(y * largeur + x) * bpp + c] = donneesSrc[(y * largeur + x) * bpp + c];
        }
        return;
    }

    extern __shared__ float sharedNoyauGaussien[];
    if (threadIdx.x < 5 && threadIdx.y < 5) {
        sharedNoyauGaussien[threadIdx.y * 5 + threadIdx.x] = noyauGaussien[threadIdx.y * 5 + threadIdx.x];
    }
    __syncthreads();

    for (int c = 0; c < bpp; ++c) {
        float somme = 0;
        for (int ky = -2; ky <= 2; ++ky) {
            for (int kx = -2; kx <= 2; ++kx) {
                somme += donneesSrc[((y + ky) * largeur + (x + kx)) * bpp + c] * sharedNoyauGaussien[(ky + 2) * 5 + (kx + 2)];
            }
        }
        donneesDst[(y * largeur + x) * bpp + c] = somme;
    }
}

void flouGaussienGPU(unsigned char *donnees, unsigned char *nouvellesDonnees, int largeur, int hauteur, int bpp, int iterations,int blockSizeX, int blockSizeY) {
    float noyauGaussien[5][5] = {
        {1.0f, 4.0f,  7.0f,  4.0f, 1.0f},
        {4.0f, 16.0f, 26.0f, 16.0f, 4.0f},
        {7.0f, 26.0f, 41.0f, 26.0f, 7.0f},
        {4.0f, 16.0f, 26.0f, 16.0f, 4.0f},
        {1.0f, 4.0f,  7.0f,  4.0f, 1.0f}
    };

    normaliserNoyauGaussien(noyauGaussien);

    float *devNoyauGaussien;
    cudaMalloc((void **)&devNoyauGaussien, 5 * 5 * sizeof(float));
    cudaMemcpy(devNoyauGaussien, noyauGaussien, 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);

    unsigned char *devDonneesSrc, *devDonneesDst;
    cudaMalloc((void **)&devDonneesSrc, largeur * hauteur * bpp * sizeof(unsigned char));
    cudaMalloc((void **)&devDonneesDst, largeur * hauteur * bpp * sizeof(unsigned char));

    cudaMemcpy(devDonneesSrc, donnees, largeur * hauteur * bpp * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(blockSizeX, blockSizeY); 
    dim3 gridDim((largeur + blockDim.x - 1) / blockDim.x, (hauteur + blockDim.y - 1) / blockDim.y);

    size_t sharedMemSize = 5 * 5 * sizeof(float);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int it = 0; it < iterations; ++it) {
        if (it < iterations - 1) {
            applyGaussianBlur<<<gridDim, blockDim, sharedMemSize, stream>>>(devDonneesSrc, devDonneesDst, largeur, hauteur, bpp, devNoyauGaussien);
            cudaStreamSynchronize(stream);
            std::swap(devDonneesSrc, devDonneesDst);
        }
        else {
            applyGaussianBlur<<<gridDim, blockDim, sharedMemSize, stream>>>(devDonneesSrc, devDonneesDst, largeur, hauteur, bpp, devNoyauGaussien);
            cudaStreamSynchronize(stream);
        }
    }

    cudaMemcpy(nouvellesDonnees, devDonneesDst, largeur * hauteur * bpp * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(devDonneesSrc);
    cudaFree(devDonneesDst);
    cudaFree(devNoyauGaussien);
    cudaStreamDestroy(stream);
}

int main(int argc, char *argv[]) {
    unsigned int image;

    ilInit();
    ilGenImages(1, &image);
    ilBindImage(image);
    ilLoadImage(argv[1]);

    int largeur, hauteur, bpp, format;

    largeur = ilGetInteger(IL_IMAGE_WIDTH);
    hauteur = ilGetInteger(IL_IMAGE_HEIGHT);
    bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
    format = ilGetInteger(IL_IMAGE_FORMAT);

    unsigned char *donnees = ilGetData();
    unsigned char *nouvellesDonnees = new unsigned char[largeur * hauteur * bpp];

    int iterations = std::stoi(argv[2]);
    int blockSizeX = std::stoi(argv[3]);
    int blockSizeY = std::stoi(argv[4]);

    flouGaussienGPU(donnees, nouvellesDonnees, largeur, hauteur, bpp, iterations, blockSizeX, blockSizeY);

    ilTexImage(largeur, hauteur, 1, bpp, format, IL_UNSIGNED_BYTE, nouvellesDonnees);

    ilEnable(IL_FILE_OVERWRITE);
    ilSaveImage(argv[5]);

    ilDeleteImages(1, &image);
    delete[] nouvellesDonnees;

    return 0;
}
