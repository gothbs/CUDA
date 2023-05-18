#include <iostream>
#include <cmath>
#include <IL/il.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void SobelCUDA(const unsigned char *input, unsigned char *output, int width, int height, int bpp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int sobelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
        int sobelY[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

        int index = (y * width + x) * bpp;

        extern __shared__ int sharedSobelX[];
        extern __shared__ int sharedSobelY[];

        if (threadIdx.x < 3 && threadIdx.y < 3) {
            sharedSobelX[threadIdx.y * 3 + threadIdx.x] = sobelX[threadIdx.y][threadIdx.x];
            sharedSobelY[threadIdx.y * 3 + threadIdx.x] = sobelY[threadIdx.y][threadIdx.x];
        }

        __syncthreads();

        if (x >= 1 && y >= 1 && x + 1 < width && y + 1 < height) {
            for (int c = 0; c < bpp; ++c) {
                int gradX = 0;
                int gradY = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int pixel = input[((y + ky) * width + (x + kx)) * bpp + c];
                        gradX += pixel * sharedSobelX[(ky + 1) * 3 + (kx + 1)];
                        gradY += pixel * sharedSobelY[(ky + 1) * 3 + (kx + 1)];
                    }
                }
                int magnitude = sqrtf(gradX * gradX + gradY * gradY);
                output[index + c] = magnitude > 255 ? 255 : magnitude;
            }
        }
        else {
            for (int c = 0; c < bpp; ++c) {
                output[index + c] = input[index + c];
            }
        }
    }
}

void SobelGPU(unsigned char *donnees, unsigned char *nouvellesDonnees, int largeur, int hauteur, int bpp, int iterations, int blockSizeX, int blockSizeY) {
    unsigned char *donneesSrc = donnees;
    unsigned char *donneesDst = nouvellesDonnees;

    unsigned char *donneesSrcDevice;
    unsigned char *donneesDstDevice;

    size_t size = largeur * hauteur * bpp * sizeof(unsigned char);

    cudaMalloc((void**)&donneesSrcDevice, size);
    cudaMalloc((void**)&donneesDstDevice, size);

    cudaMemcpy(donneesSrcDevice, donneesSrc, size, cudaMemcpyHostToDevice);

    dim3 blockDim(blockSizeX, blockSizeY);
    dim3 gridDim((largeur + blockDim.x - 1) / blockDim.x, (hauteur + blockDim.y - 1) / blockDim.y);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    size_t sharedMemSize = 3 * 3 * sizeof(int); // Taille de la mémoire partagée nécessaire pour les matrices Sobel

    for (int i = 0; i < iterations; ++i) {
        if (i % 2 == 0) {
            SobelCUDA<<<gridDim, blockDim, sharedMemSize, stream1>>>(donneesSrcDevice, donneesDstDevice, largeur, hauteur, bpp);
        } else {
            SobelCUDA<<<gridDim, blockDim, sharedMemSize, stream2>>>(donneesSrcDevice, donneesDstDevice, largeur, hauteur, bpp);
        }

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Erreur lors de l'exécution du kernel CUDA : " << cudaGetErrorString(cudaStatus) << std::endl;
            cudaFree(donneesSrcDevice);
            cudaFree(donneesDstDevice);
            return;
        }

        std::swap(donneesSrcDevice, donneesDstDevice);

        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
    }

    cudaMemcpy(nouvellesDonnees, donneesDstDevice, size, cudaMemcpyDeviceToHost);

    cudaFree(donneesSrcDevice);
    cudaFree(donneesDstDevice);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
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

    // Appliquer l'opérateur de Sobel sur l'image avec le nombre d'itérations spécifié
    SobelGPU(donnees, nouvellesDonnees, largeur, hauteur, bpp, iterations, blockSizeX, blockSizeY);

    // Mettre à jour l'image avec les données traitées
    ilTexImage(largeur, hauteur, 1, bpp, format, IL_UNSIGNED_BYTE, nouvellesDonnees);

    // Activer l'écrasement de fichier lors de la sauvegarde
    ilEnable(IL_FILE_OVERWRITE);
    ilSaveImage(argv[5]);

    // Supprimer l'image de la mémoire de DevIL
    ilDeleteImages(1, &image);

    // Libérer la mémoire allouée pour les nouvelles données de l'image
    delete[] nouvellesDonnees;

    return 0;
}
