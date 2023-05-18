#include <iostream>
#include <cmath>
#include <IL/il.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void LaplacianOfGaussianCUDA(const unsigned char *input, unsigned char *output, int width, int height, int bpp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int masque[5][5] = {
            {0, 0, -1, 0, 0},
            {0, -1, -2, -1, 0},
            {-1, -2, 16, -2, -1},
            {0, -1, -2, -1, 0},
            {0, 0, -1, 0, 0}
        };

        int index = (y * width + x) * bpp;

        extern __shared__ int sharedMasque[];  // Mémoire partagée pour le masque
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                sharedMasque[i * 5 + j] = masque[i][j];
            }
        }

        __syncthreads();

        if (x >= 2 && y >= 2 && x + 2 < width && y + 2 < height) {
            for (int c = 0; c < bpp; ++c) {
                int somme = 0;
                for (int fy = -2; fy <= 2; ++fy) {
                    for (int fx = -2; fx <= 2; ++fx) {
                        somme += sharedMasque[(fy + 2) * 5 + (fx + 2)] * input[((y + fy) * width + (x + fx)) * bpp + c];
                    }
                }

                output[index + c] = fminf(fmaxf(somme, 0), 255);
            }
        }
        else {
            for (int c = 0; c < bpp; ++c) {
                output[index + c] = input[index + c];
            }
        }
    }
}

void LaplacienDeGaussienne(unsigned char *donnees, unsigned char *nouvellesDonnees, int largeur, int hauteur, int bpp, int iterations) {
  
    
    unsigned char *donneesDst = nouvellesDonnees;
    unsigned char *donneesSrc = donnees;

    unsigned char *donneesSrcDevice;
    unsigned char *donneesDstDevice;

    cudaError_t cudaStatus;

    size_t size = largeur * hauteur * bpp * sizeof(unsigned char);

    cudaStatus = cudaMalloc((void **)&donneesSrcDevice, size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erreur lors de l'allocation de mémoire sur le GPU pour les données source" << std::endl;
        return;
    }

    cudaStatus = cudaMalloc((void **)&donneesDstDevice, size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erreur lors de l'allocation de mémoire sur le GPU pour les données de destination" << std::endl;
        cudaFree(donneesDstDevice);
        return;
    }

    cudaStatus = cudaMemcpy(donneesSrcDevice, donneesSrc, size, cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erreur lors de la copie des données source de l'hôte vers le GPU" << std::endl;
        cudaFree(donneesDstDevice);
        cudaFree(donneesSrc);
        return;
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((largeur + blockDim.x - 1) / blockDim.x, (hauteur + blockDim.y - 1) / blockDim.y);

    cudaStream_t stream;
    cudaStatus = cudaStreamCreate(&stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erreur lors de la création du stream CUDA" << std::endl;
        cudaFree(donneesDstDevice);
        cudaFree(donneesSrc);
        return;
    }

    size_t sharedMemSize = 5 * 5 * sizeof(int);  // Taille de la mémoire partagée nécessaire pour le masque

    for (int i = 0; i < iterations; ++i) {
        LaplacianOfGaussianCUDA<<<gridDim, blockDim, sharedMemSize, stream>>>(donneesSrcDevice, donneesDstDevice, largeur, hauteur, bpp);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Erreur lors de l'exécution du kernel CUDA : " << cudaGetErrorString(cudaStatus) << std::endl;
            cudaFree(donneesDstDevice);
            cudaFree(donneesSrc);
            cudaStreamDestroy(stream);
            return;
        }

        std::swap(donneesSrcDevice, donneesDstDevice);
    }
    
    cudaMemcpyAsync(donneesDst, donneesSrcDevice, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    //cudaMemcpy(nouvellesDonnees, devNouvellesDonnees, largeur * hauteur * bpp * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erreur lors de la copie des données de destination du GPU vers l'hôte : " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(donneesDst);
        cudaFree(donneesSrcDevice);
        cudaStreamDestroy(stream);
        return;
    }

    
    cudaFree(donneesSrcDevice);
    cudaFree(donneesDstDevice);
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

    LaplacienDeGaussienne(donnees, nouvellesDonnees, largeur, hauteur, bpp, iterations);

    ilTexImage(largeur, hauteur, 1, bpp, format, IL_UNSIGNED_BYTE, nouvellesDonnees);

    ilEnable(IL_FILE_OVERWRITE);
    ilSaveImage(argv[3]);

    ilDeleteImages(1, &image);

    delete[] nouvellesDonnees;

    return 0;
}
