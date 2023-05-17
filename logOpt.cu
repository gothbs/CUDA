#/**include <iostream>
#include <cmath>
#include <IL/il.h>
#include <cuda_runtime.h>

__global__ void cudaLaplacienDeGaussienne(unsigned char *donneesSrc, unsigned char *donneesDst, int largeur, int hauteur, int bpp, int blockSizeX, int blockSizeY) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ unsigned char sharedBlock[16][16][4];

    int masque[5][5] = {
        {0, 0, -1, 0, 0},
        {0, -1, -2, -1, 0},
        {-1, -2, 16, -2, -1},
        {0, -1, -2, -1, 0},
        {0, 0, -1, 0, 0}
    };

    if (x < blockSizeX || y < blockSizeY || x + blockSizeX >= largeur || y + blockSizeY >= hauteur) {
        for (int c = 0; c < bpp; ++c) {
            donneesDst[(y * largeur + x) * bpp + c] = donneesSrc[(y * largeur + x) * bpp + c];
        }
        return;
    }

    for (int c = 0; c < bpp; ++c) {
        int somme = 0;

        if (threadIdx.x < blockSizeX && threadIdx.y < blockSizeY) {
            sharedBlock[threadIdx.y][threadIdx.x][c] = donneesSrc[((y - blockSizeY + threadIdx.y) * largeur + (x - blockSizeX + threadIdx.x)) * bpp + c];
            __syncthreads();
        }

        if (threadIdx.x >= blockSizeX / 2 && threadIdx.y >= blockSizeY / 2 && threadIdx.x < blockSizeX / 2 + blockSizeX && threadIdx.y < blockSizeY / 2 + blockSizeY) {
            for (int fy = -2; fy <= 2; ++fy) {
                for (int fx = -2; fx <= 2; ++fx) {
                    somme += masque[fy + 2][fx + 2] * sharedBlock[threadIdx.y - blockSizeY / 2 + fy][threadIdx.x - blockSizeX / 2 + fx][c];
                }
            }
        }

        donneesDst[(y * largeur + x) * bpp + c] = fminf(fmaxf(somme, 0), 255);
    }
}

void LaplacienDeGaussienneGPU(unsigned char *donnees, unsigned char *nouvellesDonnees, int largeur, int hauteur, int bpp, int iterations, int blockSizeX, int blockSizeY) {
    unsigned char *donneesSrc = donnees;
    unsigned char *donneesDst = nouvellesDonnees;

    unsigned char *donneesSrcDevice;
    unsigned char *donneesDstDevice;

    size_t size = largeur * hauteur * bpp * sizeof(unsigned char);

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void **)&donneesSrcDevice, size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erreur lors de l'allocation de mémoire sur le GPU pour les données source" << std::endl;
        return;
    }

    cudaStatus = cudaMalloc((void **)&donneesDstDevice, size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erreur lors de l'allocation de mémoire sur le GPU pour les données de destination" << std::endl;
        cudaFree(donneesSrcDevice);
        return;
    }

    cudaStatus = cudaMemcpy(donneesSrcDevice, donneesSrc, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erreur lors de la copie des données source de l'hôte vers le GPU ligne : 73"  << std::endl;
        cudaFree(donneesSrcDevice);
        cudaFree(donneesDstDevice);
        return;
    }

    dim3 blockSize(blockSizeX, blockSizeY);
    dim3 gridSize((largeur + blockSize.x - 1) / blockSize.x, (hauteur + blockSize.y - 1) / blockSize.y);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    for (int i = 0; i < iterations; ++i) {
        if (i % 2 == 0) {
            cudaLaplacienDeGaussienne<<<gridSize, blockSize, 0, stream1>>>(donneesSrcDevice, donneesDstDevice, largeur, hauteur, bpp, blockSizeX, blockSizeY);
        }
        else {
            cudaLaplacienDeGaussienne<<<gridSize, blockSize, 0, stream2>>>(donneesDstDevice, donneesSrcDevice, largeur, hauteur, bpp, blockSizeX, blockSizeY);
        }

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Erreur lors de l'exécution du kernel CUDA" << std::endl;
            cudaFree(donneesSrcDevice);
            cudaFree(donneesDstDevice);
            cudaStreamDestroy(stream1);
            cudaStreamDestroy(stream2);
            return;
        }
    }

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaFree(donneesSrcDevice);
    cudaFree(donneesDstDevice);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

int main(int argc, char *argv[]) {
    unsigned int image;

    // Initialisation de DevIL
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

    // Appliquer le filtre LoG sur l'image avec le nombre d'itérations spécifié
    LaplacienDeGaussienneGPU(donnees, nouvellesDonnees, largeur, hauteur, bpp, iterations, blockSizeX, blockSizeY);

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
**/

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
    unsigned char *devDonnees = nullptr;
    unsigned char *devNouvellesDonnees = nullptr;
    cudaMalloc((void**)&devDonnees, largeur * hauteur * bpp * sizeof(unsigned char));
    cudaMalloc((void**)&devNouvellesDonnees, largeur * hauteur * bpp * sizeof(unsigned char));

    cudaMemcpy(devDonnees, donnees, largeur * hauteur * bpp * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((largeur + blockDim.x - 1) / blockDim.x, (hauteur + blockDim.y - 1) / blockDim.y);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t sharedMemSize = 5 * 5 * sizeof(int);  // Taille de la mémoire partagée nécessaire pour le masque

    for (int i = 0; i < iterations; ++i) {
        LaplacianOfGaussianCUDA<<<gridDim, blockDim, sharedMemSize, stream>>>(devDonnees, devNouvellesDonnees, largeur, hauteur, bpp);
        cudaStreamSynchronize(stream);

        unsigned char *temp = devDonnees;
        devDonnees = devNouvellesDonnees;
        devNouvellesDonnees = temp;
    }

    cudaMemcpy(nouvellesDonnees, devNouvellesDonnees, largeur * hauteur * bpp * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(devDonnees);
    cudaFree(devNouvellesDonnees);
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
