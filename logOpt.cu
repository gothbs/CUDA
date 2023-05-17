#include <iostream>
#include <cmath>
#include <IL/il.h>
#include <cuda_runtime.h>

__global__ void cudaLaplacienDeGaussienne(unsigned char *donneesSrc, unsigned char *donneesDst, int largeur, int hauteur, int bpp, int blockSizeX, int blockSizeY) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    __shared__ unsigned char sharedBlock[blockSizeY][blockSizeX][4];

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

        if (threadIdx.x >= blockSizeX/2 && threadIdx.y >= blockSizeY/2 && threadIdx.x < blockSizeX/2 + blockSizeX && threadIdx.y < blockSizeY/2 + blockSizeY) {
            for (int fy = -2; fy <= 2; ++fy) {
                for (int fx = -2; fx <= 2; ++fx) {
                    somme += masque[fy + 2][fx + 2] * sharedBlock[threadIdx.y - blockSizeY/2 + fy][threadIdx.x - blockSizeX/2 + fx][c];
                }
            }
        }

        donneesDst[(y * largeur + x) * bpp + c] = fminf(fmaxf(somme, 0), 255);
    }
}

void LaplacienDeGaussienneGPU(unsigned char *donnees, unsigned char *nouvellesDonnees, int largeur, int hauteur, int bpp, int iterations, int blockSizeX, int blockSizeY) {
    unsigned char *donneesSrc = donnees;

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
        std::cerr << "Erreur lors de la copie des données source de l'hôte vers le GPU" << std::endl;
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
        } else {
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

    if (iterations % 2 == 1) {
        cudaMemcpyAsync(nouvellesDonnees, donneesSrcDevice, size, cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpyAsync(nouvellesDonnees, donneesDstDevice, size, cudaMemcpyDeviceToHost);
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erreur lors de la copie des données de destination du GPU vers l'hôte" << std::endl;
        cudaFree(donneesSrcDevice);
        cudaFree(donneesDstDevice);
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        return;
    }

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
