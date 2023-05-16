#include <iostream>
#include <cmath>
#include <IL/il.h>
#include <cuda_runtime.h>

// Fonction CUDA pour appliquer le flou boîte
__global__ void cudaFlouBoite(unsigned char *donneesSrc, unsigned char *donneesDst, int largeur, int hauteur, int bpp, cudaStream_t stream) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    // Déclaration de la mémoire partagée
    __shared__ unsigned char sharedData[16][16][3];
    
    if (x >= 1 && y >= 1 && x + 1 < largeur && y + 1 < hauteur) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        
        // Chargement des données dans la mémoire partagée
        sharedData[ty][tx][0] = donneesSrc[((y - 1) * largeur + (x - 1)) * bpp];
        sharedData[ty][tx][1] = donneesSrc[((y - 1) * largeur + x) * bpp];
        sharedData[ty][tx][2] = donneesSrc[((y - 1) * largeur + (x + 1)) * bpp];
        sharedData[ty+1][tx][0] = donneesSrc[(y * largeur + (x - 1)) * bpp];
        sharedData[ty+1][tx][1] = donneesSrc[(y * largeur + x) * bpp];
        sharedData[ty+1][tx][2] = donneesSrc[(y * largeur + (x + 1)) * bpp];
        sharedData[ty+2][tx][0] = donneesSrc[((y + 1) * largeur + (x - 1)) * bpp];
        sharedData[ty+2][tx][1] = donneesSrc[((y + 1) * largeur + x) * bpp];
        sharedData[ty+2][tx][2] = donneesSrc[((y + 1) * largeur + (x + 1)) * bpp];
        
        __syncthreads();
        
        if (tx > 0 && tx < blockDim.x - 1 && ty > 0 && ty < blockDim.y - 1) {
            for (int c = 0; c < bpp; ++c) {
                int somme = sharedData[ty-1][tx-1][c] + sharedData[ty-1][tx][c] + sharedData[ty-1][tx+1][c]
                          + sharedData[ty][tx-1][c]   + sharedData[ty][tx][c]   + sharedData[ty][tx+1][c]
                          + sharedData[ty+1][tx-1][c] + sharedData[ty+1][tx][c] + sharedData[ty+1][tx+1][c];
                
                donneesDst[(y * largeur + x) * bpp + c] = somme / 9;
            }
        }
    }
}

void FlouBoiteGPU(unsigned char *donnees, unsigned char *nouvellesDonnees, int largeur, int hauteur, int bpp, int iterations, int blockSizeX, int blockSizeY) {
    unsigned char *donneesSrc = donnees;
    unsigned char *donneeosDst = nouvellesDonnees;
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
    // Création des streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    for (int i = 0; i < iterations; ++i) {
        if (i % 2 == 0) {
            cudaFlouBoite<<<gridSize, blockSize, 0, stream1>>>(donneesSrcDevice, donneesDstDevice, largeur, hauteur, bpp, stream1);
        } else {
            cudaFlouBoite<<<gridSize, blockSize, 0, stream2>>>(donneesSrcDevice, donneesDstDevice, largeur, hauteur, bpp, stream2);
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

        std::swap(donneesSrcDevice, donneesDstDevice);
        
        // Synchronisation des streams
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
    }

    // Destruction des streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaMemcpyAsync(donneesDst, donneesSrcDevice, size, cudaMemcpyDeviceToHost);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erreur lors de la copie des données de destination du GPU vers l'hôte" << std::endl;
        cudaFree(donneesSrcDevice);
        cudaFree(donneesDstDevice);
        return;
    }

    cudaFree(donneesSrcDevice);
    cudaFree(donneesDstDevice);

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

    // Appliquer le flou boîte sur l'image avec le nombre d'itérations spécifié
    FlouBoiteGPU(donnees, nouvellesDonnees, largeur, hauteur, bpp, iterations, blockSizeX, blockSizeY);

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