#include <iostream>
#include <cmath>
#include <IL/il.h>
#include <cuda_runtime.h>

// Fonction CUDA pour appliquer le flou boîte
__global__ void cudaFlouBoite(unsigned char *donneesSrc, unsigned char *donneesDst, int largeur, int hauteur, int bpp) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= 1 && y >= 1 && x + 1 < largeur && y + 1 < hauteur) {
        for (int c = 0; c < bpp; ++c) {
            int somme = donneesSrc[((y - 1) * largeur + (x - 1)) * bpp + c] +
                        donneesSrc[((y - 1) * largeur + x) * bpp + c] +
                        donneesSrc[((y - 1) * largeur + (x + 1)) * bpp + c] +
                        donneesSrc[(y * largeur + (x - 1)) * bpp + c] +
                        donneesSrc[(y * largeur + x) * bpp + c] +
                        donneesSrc[(y * largeur + (x + 1)) * bpp + c] +
                        donneesSrc[((y + 1) * largeur + (x - 1)) * bpp + c] +
                        donneesSrc[((y + 1) * largeur + x) * bpp + c] +
                        donneesSrc[((y + 1) * largeur + (x + 1)) * bpp + c];

            donneesDst[(y * largeur + x) * bpp + c] = somme / 9;
        }
    }
}

void FlouBoiteGPU(unsigned char *donnees, unsigned char *nouvellesDonnees, int largeur, int hauteur, int bpp, int iterations, int blockSizeX, int blockSizeY) {
    unsigned char *donneesSrc = donnees;
    unsigned char *donneesDst = nouvellesDonnees;
    
    unsigned char *donneesSrcDevice;
    unsigned char *donneesDstDevice;
    
    size_t size = largeur * hauteur * bpp * sizeof(unsigned char);
    
    cudaMalloc((void **)&donneesSrcDevice, size);
    cudaMalloc((void **)&donneesDstDevice, size);
    
    cudaMemcpy(donneesSrcDevice, donneesSrc, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(blockSizeX, blockSizeY);
    dim3 gridSize((largeur + blockSize.x - 1) / blockSize.x, (hauteur + blockSize.y - 1) / blockSize.y);
    
    for (int i = 0; i < iterations; ++i) {
        cudaFlouBoite<<<gridSize, blockSize>>>(donneesSrcDevice, donneesDstDevice, largeur, hauteur, bpp);
        
        std::swap(donneesSrcDevice, donneesDstDevice);
    }
    
    cudaMemcpy(donneesDst, donneesSrcDevice, size, cudaMemcpyDeviceToHost);
    
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