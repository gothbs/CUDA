#include <iostream>
#include <cmath>
#include <IL/il.h>
#include <cuda_runtime.h>

// Le noyau du filtre gaussien
__constant__ float noyauGaussien[5][5] = {
    {1.0f, 4.0f, 7.0f, 4.0f, 1.0f},
    {4.0f, 16.0f, 26.0f, 16.0f, 4.0f},
    {7.0f, 26.0f, 41.0f, 26.0f, 7.0f},
    {4.0f, 16.0f, 26.0f, 16.0f, 4.0f},
    {1.0f, 4.0f, 7.0f, 4.0f, 1.0f}
};

// Appliquer le flou gaussien sur l'image en utilisant CUDA
__global__ void cudaFlouGaussien(unsigned char *donneesSrc, unsigned char *donneesDst, int largeur, int hauteur, int bpp) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= 2 && y >= 2 && x + 2 < largeur && y + 2 < hauteur) {
        for (int c = 0; c < bpp; ++c) {
            float somme = 0.0f;
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    somme += donneesSrc[((y + ky) * largeur + (x + kx)) * bpp + c] * noyauGaussien[ky + 2][kx + 2];
                }
            }
            donneesDst[(y * largeur + x) * bpp + c] = static_cast<unsigned char>(somme);
        }
    }
}

void FlouGaussienGPU(unsigned char *donnees, unsigned char *nouvellesDonnees, int largeur, int hauteur, int bpp, int iterations, int blockSizeX, int blockSizeY) {
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
        cudaFlouGaussien<<<gridSize, blockSize>>>(donneesSrcDevice, donneesDstDevice, largeur, hauteur, bpp);
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

    // Normaliser le noyau du filtre gaussien
    float somme = 0.0f;
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

    // Appliquer le flou gaussien sur l'image avec le nombre d'itérations spécifié
    FlouGaussienGPU(donnees, nouvellesDonnees, largeur, hauteur, bpp, iterations, 16, 16);

    // Mettre à jour l'image avec les données traitées
    ilTexImage(largeur, hauteur, 1, bpp, format, IL_UNSIGNED_BYTE, nouvellesDonnees);

    // Activer l'écrasement de fichier lors de la sauvegarde
    ilEnable(IL_FILE_OVERWRITE);
    ilSaveImage(argv[3]);

    // Supprimer l'image de la mémoire de DevIL
    ilDeleteImages(1, &image);

    // Libérer la mémoire allouée pour les nouvelles données de l'image
    delete[] nouvellesDonnees;

    return 0;
}

