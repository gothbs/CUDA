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

        if (x >= 2 && y >= 2 && x + 2 < width && y + 2 < height) {
            for (int c = 0; c < bpp; ++c) {
                int somme = 0;
                for (int fy = -2; fy <= 2; ++fy) {
                    for (int fx = -2; fx <= 2; ++fx) {
                        somme += masque[fy + 2][fx + 2] * input[((y + fy) * width + (x + fx)) * bpp + c];
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

    for (int i = 0; i < iterations; ++i) {
        LaplacianOfGaussianCUDA<<<gridDim, blockDim>>>(devDonnees, devNouvellesDonnees, largeur, hauteur, bpp);
        cudaDeviceSynchronize();

        unsigned char *temp = devDonnees;
        devDonnees = devNouvellesDonnees;
        devNouvellesDonnees = temp;
    }

    cudaMemcpy(nouvellesDonnees, devNouvellesDonnees, largeur * hauteur * bpp * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(devDonnees);
    cudaFree(devNouvellesDonnees);
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
