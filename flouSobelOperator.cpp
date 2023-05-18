#include <iostream>
#include <cmath>
#include <IL/il.h>

void Sobel(unsigned char *donnees, unsigned char *nouvellesDonnees, int largeur, int hauteur, int bpp, int iterations) {
    int sobelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int sobelY[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    unsigned char *donneesSrc = donnees;
    unsigned char *donneesDst = nouvellesDonnees;

    for (int it = 0; it < iterations; ++it) {
        for (int y = 0; y < hauteur; ++y) {
            for (int x = 0; x < largeur; ++x) {
                if (x < 1 || y < 1 || x + 1 == largeur || y + 1 == hauteur) {
                    for (int c = 0; c < bpp; ++c) {
                        donneesDst[(y * largeur + x) * bpp + c] = donneesSrc[(y * largeur + x) * bpp + c];
                    }
                    continue;
                }

                for (int c = 0; c < bpp; ++c) {
                    int gradX = 0;
                    int gradY = 0;
                    for (int ky = -1; ky <= 1; ++ky) {
                        for (int kx = -1; kx <= 1; ++kx) {
                            int pixel = donneesSrc[((y + ky) * largeur + (x + kx)) * bpp + c];
                            gradX += pixel * sobelX[ky + 1][kx + 1];
                            gradY += pixel * sobelY[ky + 1][kx + 1];
                        }
                    }
                    int magnitude = sqrt(gradX * gradX + gradY * gradY);
                    donneesDst[(y * largeur + x) * bpp + c] = magnitude > 255 ? 255 : magnitude;
                }
            }
        }

        if (it < iterations - 1) {
            unsigned char *temp = donneesSrc;
            donneesSrc = donneesDst;
            donneesDst = temp;
        }
    }
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

    // Appliquer l'opérateur de Sobel sur l'image avec le nombre d'itérations spécifié
    Sobel(donnees, nouvellesDonnees, largeur, hauteur, bpp, iterations);

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