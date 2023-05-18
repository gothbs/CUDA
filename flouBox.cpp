#include <iostream>
#include <cmath>
#include <chrono>
#include <IL/il.h>

void FlouBoite(unsigned char *donnees, unsigned char *nouvellesDonnees, int largeur, int hauteur, int bpp, int iterations) {
    unsigned char *donneesSrc = donnees;
    unsigned char *donneesDst = nouvellesDonnees;

    for (int i = 0; i < iterations; ++i) {
        for (int y = 0; y < hauteur; ++y) {
            for (int x = 0; x < largeur; ++x) {
                if (x < 1 || y < 1 || x + 1 == largeur || y + 1 == hauteur) {
                    for (int c = 0; c < bpp; ++c) {
                        donneesDst[(y * largeur + x) * bpp + c] = donneesSrc[(y * largeur + x) * bpp + c];
                    }
                    continue;
                }

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

        if (i < iterations - 1) {
            unsigned char *temp = donneesSrc;
            donneesSrc = donneesDst;
            donneesDst = temp;
        }
    }
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

    // Mesurer le temps de calcul
    auto start = std::chrono::high_resolution_clock::now();

    // Appliquer le flou boîte sur l'image avec le nombre d'itérations spécifié
    FlouBoite(donnees, nouvellesDonnees, largeur, hauteur, bpp, iterations);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Temps de calcul cpu flouBox : " << elapsed.count() << " secondes" << std::endl;

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

