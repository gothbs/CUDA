#include <iostream>
#include <cmath>
#include <IL/il.h>

// Matrice du filtre LoG (Laplacien de Gaussienne) 5x5
int filtre[5][5] = {
    {0, 0, -1, 0, 0},
    {0, -1, -2, -1, 0},
    {-1, -2, 16, -2, -1},
    {0, -1, -2, -1, 0},
    {0, 0, -1, 0, 0}
};


/***  noire et blanc **/
/**void LaplacienDeGaussienne(unsigned char *donnees, unsigned char *nouvellesDonnees, int largeur, int hauteur, int bpp, int iterations) {
    unsigned char *donneesSrc = donnees;
    unsigned char *donneesDst = nouvellesDonnees;

    for (int it = 0; it < iterations; ++it) {
        for (int y = 0; y < hauteur; ++y) {
            for (int x = 0; x < largeur; ++x) {
                if (x < 2 || y < 2 || x + 2 >= largeur || y + 2 >= hauteur) {
                    for (int c = 0; c < bpp; ++c) {
                        donneesDst[(y * largeur + x) * bpp + c] = donneesSrc[(y * largeur + x) * bpp + c];
                    }
                    continue;
                }

                for (int c = 0; c < bpp; ++c) {
                    int somme = 0;
                    for (int ky = -2; ky <= 2; ++ky) {
                        for (int kx = -2; kx <= 2; ++kx) {
                            somme += donneesSrc[((y + ky) * largeur + (x + kx)) * bpp + c] * filtre[ky + 2][kx + 2];
                        }
                    }

                    // Assurez-vous que la valeur est dans la plage valide [0, 255]
                    if (somme < 0) somme = 0;
                    if (somme > 255) somme = 255;

                    donneesDst[(y * largeur + x) * bpp + c] = somme;
                }
            }
        }

        if (it < iterations - 1) {
            unsigned char *temp = donneesSrc;
            donneesSrc = donneesDst;
            donneesDst = temp;
        }
    }
}**/

/** couleur **/

void LaplacienDeGaussienne(unsigned char *donnees, unsigned char *nouvellesDonnees, int largeur, int hauteur, int bpp, int iterations) {
    int masque[5][5] = {
        {0, 0, -1, 0, 0},
        {0, -1, -2, -1, 0},
        {-1, -2, 16, -2, -1},
        {0, -1, -2, -1, 0},
        {0, 0, -1, 0, 0}
    };

    for (int i = 0; i < iterations; ++i) {
        for (int y = 0; y < hauteur; ++y) {
            for (int x = 0; x < largeur; ++x) {
                for (int c = 0; c < bpp; ++c) {
                    if (x < 2 || y < 2 || x + 2 >= largeur || y + 2 >= hauteur) {
                        nouvellesDonnees[(y * largeur + x) * bpp + c] = donnees[(y * largeur + x) * bpp + c];
                        continue;
                    }

                    int somme = 0;
                    for (int fy = -2; fy <= 2; ++fy) {
                        for (int fx = -2; fx <= 2; ++fx) {
                            somme += masque[fy+2][fx+2] * donnees[((y + fy) * largeur + (x + fx)) * bpp + c];
                        }
                    }

                    nouvellesDonnees[(y * largeur + x) * bpp + c] = std::min(std::max(somme, 0), 255);
                }
            }
        }

        if (i < iterations - 1) {
            unsigned char *temp = donnees;
            donnees = nouvellesDonnees;
            nouvellesDonnees = temp;
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

    // Appliquer le filtre LoG sur l'image avec le nombre d'itérations spécifié
    LaplacienDeGaussienne(donnees, nouvellesDonnees, largeur, hauteur, bpp, iterations);

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
