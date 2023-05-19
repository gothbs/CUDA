# CUDA

 - Le make produit 12 executables :
    - Les 4 premiers sont du coté cpu , elles prennent en paramètre l'image à flitrer
        - le nombre de fois que l'on souhaite appliquer le filtre et le nom de l'image produite
        - exemple : ./flouBox in.jpg 4 out_in.jpg

    - Les 4 suivants sont coté gpu sont de la forme XXXGpu 
        - elles prennent en charge de nouveeau paramètre le nombre de Thread en x et en y
        - exemple : ./flouBoxGpu in.jpg 4 16 16 out_in.jpg    
   
    - Les 4 suivants sont des versions optimisé
        - de la forme suivante XXXOpt 
        - ajout de la mémoire partagée au niveau du stream  et l'utilisation de stream afin d'executer les itération en asynchrone
        - Les paramètres sont les mêmes que coté gpu basic   