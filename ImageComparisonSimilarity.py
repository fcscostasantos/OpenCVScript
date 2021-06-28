#    - Esse código utiliza o modulo ORB da biblioteca OpenCV.
#    - Para esse exemplo não foi necessário utilizar p modulo numpy,
#    - pois o próprio modulo ORB já possui recurso para trabalhar as imagens.


# importação módulo Opencv
import cv2


# Works well with images of different dimensions
# Foi criando um função para recepcionar as duas imagens
def orb_sim(img1, img2):

    # Antigamente era utilizado o módulo SIFT, mas o mesmo foi descontinuado e o ORB passou ser utilizado
    # possuindo mais recursos para trabalhar as imagens
    orb = cv2.ORB_create()

    # Esse é o ponto onde serão dectados pontos chaves e descritores das imagens
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # Definir o objeto combinador de força bruta.
    # Isso quer dizer que ele irá testar muitas combinações diferentes.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Nessa etapa será feito o match das imagens, ou seja, a combinação das imagens
    matches = bf.match(desc_a, desc_b)

    # Aqui será feito a procura por regiões similares das imagens com a distância menor que 50.
    # É possível ajustar a distância entre 0 a 100, sendo 0 as imagens totalmente diferente e 100 as imagens iguais
    # Nesse caso é escolhido a distância de 50.
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)

# Função do Opencv para fazer a leitura das imagens
img00 = cv2.imread('sig3.jpg', 0)
img01 = cv2.imread('sig4.jpg', 0)

# Função opencv para redimensionar as inmagens, pois se elas estivem em dimensões diferente dá erro no código
img00Res = cv2.resize(img00, (1200, 600))
img01Res = cv2.resize(img01, (1200, 600))

# Essa é a etapa on é chamada a função criada no início do código passando como parâmetro as imagens redimensionadas.
# Se o resultado for igual 1.0, quer dizer as imagens são extamente iguais.
# Qualquer outro resultado que apresentar vai representar o percentual de similaridade entre as imagens.
orb_similarity = orb_sim(img00Res, img01Res)
print("O Percentual de Similaridade Utilizando o ORB é: ", orb_similarity)