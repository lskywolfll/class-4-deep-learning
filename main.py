import torch

# la estructura típica de una imagen: canales de color, alto y ancho en píxeles.
# Cada posición [canal, fila, columna] contiene un valor (en este caso, aleatorio) que representa la intensidad del color en ese canal y píxel.

# channels => rgb (rojo,verde,azul) para extraer los colores de una imagen -> el canal es que informaciones quiero
# obtener desde la imagen
# height => 224 para obtener la altura de la imagen
# width => 244 para obtener el ancho de la imagen
channels = 3
height = 224
width = 244
new_tensor = torch.randn((channels, height, width))

print(new_tensor)