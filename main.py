import torch

#images
# channels => rgb (rojo,verde,azul) para extraer los colores de una imagen -> el canal es que informaciones quiero
# obtener desde la imagen
# height => 224 para obtener la altura de la imagen
# width => 224 para obtener el ancho de la imagen
channels = 3
height = 224
width = 224
new_tensor = torch.randn((channels, height, width))

print(new_tensor)