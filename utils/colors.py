import matplotlib.pyplot as plt

def continuous_color(values, opacity=None):
    norm_v = plt.Normalize(values.min(), values.max())
    if opacity is None:
        opacity =[norm_v(value) for value in values]
    else:
        norm_o = plt.Normalize(opacity.min(), opacity.max())
        opacity = [norm_o(o) for o in opacity]
    print(opacity)
    cmap = plt.cm.turbo
    rgba_colors = values.apply(lambda x: cmap(norm_v(x)))
    rgbas = [
        f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{o})'
        for color, o in zip(rgba_colors, opacity)
    ]
    print(rgbas)
    return rgbas