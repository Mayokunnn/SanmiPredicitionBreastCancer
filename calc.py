def calculate_real_size(image_size, magnification):
    try:
        real_size = image_size / magnification
        return round(real_size, 2)
    except ZeroDivisionError:
        return "Magnification cannot be zero"
    except Exception as e:
        return str(e)
