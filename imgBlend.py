import cv2

def blend(blenders,img_path,confirm='n'):
    img=cv2.imread(img_path)
    cv2.imwrite('blended.jpg',img)
    while not confirm == 'y':
        color = input('select color theme 1.red 2.purple 3.black: ')
        confirm=input('are you sure? y/n: ')
    color=int(color)
    blended=cv2.multiply(blenders[color],img)
    cv2.imwrite('blended.jpg',blended)