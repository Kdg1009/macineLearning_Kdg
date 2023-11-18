def getPoints(add='y',confirm='n',points=[]):
    points.clear()
    while not add == 'n':
        p1=tuple(map(int,input('input point1: ').split()))
        p2=tuple(map(int,input('input point2: ').split()))
        confirm=input('are you sure? y/n: ')
        if confirm == 'y':
            points.append([p1,p2])
        add=input('more points? y/n:')
    return points