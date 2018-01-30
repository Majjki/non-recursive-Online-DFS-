# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import time
import py_compile
from collections import deque
from heapq import heappop, heappush
#import serial
#ser = serial.Serial('/dev/tty.usbmodem1411', 9600)

###
# Global Values 
###
fig = 1
image = []

xl = 0
yl = 0

def initialize(nrows,ncols):
    global fileExist
    fileExist = False 
    global image
    image = np.zeros(nrows*ncols)
    global futurePQ
    futurePQ = []
    image = image.reshape((nrows, ncols))  
    setWalls()
    plt.figure()
    global fig
    fig = plt.matshow(image,fignum = 0)  
    image[0,0] = 0 #för att ändra färgerna för simuleringar
    plt.ylim(plt.ylim()[::-1])  # reverse y-axis
    plt.xticks(range(ncols), range(ncols)) # range gör att man gör 0-args
    plt.yticks(range(nrows), range(nrows))
    plt.xticks(np.arange(-.5, ncols, 1))
    plt.yticks(np.arange(-.5, nrows, 1))
    # interactive mode on
    plt.ion()
    # visar plotten
    # pausar so den hinner rita ut
    plt.pause(0.0001)
    global fileMap
    try:
        with open('yardmap.tex','r'): 
            print("file exist")
            fileMap = open('yardmap.tex','r')
            fileExist = True

            pass
    except FileNotFoundError:
        print('File not existing, making a new...')
        fileMap = open('yardmap.tex', 'w')
        
###
# Sätt ut väggar för simulering endast!
###
def setWalls():
    image[0,0] = 5
    
    """image[7,7] = 2
    image[6,2] = 2
    #image[7,3] = 6
    image[1,3] = 2 
    image[9,7] = 2
    image[9,8] = 2
    image[9,9] = 2
    image[9,10] = 2
    image[6,11] = 2
    image[9,11] = 2
    image[9,12] = 2
    image[9,13] = 2
    image[9,14] = 2
    image[9,15] = 2
    image[9,16] = 2
    image[9,17] = 2
    image[8,16] = 2
    image[8,12] = 2
    #image[8,8] = 2
    image[7,8] = 2
    image[7,9] = 2
    image[7,10] = 2
    image[7,11] = 2
    image[7,12] = 2
    image[7,13] = 2
    image[7,14] = 2
    image[7,15] = 2
    image[7,16] = 2
    image[7,17] = 2
    image[0,4] = 2
    image[1,5] = 2
    image[3,9] = 2
    image[5,0] = 2
    image[3,4] = 2
    image[8,2] = 2
    image[16,6] = 2
    image[11,17] = 2
    image[1,17] = 2
    image[4,13] = 2
    image[17,15] = 2
    """
    image[5,5] = 2
    image[4,4] = 2
    image[6,4] = 2
    image[6,3] = 2
    image[6,2] = 2
    image[4,3] = 2
    #image[9,9] = 2
    image[4,2] = 2
    image[1,3] = 2
    image[1,4] = 2
    image[1,5] = 2
    image[1,6] = 2
    image[0,5] = 2
    image[5,4] = 2
    image[9,5] = 2
    image[9,4] = 2
    image[9,3] = 2
    image[9,2] = 2
    image[8,6] = 2
    image[7,6] = 2
    image[5,6] = 2
    image[6,7] = 2

    image[9,7] = 2
    image[9,8] = 2
    image[9,9] = 2
    image[7,8] = 2
    image[7,9] = 2
    image[7,8] = 2

    

    """
    image[4,11] = 2
    image[4,10] = 2

    image[3,9] = 2

    image[2,9] = 2
    image[1,9] = 2
    image[14,9] = 2
    """








    


def update():
    fig.set_data(image)
    #plt.imshow(image,interpolation='nearest') # för användning av A*
    plt.draw()
    #plt.matshow(image, fignum = 0)
    plt.pause(4.0001)
    #time.sleep(0.2) # För att delaya plotten

###
# Algoritm B
###
def recDFS(x, y, string, dweg):
    if image[x][y] == 0:       
        image[x][y] = 1
        old = image[x][y]
        image[x][y] = 5
        update()
        image[x,y] = old
        futurePQ.append([(x,y)])
        dweg += 1
        if y > 0:
            recDFS(x,y-1,"left",dweg)
        if x > 0:
            recDFS(x-1,y,"down",dweg)
        if x < xl - 1:
            recDFS(x+1,y,"up",dweg)
        if y < yl - 1:
            recDFS(x,y+1,"right",dweg)
    return(futurePQ)

###
# Kortaste Vägen mellan två punkter, Astar algoritm
# taget från http://bryukh.com/labyrinth-algorithms/
###
def heuristic(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def find_path_astar(maze,startnode,goalnode):
    #start, goal = (1, 1), (len(maze) - 2, len(maze[0]) - 2)
    start = startnode[0]
    goal = goalnode[0]
    print("goal and start",goal, start, image[goal[0],goal[1]], image[start[0],start[1]])
    pr_queue = []
    heappush(pr_queue, (0 + heuristic(start, goal), 0, [], start))
    visited = set()
    image[goal[0],goal[1]] = 1
    graph = maze2graph(maze)
    image[goal[0],goal[1]] = 0

    while pr_queue:
        _, cost, path, current = heappop(pr_queue)


        #print(current)
        x = current[0] 
        y = current[1]
        
        """old = image[x][y]
        image[x][y] = 5
        update()
        image[x,y] = old
        image[(x,y)] = 1 """
        

        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            heappush(pr_queue, (cost + heuristic(neighbour, goal), cost + 1,
                                path + direction, neighbour))
    return "NO WAY!"

def find_path_bfs(maze,startnode,goalnode):
    #start, goal = (1, 1), (len(maze) - 2, len(maze[0]) - 2)
    start = startnode[0]
    goal = goalnode[0]
    print("goal and start",goal, start, image[goal[0],goal[1]], image[start[0],start[1]])
    queue1 = deque([([], start)])
    visited = set()
    image[goal[0],goal[1]] = 1
    graph = maze2graph(maze)
    image[goal[0],goal[1]] = 0

    print("this is graph", graph)
    while queue1:
        path, current = queue1.popleft()
        print(current)
        x = current[0] 
        y = current[1]
        
        """old = image[x][y]
        image[x][y] = 5
        update()
        image[x,y] = old
        image[(x,y)] = 1""" 

        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            queue1.append((path + direction, neighbour))
    return "NO WAY!"


###
# Hittar en ny punkt långt ifrån nuvarande
# Genom att leta efter närmsta grannen i minnet
###
def findPath(tuplez,futurePQ):
    #tuplez[0] = prev
    #tuplez[1] = curr
    px = tuplez[1][0][0]
    py = tuplez[1][0][1]
    for k in [[px,py-1],[px+1,py],[px,py+1],[px-1,py]]:
        if(k[0] >= 0 and k[0] < xl-1 and k[1] >= 0 and k[1] < yl-1):
            if(image[k[0]][k[1]] == 1 and [(k[0],k[1])] in futurePQ ):
                indexP = futurePQ.index([(k[0],k[1])])                            
                indexc = futurePQ.index(tuplez[0])
                return(indexP,indexc)
    print("NOTHING")            

###
# Algoritm C - Online-DFS-algoritm anpassad till projektet
###
def DFS2(x,y):
    stack = deque([[(x,y)]])
    visited = []
    index = 0
    prev = ([(x,y)])
    while stack:
        current = stack.pop()
        if current in visited:
            continue        
        x = current[0][0]  
        y = current[0][1]        
        if(abs(prev[0][0] - x) > 1 or 
                abs(prev[0][1] - y) > 1 or 
                    (abs(prev[0][0] - x) >= 1 and 
                        abs(prev[0][1] - y) >= 1)  ):
            print("curr and prev§",[(x,y)],prev, image[x,y], image[prev[0][0]][prev[0][1]])
            
            reverse = find_path_astar(image,prev,[(x,y)])
            #reverse = find_path_bfs(image,prev,[(x,y)])
            print(reverse)
            walkPath(deque(reverse))
            #tuplepath = findPath([prev,current],visited)
            # print("this is tuple path",tuplepath)
            """
            reversePath = (visited[tuplepath[0]:tuplepath[1]][::-1]) 
            visited = visited[:tuplepath[1]] + reversePath + visited[tuplepath[1]:]
            reversePath = deque(reversePath)
            walkPath(reversePath)"""
        prev=([(x,y)])
        old = image[x][y]
        image[x][y] = 5
        update()
        image[x,y] = old
        visited.append([(x,y)])
        image[(x,y)] = 1 
        if((x,y+1) not in visited and y+1 < yl ):
            #print("y+1 and yl", y+1, yl)
            if image[x][y+1] == 0 :
                stack.append([(x,y+1)])            
        if((x+1,y) not in visited and x+1 < xl ):
            if image[x+1][y] == 0 :
                stack.append([(x+1,y)])  
        if((x-1,y) not in visited and x-1 >= 0 ):
            if image[x-1][y] == 0 :
                stack.append([(x-1,y)])  
        if((x,y-1) not in visited and y-1 >= 0 ):
            if image[x][y-1] == 0 :
                stack.append([(x,y-1)])      
        index += 1

    return visited

###
# Gör om en grid till en graf
# taget från http://bryukh.com/labyrinth-algorithms/
###
def maze2graph(maze):

    height = len(maze)
    width = len(maze[0]) if height else 0
    #maze = maze - [[1 for x in range(height)]]
    graph = {(i, j): [] for j in range(width) for i in range(height) if maze[i][j] != 2}
    for row, col in graph.keys():
        if row < height - 1 and (maze[row + 1][col] != 2):
            graph[(row, col)].append(([(row, col)], (row + 1, col))) #S
            graph[(row + 1, col)].append(([(row + 1, col)], (row, col))) #N
        if col < width - 1 and maze[row][col + 1] != 2:
            graph[(row, col)].append(([(row, col)], (row, col + 1))) #E
            graph[(row, col + 1)].append(([(row, col + 1)], (row, col))) #W
    #print("da graph",graph)
    return graph

###
# Läser från minnet
###
def readPath():
    fileMap.read()
    print(fileMap)

###
# Gör en sicksack-spår åt algoritm A
###
def makePath(x,y):
    print("check if fime exist")
    if(fileExist):
        print("map exist, taking from map...")
        readPath()
    
    print("file does not exist, making a new one")

    queue = deque([(x,y)])
    index = 1
    counter = 1
    while(counter < (xl*yl)):
        counter += 1
        if y > 0 and y < yl-1:
            y += index
            queue.append(([y,x])) 
        elif y == yl-1 and index < 0:
            y += index
            queue.append(([y,x])) 
        elif y == 0 and index > 0:
            y += index 
            queue.append([y,x])
        else :
            if x >= 0 and x < xl-1:
                x += 1
                index = index * (-1) 
                queue.append([y,x])
    return queue
  
###
# Algoritm A som går efter sicksack-karta
### 
def walkPath(queue):
    #print("this is queue", queue)
    xold = 0
    yold = 0
    start = 0
    #print("this is queue", queue)
    #print("this is image", image)
    while len(queue) > 0:
        node = queue.popleft()
        #print("this is noode", node)
        #print("this is node",node)
        if node == 'N':
            print("ERROR NO WAY")
            break
        try:
            x = node[0]
            y = node[1]
        except IndexError:
            x = node[0][0]
            y = node[0][1]
        
        if(abs(x-xold) > 1 or abs(y-yold) > 1 or (abs(x-xold) >= 1 and abs(y-yold >= 1)) ):
            if start != 0:
                print("going from", ([xold,yold]), " to ", ([x,y]))
                walkPath(deque(find_path_astar(image,[(xold,yold)],[(x,y)])))

        
        old = image[x][y]
        image[x][y] = 5
        update()
        image[x,y] = old     
        objectAvoidz(x,y,xold - x, yold-y, queue)
        if(image[x,y] != 2):
            image[x,y] = 1  
        start = 1
        xold = x
        yold = y


###
# Hinderhantering, "rundar" objektet och närliggande. 
###
def objectAvoidz(x,y,xo,yo,queue):
    l = []
    #print("avoid",x,y)
    #print("is object?", image[x][y],image[x][y] == 2)
    if image[x][y] == 2:
        print("obstacle at ",x,y,xo,yo)
        if xo > 0:
            print("down")
            #drive("forward");
            l = [[x-1,y],[x-1,y-1],[x,y-1],[x+1,y-1]]
        elif xo < 0:
            print("up")
            #drive("forward");
            l = [[x+1,y-1],[x,y-1],[x-1,y-1],[x-1,y]]
        elif yo > 0:
            print("left")
            #drive("left");
            l = [[x,y-1],[x-1,y-1],[x-1,y],[x-1,y+1]]
        elif yo < 0:
            print("right")  
            #drive("left")
            l = [[x-1,y+1],[x-1,y],[x-1,y-1],[x,y-1]]        
        
        for i in l:
            if(i[1] > 0 and i[1] < xl-1 and i[0] > 0 and i[0] < yl-1):
                print("adding", l)
                if(image[i[0]][i[1]] != 2):
                    queue.appendleft((i[0],i[1]))             
                else:
                    objectAvoidz(i[0],i[1],x-i[0],y-i[1],queue)

###
# Optimerar en rutt så den blir bättre
###
def learnQueue(futurePQ):
    learnedQueue = []
    for i in range(0,len(futurePQ)):
        if i < len(futurePQ)-1:
            
            try:                
                x = futurePQ[i][0][0]
                try:
                    y = futurePQ[i][0][1] 
                except IndexError:
                    continue
                x + 1

            except TypeError:
                x = futurePQ[i][0][0][0]
                y = futurePQ[i][0][0][1]
            try: 
                xn = futurePQ[i+1][0][0]
                try:
                    yn = futurePQ[i+1][0][1]
                except IndexError:
                    continue
                xn + 1

            except TypeError:
                xn = futurePQ[i+1][0][0][0]
                yn = futurePQ[i+1][0][0][1]
           
            if (abs(x-xn >= 2) or abs(y-yn >= 2)):
                popedNode = futurePQ.pop(i+1)
                print("jumped from", popedNode, " to " ,x,y)

                px = popedNode[0][0]
                py = popedNode[0][1]
            
                for k in [[px,py-1],[px+1,py],[px,py+1],[px-1,py]]:
                    if(k[0] >= 0 and k[0] < xl-1 and k[1] >= 0 and k[1] < yl-1):
                        if(image[k[0]][k[1]] == 1 and [(k[0],k[1])] in futurePQ ):
                            indexP = futurePQ.index([(k[0],k[1])])
                            futurePQ = futurePQ[:(indexP)] + [popedNode] + futurePQ[(indexP):] 
                            break
            else:
                learnedQueue.append(futurePQ[i])    
    return(futurePQ)

###
# Sparar kartan
###
def saveMap():
    fileMap.write('['+(''.join(str(e) for e in image))+']')

###
# Sparar sina steg
###
def saveQueue(futurePQ):
    print("saveing")
    futurePQ = learnQueue(futurePQ)    
    string = str(futurePQ)
    fileMap.write(string)  
    
###
# Går efter sitt minne av stegen
###
def getQueue():
    queue = deque([(0,0)])
    
    for i in fileMap.read().strip('[[').strip(']]').split('], ['):
        i = i[1:]
        i = (i.strip('(').strip(')').strip('\n').strip(']]').strip(')').strip('').split(','))
        i = list(map(int, i))
        queue.append(i)    
    walkPath(queue)
    
###
# Huvudkoden
###
def main():
    global xl #hur långt i x-axeln   
    global yl #hur långt i y-axeln
    xl = 12  
    yl = 12
    initialize(xl,yl) #gör area av x och y
    print(input('start? > '))
    #find_path_astar(image,[(6,2)],[(8,8)])
    #walkPath(makePath(0,0))
    if(fileExist):
        print("get queue")
        getQueue()   
    else:
        print("dfs explore!")
        futurePQ = DFS2(0,0)  #Använd algoritm C
        #futurePQ = recDFS(0,0,"begin",0) #Använd algoritm B
        saveQueue(futurePQ)


main()


