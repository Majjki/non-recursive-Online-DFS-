# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import time
import py_compile
from collections import deque

#import serial
#ser = serial.Serial('/dev/tty.usbmodem1411', 9600)

####

fig = 1
image = []
xl = 0
yl = 0

def initialize(nrows,ncols):
    global fileExist
    fileExist = False
    # Make a 9x9 grid...
    # nrows, ncols = 9,9
    # nrows = input('rows : ')
    # ncols = input('cols : ')
    global image
    image = np.zeros(nrows*ncols)
    global futurePQ
    futurePQ = []
    # Set every other cell to a random number (this would be your data)
    # image[::2] = np.random.random(nrows*ncols //2 + 1)
    # image[0::3] = börjar från col 0, och färgar var 3e
    # image[1:3] = färgar endast rad 1 col 3.
    
    # Reshape things into a 9x9 grid.
    image = image.reshape((nrows, ncols))
    
    setWalls()

    plt.figure()
    global fig
    #fig = plt.imshow(image, interpolation='nearest')
    fig = plt.matshow(image,fignum = 0)
    
    image[4,4] = 2
    plt.ylim(plt.ylim()[::-1]) # reverse y-axis
    # ax = plt.gca();
    # plt.xlim(plt.xlim()[::-1])
    plt.xticks(range(ncols), range(ncols)) # range gör att man gör 0-args
    plt.yticks(range(nrows), range(nrows))
    plt.xticks(np.arange(-.5, ncols, 1))
    plt.yticks(np.arange(-.5, nrows, 1))
    # interactive mode on
    plt.ion()
    # visar plotten
    # pausar so den hinner rita ut
    plt.pause(0.0001)

    #with open(thepath, 'a'): pass 
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
        

def setWalls():
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
    image[17,15] = 2"""
    image[5,5] = 2
    image[4,4] = 6

def update():
    #while(True):
        #var1, var2, var3 = [int(x) for x in raw_input("X,Y,C: ").split(',')]
        #image[var2,var1] = a = var3 if var3 != 0 else 0
        #print(var1,var2)
    fig.set_data(image)
    #plt.imshow(image,interpolation='nearest')
    plt.draw()
    #plt.matshow(image, fignum = 0)
    plt.pause(1.0001)
    #time.sleep(0.1)

def floodfill(x, y, string, dweg):

    #"hidden" stop clause - not reinvoking for "c" or "b", only for "a".
    if image[x][y] == 0:
        
        image[x][y] = 1
        #print(string)
        old = image[x][y]
        image[x][y] = 5
        update()
        image[x,y] = old
        futurePQ.append([(x,y)])
        dweg += 1
        print(dweg, " at ", (x,y))

        if(x == 0 and y == 3):
            print("THIS IS IT")
            print([(x,y)])
        if y > 0:
            floodfill(x,y-1,"left",dweg)
        if x > 0:
            floodfill(x-1,y,"down",dweg)
        if x < xl - 1:
            floodfill(x+1,y,"up",dweg)
        if y < yl - 1:
            floodfill(x,y+1,"right",dweg)
        """
        while(x < xl):
            image[x+1][y]
        while(x < 0 and x != xl):
            image[x-1][y]
        while(y < yl and x == xl or x == 0):
            image[x][y+1]
        """
        print("at the end", dweg,string)
        print("this is length",len(futurePQ))
        print("this is prev",futurePQ[dweg-1])
        #futurePQ.append(futurePQ[dweg-1])
        #walkPath(queue = deque(futurePQ[dweg-1]))
        
        print(futurePQ)
        #if(futurePQ[dweg-1] != [(x,y)]):
         #   futurePQ.append(futurePQ[dweg-1])  
                

    return(futurePQ)


def BFS2(maze):
    start = (0, 0) 
    queue = deque([("", start)])
    visited = set()
    graph = maze2graph(maze)
    while queue:
        path, current = queue.popleft()
        x = current[0]
        y = current[1]
        old = image[x][y]
        image[x][y] = 5
        update()
        image[x,y] = old
        if current in visited:
            continue
        visited.add(current)
        image[x,y] = 1
        for direction, neighbour in graph[current]:
            queue.append((path + direction, neighbour))
    return "NO WAY!"

def findPath(tuplez,futurePQ):
    #tuplez[0] = prev
    #tuplez[1] = curr
    print(tuplez)
    print(tuplez[0])
    print(tuplez[1][0])
    px = tuplez[1][0][0]
    py = tuplez[1][0][1]
    for k in [[px,py-1],[px+1,py],[px,py+1],[px-1,py]]:
        
        if(k[0] >= 0 and k[0] < xl-1 and k[1] >= 0 and k[1] < yl-1):
            print("in border",k[0] >= 0 and k[0] < xl-1 and k[1] >= 0 and k[1] < yl-1)
            if(image[k[0]][k[1]] == 1 and [(k[0],k[1])] in futurePQ ):
                indexP = futurePQ.index([(k[0],k[1])])                            
                #futurePQ = futurePQ[:(indexP+1)] + [popedNode] + futurePQ[(indexP+1):]   

                indexc = futurePQ.index(tuplez[0])
                print("detta är index",indexP,indexc)
                return(indexP,indexc)
    print("NOTHING :C")            


def DFS2(x,y):
    stack = deque([[(x,y)]])
    visited = []
    index = 0
    prev = ([(x,y)])
    #graph = maze2graph(image)
    while stack:
        current = stack.pop()
        #print("curr",current)
        #print("reverse",reverse)
        if current in visited:
            continue        

        print(current)
        x = current[0][0]  
        y = current[0][1]
        
        print("test :", prev[0][0], " - ", x, "  ", prev[0][1], " - ", y )
        
        if(abs(prev[0][0] - x) > 1 or abs(prev[0][1] - y) > 1 or (abs(prev[0][0] - x) >= 1 and abs(prev[0][1] - y) >= 1)  ):
            print("jumping !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("from ", prev, "  to ", current)
            tuplepath = findPath([prev,current],visited)
            print("tuplepath",tuplepath[0],tuplepath[1])
            print("tuppleindex :", visited[tuplepath[0]])

            reversePath = (visited[tuplepath[0]:tuplepath[1]][::-1]) 
            #reversePath = reversePath[::-1]
            #visited += visited[:tuplepath[1]] + test + visited[tuplepath[1]:]
            #visited.insert(tuplepath[1],test)
            visited = visited[:tuplepath[1]] + reversePath + visited[tuplepath[1]:]
            reversePath = deque(reversePath)
            print(reversePath)
            walkPath(reversePath)
            
            

        prev=([(x,y)])
        #print(visited)

        old = image[x][y]
        image[x][y] = 5
        update()
        image[x,y] = old

        visited.append([(x,y)])
        image[(x,y)] = 1 

        if((x,y+1) not in visited and image[x][y+1] == 0 and y+1 < yl-1 ):
            #print("added: ",[x,y+1])
            stack.append([(x,y+1)])            
        if((x+1,y) not in visited and image[x+1][y] == 0 and x+1 < xl-1 ):
            #print("added: ",[x+1,y])

            stack.append([(x+1,y)])  
        if((x-1,y) not in visited and image[x-1][y] == 0 and x-1 >= 0 ):
            #print("added: ",[x-1,y])

            stack.append([(x-1,y)])  
        if((x,y-1) not in visited and image[x][y-1] == 0 and y-1 >= 0 ):
            #print("added: ",[x,y-1])

            stack.append([(x,y-1)])      
        index += 1

    return visited

def maze2graph(maze):
    height = len(maze)
    width = len(maze[0]) if height else 0
    graph = {(i, j): [] for j in range(width) for i in range(height) if not maze[i][j]}
    for row, col in graph.keys():
        if row < height - 1 and not maze[row + 1][col]:
            graph[(row, col)].append(("S", (row + 1, col)))
            graph[(row + 1, col)].append(("N", (row, col)))
        if col < width - 1 and not maze[row][col + 1]:
            graph[(row, col)].append(("E", (row, col + 1)))
            graph[(row, col + 1)].append(("W", (row, col)))
    return graph

def BFS(x,y):
    queue = deque( [(x,y)]) #create queue
    index = 1
    while len(queue)>0: #make sure there are nodes to check left
        node = queue.pop() #grab the first node
        x = node[0] #get x and y
        y = node[1]
        print([x,y])
        update()
        # if image[x][y] == "exit": #check if it's an exit
        #    return GetPathFromNodes(node) #if it is then return the path
        """if (image[x][y] == 0): #if it's not a path, we can't try this spot
            image[x][y] = 1
            for i in [[x+1,y+1],[x+1,y],[x+1,y-1],[x,y-1],[x-1,y-1],[x-1,y],[x-1,y+1],[x,y+1]]:
                if(image[i[0],i[1]] == 0):
                    queue.append((i[0],i[1],node))

            
            #for i in [[x-1,y+1],[x,y+1],[x+1,y+1],[x+1,y]]:
             #   queue.append((i[0],i[1],node))  

        #if x+1 < xl: 
        # for i in [[x+1,y],[x+1,y+1],[x,y+1],[x-1,y+1],[x-1,y],[x-1,y-1],[x,y-1],[x+1,y-1]]:"""
                    
        
    
        if(image[x][y] == 0):
            image[x][y]= 1 #make this spot explored so we don't try again
            update()
            if y > 0 and y < yl-1:
                y += index
                queue.append(([x,y])) 
            elif y == yl-1 and index < 0:
                y += index
                queue.append(([x,y])) 
            elif y == 0 and index > 0:
                y += index 
                queue.append([x,y])
            else :
                if x >= 0 and x < xl-1:
                    x += 1
                    index = index * (-1) 
                    queue.append([x,y])                                    
                
    return []

def readPath():
    fileMap.read()
    print(fileMap)

def makePath(x,y):
    print("check if fime exist")
    if(fileExist):
        print("map exist, taking from map...")
        readPath()
    
    print("file does not exist, making a new one")

    queue = deque([(x,y)])
    #x = x
    #y = y
    index = 1
    counter = 1
    """for xi in range(xl):
        for yi in range(yl):
            queue.append([xi,yi])"""

    #for i in pqueue:
     #   print(i)

    #print(pqueue)"""
    while(counter < (xl*yl)):
        counter += 1
        """if(image[x][y] == 0):
            image[x][y]= 1 #make this spot explored so we don't try again
            update()"""
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
   
def walkPath(queue):
    print("this is queue", queue)
    xold = 0
    yold = 0
    while len(queue) > 0:
        #str = ser.readline().decode('utf-8').split(",")
        #print(str)
        
        #print(queue)
        node = queue.popleft()
        print("this is noode", node)
        try:
            x = node[0]
            y = node[1]
        except IndexError:
            x = node[0][0]
            y = node[0][1]

        #if(image[x,y] == 0):dwadwad:
       
        old = image[x][y]
        image[x][y] = 5
        update()
        image[x,y] = old     
        """
        if(xold - x) > 0:
            print("down")
            #drive(1) #down
        elif(xold - x) < 0:
            print("up")
            #ser.write(2) #up
            #drive(2)   
        elif(yold - y) > 0:
            print("left")
            #ser.write(3) #left   
            #drive(3)
        elif(yold - x) < 0:
            print("right")
            #ser.write(4) #right           
            #drive(4)
        """
        objectAvoidz(x,y,xold - x, yold-y, queue)
        if(image[x,y] != 2):
            image[x,y] = 1  
        xold = x
        yold = y
        
        #print(image(node))

def objectAvoid(x,y,queue):
    if(image[x][y] == 2):
        image[x][y] = 3
        print("inne")
        for i in [[x+1,y+1],[x+1,y],[x+1,y-1],[x,y-1],[x-1,y-1],[x-1,y],[x-1,y+1],[x,y+1]]:
            if(i[1] > 0 and i[1] < xl-1 and i[0] > 0 and i[0] < yl-1):
                objectAvoid(i[0],i[1],queue)
                if (image[i[0]][i[1]] != 2 and image[i[0]][i[1]] != 3):
                    image[i[0]][i[1]] = 6
                    if not((i[0],i[1]) in queue) :
                        queue.appendleft((i[0],i[1]))

def objectAvoidz(x,y,xo,yo,queue):
    l = []
    print("avoid",x,y)
    print("is object?", image[x][y],image[x][y] == 2)
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
                #print("check")
                #if (image[i[0]][i[1]] != 2): 
                 #   image[i[0]][i[1]] = 0
                #if (i[0],i[1]) not in queue :
                    #print("already in queue")
                print("adding", l)
                    #queue.clear()
                    #floodfill(x,y,"start")
                #objectAvoidz(x,y,x-i[0],y-l[1],queue)
                if(image[i[0]][i[1]] != 2):
                    queue.appendleft((i[0],i[1]))             
                else:
                    objectAvoidz(i[0],i[1],x-i[0],y-i[1],queue)

"""
def drive(instr): 
    ser.write(instr)
    while(True):
        if(ser.readline().decode('utf-8').split(",") == "ok"):
            continue
            
"""

def learnQueue(futurePQ):
    learnedQueue = []
    #print(futurePQ)
    for i in range(0,len(futurePQ)):
        #print(futurePQ[i]) 
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
                 
            


            #print("this is from i: ",x,y)
            #print("this is from i+1: ",(futurePQ[i+1][0][0]),(futurePQ[i+1][0][1])," - ")
            #print('')
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


            
            if (abs(x-xn >= 2) and abs(y-yn >= 2)):
                print("JUMPING!")
                #print((futurePQ[i+1][0][0]),((futurePQ[i+1][0][1])))
                #print(futurePQ.index([((futurePQ[i+1][0][0]),((futurePQ[i+1][0][1])))]))
                #print(i+1)
                #print(futurePQ.index([(0, 0)]))
                #indexPop = futurePQ.index([((futurePQ[i+1][0][0]),(futurePQ[i+1][0][1]))])
                popedNode = futurePQ.pop(i+1)
                print ("Jumping from: ",futurePQ[i],popedNode)
                px = popedNode[0][0]
                py = popedNode[0][1]
                #indexP = futurePQ.index([(popedNode[0][0],popedNode[0][1])])
                #indexP = futurePQ.index([(popedNode[0][0],popedNode[0][1]-1)])
            
                for k in [[px,py-1],[px+1,py],[px,py+1],[px-1,py]]:
                    if(k[0] >= 0 and k[0] < xl-1 and k[1] >= 0 and k[1] < yl-1):
                        #print("this is image",image)
                        #print("this is k",k)
                        #print("this is k in fq", ([(k[0],k[1])] in futurePQ))
                        #print("image of k",image[k[0]][k[1]])
                        if(image[k[0]][k[1]] == 1 and [(k[0],k[1])] in futurePQ ):
                            indexP = futurePQ.index([(k[0],k[1])])
                            #print("yesyes adding this",indexP,k,futurePQ[indexP],futurePQ[indexP-1:indexP+2])
                            
                            futurePQ = futurePQ[:(indexP+1)] + [popedNode] + futurePQ[(indexP+1):] 
                            #print("yesyes adding this2",indexP,k,futurePQ[indexP],futurePQ[indexP-1:indexP+3])

                            break
                                          

            else:
                learnedQueue.append(futurePQ[i])    
    return(futurePQ)


def saveMap():
    #print('['+(''.join(str(e) for e in image))+']')
    fileMap.write('['+(''.join(str(e) for e in image))+']')

def saveQueue(futurePQ):
    futurePQ = learnQueue(futurePQ) 
    
    string = str(futurePQ)
    #print(string)
    fileMap.write(string)  
    

def getQueue():
    queue = deque([(0,0)])
    
    for i in fileMap.read().strip('[[').strip(']]').split('], ['):
        i = i[1:]
        #print("this is quee")
        i = (i.strip('(').strip(')').strip('\n').strip(']]').strip(')').strip('').split(','))
        i = list(map(int, i))
        #i = [int(j) for j in i]
        #print(i)

        queue.append(i)
    
    walkPath(queue)
    

def main():
    #x,y = 25
    global xl   
    global yl
    #xl,yl= [int(x) for x in raw_input("X,Y : ").split(',')] 
    xl = 10  
    yl = 10
    initialize(xl,yl)
    #time.sleep(1)
    #print(image[xl-1,yl-1])
    #print(image[3,4])
    #futurePQ = floodfill(0,0,"begin")
    #print(futurePQ)
    #21.2s
    #20,3s
    #0.67s
    #print(raw_input(','))
    
    #DFS2(18,18)
    #walkPath(makePath(4,8))
    
    #learnQueue()
    if(fileExist):
        print("get queue")
        getQueue()   
    else:
        print("dfs")
        futurePQ = DFS2(4,8)#floodfill(0,0,"begin",0)
        #print("this is fpq",futurePQ)
        saveQueue(futurePQ)
    
    #print(image)

        
    #print(x)
    #print(y)
    
    #real0m21.461s
    #user0m20.688s
    #sys0m0.659s

#real	0m9.993s
#user	0m9.072s

    #print(image[xl-1,yl-1])
    #update()
    #saveMap()
    #saveQueue(futurePQ)
    #time.sleep(10)
    #update(image)
    #time.sleep(1)

main()


