from collections import deque

#Dec##########################
ROWS = 4
COLS = 4

START = (1,1)
GOAL  = (4,4)
WALLS = {(2,2)}
FIRES = {(3,3),(4,2)}

ACTIONS ={ "UP": (0,1),"DOWN": (0,-1), "RIGHT": (1,0), "LEFT": (-1,0)}
################################

def is_valid(x,y):
    
    if(x < 1 or x > COLS or y < 1 or y > ROWS):
        return False
    if(x,y) in WALLS:
        return False
    if(x,y) in FIRES:
        return False
        
    return True
    
def bfs_shortest_path():
    qeque = deque();
    qeque.append({"cell": START, "path": [START],"actions": []})
    
    visitedNode = set()
    visitedNode.add(START)
    
    while qeque:
        current = qeque.popleft()
        
        x ,y = current["cell"]
        
        if(x ,y) == GOAL:
            return current['path'],current['actions']
        
        for action_names ,move in ACTIONS.items():
            dx ,dy = move
            new_x = dx + x
            new_y = dy + y
            nextcell = (new_x, new_y)
            
            if(is_valid(new_x,new_y) and nextcell not in visitedNode):
                visitedNode.add(nextcell)
                
                qeque.append({"cell": nextcell, 
                "path": current["path"] + [nextcell],
                "actions": current["actions"]+[action_names]})
                
    return None , None
    
    
path ,actions = bfs_shortest_path()
if path is None:
    print("no safe path found.")
    
else:
    print("shorted safe path found:\n")
    print("path:")
    for cell in path:
        print(cell)
    print("\n Action seq:")
    print("->".join(actions))
    print("\nTotal move",len(actions))
