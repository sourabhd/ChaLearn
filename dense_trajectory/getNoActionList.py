

def getNoActionList(sortedActionList):

#sortedActionList = [[1, 63, 86], [2, 95, 118], [2, 430, 467], [3, 131, 157], [4, 228, 274], [5, 186, 195], [5, 196, 203], [5, 204, 213], [6, 1, 15], [6, 355, 381], [6, 397, 425], [6, 441, 480], [6, 494, 546], [6, 620, 661], [6, 674, 696], [6, 870, 889], [6, 870, 895], [7, 560, 589], [8, 688, 729], [8, 689, 729], [9, 789, 828], [9, 790, 822], [10, 830, 871], [10, 837, 871], [11, 730, 776], [11, 735, 776]]


    noActionList = [[min(sortedActionList, key=lambda x: x[1])[1], max(sortedActionList, key=lambda x: x[2])[2]]]

    for sal in sortedActionList:
        for nal in noActionList:
            if sal[1] <= nal[0] and sal[2] >= nal[1]:   # exact match or bigger : remove
                noActionList.remove(nal)
            elif sal[1] >= nal[0] and sal[1] <= nal[1] and sal[2] >= nal[0] and sal[2] <= nal[1]:   # inside : chop
                noActionList.remove(nal)
                if nal[0] < sal[1]-1:
                    noActionList.append([nal[0],sal[1]-1])
                if sal[2]+1 < nal[1]:
                    noActionList.append([sal[2]+1,nal[1]]) 
            elif sal[1] <= nal[0] and sal[2] >= nal[0] and sal[2] <= nal[1]:   # left overlap : chop
                noActionList.remove(nal)
                if sal[2]+1 < nal[1]:
                    noActionList.append([sal[2]+1,nal[1]]) 
            elif sal[1] >= nal[0] and sal[1] <= nal[1] and sal[2] >= nal[1]:   # right overlap : chop
                noActionList.remove(nal)
                if nal[0] < sal[1]-1:
                    noActionList.append([nal[0],sal[1]-1])
            elif sal[2] <= nal[0] or sal[1] >= nal[1]:  #no overlap
                pass    
            else:
                print 'else reached'
                print [sal[1], sal[2]], [nal[0],nal[1]]
                
    noActionList.sort()  

    print
    print 'Sorted Action List: ', sorted(sortedActionList,key=lambda x: [x[1],x[2]]) 
    print
    print 'No Action List: ', noActionList        
    return noActionList
