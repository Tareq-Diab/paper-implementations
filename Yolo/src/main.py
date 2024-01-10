

def parse_cfg(config_file):
    file = open(config_file,'r')
    file = file.read().split('\n')
    file =  [line for line in file if len(line)>0 and line[0] != '#']
    file = [line.lstrip().rstrip() for line in file]

    final_list = []
    element_dict = {}
    for line in file:

        if line[0] == '[':
            if len(element_dict) != 0:     # appending the dict stored on previous iteration
                    final_list.append(element_dict)
                    element_dict = {} # again emtying dict
            element_dict['type'] = ''.join([i for i in line if i != '[' and i != ']'])
            
        else:
            val = line.split('=')
            element_dict[val[0].rstrip()] = val[1].lstrip()  #removing spaces on left and right side
        
    final_list.append(element_dict) # appending the values stored for last set
    return final_list

x=parse_cfg("yolov3.cfg")
with open("yolov3_parsed.cfg","w") as file:
    for i in x:
        file.write(str(i)+"\n")
        
file.close()
        