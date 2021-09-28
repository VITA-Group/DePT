import json
import numpy as np

def get_intersection_positions(file):
    ''' get the absolute position of each intersection
    input
        file: road file
    return
        inter_map: a dict with keys as original intersection id (e.g., intersection_0_1) and values as the mapped intersection id (e.g., 0)
        inter_pos: a list where inter_pos[i] is the position [x, y] of intersection i
    '''
    with open(file, 'r') as jsonfile:
        json_string = json.load(jsonfile)
    # print(json_string['intersections'][0])
    inter_map = {}
    inter_pos = []
    inter_inner_map = {}
    inter_inner_pos = []
    inner_count = 0
    for i in range(len(json_string['intersections'])):
        inter_map[json_string['intersections'][i]['id']] = i
        inter_pos.append([json_string['intersections'][i]['point']['x'], json_string['intersections'][i]['point']['y']])
        if len(json_string['intersections'][i]['roads']) > 2:
            inter_inner_map[json_string['intersections'][i]['id']] = inner_count
            inter_inner_pos.append([json_string['intersections'][i]['point']['x'], json_string['intersections'][i]['point']['y']])
            inner_count += 1
    return inter_map, inter_pos, inter_inner_map, inter_inner_pos

def getID2Pos(roadnet_file, dic_traffic_env_conf):
    list_intersection = ["intersection_{0}_{1}".format(i+1, j+1)
                              for i in range(dic_traffic_env_conf["NUM_ROW"])
                              for j in range(dic_traffic_env_conf["NUM_COL"])]
    # list_intersection = ["intersection_{0}_{1}".format(i+1, j+1)
    #                           for i in range(3)
    #                           for j in range(4)]
    inter_map, inter_pos, inter_inner_map, inter_inner_pos = get_intersection_positions(roadnet_file)
    inter_inner_pos_new = []
    for inter in list_intersection:
        inter_inner_pos_new.append(inter_inner_pos[inter_inner_map[inter]])

    inter_inner_pos_new = np.array(inter_inner_pos_new)
    # np.save(roadnet_file+'.npy', inter_inner_pos_new)
    # print(inter_inner_pos_new)
    return inter_inner_pos_new


if __name__ == "__main__":
    file = 'data/NewYork/16_3/roadnet_16_3.json'
    getID2Pos(file)

    # file = 'data/NewYork/28_7/roadnet_28_7.json'
    # getID2Pos(file)
    #
    # file = 'data/template_lsr/6_6/roadnet_6_6.json'
    # getID2Pos(file)
    #
    # file = 'data/template_lsr/10_10/roadnet_10_10.json'
    # getID2Pos(file)

    # then, manually change file name to: ID2Pos.npy
