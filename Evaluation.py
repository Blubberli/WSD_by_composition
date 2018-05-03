from scipy import spatial
import heapq

def calculate_cosinedistance(sense_matrix, gensimmodel):
    distances = []
    #f = open("/home/neele/nn_amibuge_threeSenses.txt", "w")
    for idx in range(len(sense_matrix)):
        sense1 = sense_matrix[idx][0]
        sense2 = sense_matrix[idx][1]
        #sense3 = sense_matrix[idx][2]
        word = gensimmodel.wv.index2word[idx]
        distance1 = spatial.distance.cosine(sense1, sense2)
        #distance2 = spatial.distance.cosine(sense1, sense3)

        if distance1 >  0.2:
            #f.write(word + "\n")
            print(word)
            print(gensimmodel.similar_by_vector(sense1))
            print(gensimmodel.similar_by_vector(sense2))
            #print(gensimmodel.similar_by_vector(sense3))
            '''
            f.write(str(gensimmodel.similar_by_vector(sense1)))
            f.write("\n")
            f.write(str(gensimmodel.similar_by_vector(sense2)))
            f.write("\n")
            f.write(str(gensimmodel.similar_by_vector(sense3)))
            f.write("\n")
            f.write("\n")'''
        heapq.heappush(distances, (distance1, word))
    #f.close()
    return sorted(distances)

def get_nearest_neighbours(sense_matrix, gensimmodel):
    for idx in range(len(sense_matrix)):
        sense1 = sense_matrix[idx][0]
        sense2 = sense_matrix[idx][1]
        word = gensimmodel.wv.index2word[idx]
        print("word")
        print(word)
        print(gensimmodel.similar_by_vector(sense1))
        print(gensimmodel.similar_by_vector(sense2))



    '''
    if (distance > 0.2):
        f.write(word + "\n")
        print(word)
        print(gensimmodel.similar_by_vector(sense1))
        print(gensimmodel.similar_by_vector(sense2))
        f.write(str(gensimmodel.similar_by_vector(sense1)))
        f.write("\n")
        f.write(str(gensimmodel.similar_by_vector(sense2)))
        f.write("\n")
        f.write("\n")'''