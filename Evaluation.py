from scipy import spatial
import heapq

def get_nearestn_of_original_vspace(sense_matrix, gensimmodel, threshold_distance, write_to_file, filename):
    distances = []
    if write_to_file:
        f = open(filename, "w")
    for idx in range(len(sense_matrix)):
        sense1 = sense_matrix[idx][0]
        sense2 = sense_matrix[idx][1]

        word = gensimmodel.wv.index2word[idx]
        distance = spatial.distance.cosine(sense1, sense2)

        if distance > threshold_distance:
            if write_to_file:
                f.write("w :" + word + "\n")
                f.write(str(gensimmodel.similar_by_vector(sense1)))
                f.write("\n")
                f.write(str(gensimmodel.similar_by_vector(sense2)))
                f.write("\n")

            heapq.heappush(distances, (distance, word))
    if write_to_file:
        f.close()
    return sorted(distances[len(distances)-10:])

def write_sensevectors_to_gensimfile(sense_matrix, filename, gensimmodel):
    f = open(filename, "w")
    vocabsize = sense_matrix.shape[0]*2
    embeddingdim = sense_matrix.shape[2]
    f.write(str(vocabsize) + " " + str(embeddingdim)+ "\n")
    for idx in range(len(sense_matrix)):
        sense1 = sense_matrix[idx][0]
        sense2 = sense_matrix[idx][1]
        original_word = gensimmodel.wv.index2word[idx]
        s1 = original_word + "_1 "
        s2 = original_word + "_2 "
        f.write(s1)
        for el in sense1:
            f.write(str(el))
            f.write(" ")
        f.write("\n")
        f.write(s2)
        for el in sense2:
            f.write(str(el))
            f.write(" ")
        f.write("\n")
    f.close()



def get_nearest_neighbours_of_new_vspace(sense_matrix, gensimmodel, sensimmodel, threshold_distance, write_to_file, filename):
    distances = []
    if write_to_file:
        f = open(filename, "w")
    for idx in range(len(sense_matrix)):
        sense1 = sense_matrix[idx][0]
        sense2 = sense_matrix[idx][1]
        word = gensimmodel.wv.index2word[idx]
        distance = spatial.distance.cosine(sense1, sense2)
        if distance > threshold_distance:
            if write_to_file:
                f.write("w :" + word + "\n")
                f.write(str(sensimmodel.similar_by_vector(sense1)))
                f.write("\n")
                f.write(str(sensimmodel.similar_by_vector(sense2)))
                f.write("\n")
            heapq.heappush(distances, (distance, word))

    if write_to_file:
        f.close()
    print(sorted(distances[len(distances)-10:]))
    return sorted(distances[len(distances)-10:])





