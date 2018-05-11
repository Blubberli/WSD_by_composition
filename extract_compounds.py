from holst import Data
import char_split
import random



def get_splits():
    data = open("/Users/neelewitte/Desktop/newcompounds_germante.txt", "r")
    train = open("/Users/neelewitte/Desktop/newcompounds_germante_train.txt", "w")
    test = open("/Users/neelewitte/Desktop/newcompounds_germante_test.txt", "w")
    dev = open("/Users/neelewitte/Desktop/newcompounds_germante_dev.txt", "w")

    lines = data.readlines()
    print(len(lines))
    print(lines)

    traindata = random.sample(lines, 45686)
    for el in traindata:
        train.write(el)
    restdata = []
    print(len(traindata))
    for el in lines:
        if el not in traindata:
            restdata.append(el)
    print(len(restdata))
    devdata = random.sample(restdata, 6527)
    for el in devdata:
        dev.write(el)
    newrestdata = []
    for el in restdata:
        if el not in devdata:
            newrestdata.append(el)
    for el in newrestdata:
        test.write(el)
    print("training")
    print(len(traindata))
    print("dev")
    print(len(devdata))
    print("test")
    print(len(newrestdata))

    train.close()
    dev.close()
    data.close()
    test.close()





def extract_compounds():
    compoundfile = open("/Users/neelewitte/Desktop/progamming/split_compounds_from_GermaNet13.0.txt")
    #glovefile = Data.read_word_embeddings("/Users/neelewitte/Desktop/progamming/glove/twe-adj-n.bin")
    decow = Data.read_word_embeddings("/Users/neelewitte/Desktop/progamming/decow14ax_all_min_100_vectors_l2_rows_200dim.txt")
    data = open("/Users/neelewitte/Desktop/newcompounds_germante.txt", "w")

    dev = open("/Users/neelewitte/Desktop/newcompounds_germante_dev.txt", "w")
    mixedcompoundstrain = open("/Users/neelewitte/Desktop/progamming/compounds_mixed/train_text.txt", "r")
    mixedcompoundstest = open("/Users/neelewitte/Desktop/progamming/compounds_mixed/test_text.txt", "r")
    mixedcompoundsdev = open("/Users/neelewitte/Desktop/progamming/compounds_mixed/dev_text.txt", "r")
    mixedtrain = mixedcompoundstrain.readlines()
    mixeddev = mixedcompoundsdev.readlines()
    mixedtest = mixedcompoundstest.readlines()

    compounds = []
    newcompounds = []
    print(len(decow.wv.vocab))
    for line in compoundfile:
        compound = line.split("\t")[0].strip()
        mod = line.split("\t")[1].strip()
        head = line.split("\t")[2].strip()
        if compound.lower() in decow.wv.vocab:
            print(compound)
            if mod.lower() in decow.wv.vocab:
                if head.lower() in decow.wv.vocab:
                    compounds.append(compound)
                    data.write(mod.lower() + " " + head.lower() + " " + compound.lower() + "\n")
        else:
            newcompounds.append(compound.lower())
    nverseencompounds = []
    for line in mixedtrain:
        compound = line.split(" ")[2].strip()
        if compound not in newcompounds:
            nverseencompounds.append(compound)
    for line in mixedtest:
        compound = line.split(" ")[2].strip()
        if compound not in newcompounds:
            nverseencompounds.append(compound)
    for line in mixeddev:
        compound = line.split(" ")[2].strip()
        if compound not in newcompounds:
            nverseencompounds.append(compound)

    data.close()
    print(compounds)
    print(len(compounds))
    print(len(newcompounds))
    print(len(nverseencompounds))

def modify_embeddings():
    decow = Data.read_word_embeddings("/Users/neelewitte/Desktop/progamming/decow14ax_all_min_100_vectors_l2_rows_200dim.txt")
    train = open("/Users/neelewitte/Desktop/newcompounds_germante_train.txt", "r")
    test = open("/Users/neelewitte/Desktop/newcompounds_germante_test.txt", "r")
    dev = open("/Users/neelewitte/Desktop/newcompounds_germante_dev.txt", "r")
    embeddings = open("/Users/neelewitte/Desktop/progamming/decow_embeddings/new_compounds/embeddings.txt", "w")
    modifier = []
    heads = []
    compounds = []
    for line in train.readlines():
        mod = line.split(" ")[0].strip()
        head = line.split(" ")[1].strip()
        compound = line.split(" ")[2].strip()
        modifier.append(mod)
        heads.append(head)
        compounds.append(compound)
    for line in test.readlines():
        mod = line.split(" ")[0].strip()
        head = line.split(" ")[1].strip()
        compound = line.split(" ")[2].strip()
        modifier.append(mod)
        heads.append(head)
        compounds.append(compound)
    for line in dev.readlines():
        mod = line.split(" ")[0].strip()
        head = line.split(" ")[1].strip()
        compound = line.split(" ")[2].strip()
        modifier.append(mod)
        heads.append(head)
        compounds.append(compound)
    modifier = set(modifier)
    heads = set(heads)
    compounds = set(compounds)
    unk = decow.wv["unknown"]
    embeddings.write("unknown ")
    for number in unk:
        embeddings.write(str(number))
        embeddings.write(" ")
    embeddings.write("\n")
    for el in modifier:
        vec = decow.wv[el]
        embeddings.write(el + " ")
        for number in vec:
            embeddings.write(str(number))
            embeddings.write(" ")
        embeddings.write("\n")
    for el in heads:
        vec = decow.wv[el]
        embeddings.write(el + " ")
        for number in vec:
            embeddings.write(str(number))
            embeddings.write(" ")
        embeddings.write("\n")
    for el in compounds:
        vec = decow.wv[el]
        embeddings.write(el + " ")
        for number in vec:
            embeddings.write(str(number))
            embeddings.write(" ")
        embeddings.write("\n")

    print(len(modifier))
    print(len(heads))
    print(len(compounds))
    embeddings.close()

if __name__ == '__main__':
    #extract_compounds()
    #get_splits()
    modify_embeddings()