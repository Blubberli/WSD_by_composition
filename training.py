import argparse
import tensorflow as tf
from keras.utils import generic_utils
import os
from pathlib import Path
import time
import logging
from MaskModel import MaskModel

import utils

from holst import Data,  logger_config
from TrainingGraph import TrainingGraph
from TrainingGraph import RunMode
from identity_model import Identity_Model

def train(args, wsdmodel, training_data, validation_data, gensimmodel):
    train_losses = []
    validation_losses = []

    td = training_data
    vd = validation_data

    assert (len(td.modifier_batches) == len(td.head_batches) == len(
        td.compound_batches)), "error: inconsistent training batches"
    assert (len(vd.modifier_batches) == len(vd.head_batches) == len(
        vd.compound_batches)), "error: inconsistent validation batches"
    assert (td.no_batches != 0), "error: no training data"
    assert (vd.no_batches != 0), "error: no validation data"

    lowest_loss = float("inf")
    best_epoch = 0
    epoch = 1
    current_patience = 0
    tolerance = 1e-5

    # write information for tensorboard
    with tf.Session(config=args.config) as sess:

        with tf.variable_scope("model", reuse=None):
            wsdmodel.create_architecture(batch_size=args.batch_size, lookup=lookup_table)
            train_model = TrainingGraph(wsd_model=wsdmodel,
                                        batch_size=args.batch_size,
                                        lookup_table=lookup_table,
                                        learning_rate=args.learning_rate,
                                        run_mode=RunMode.training,
                                        alpha=0.3)
        with tf.variable_scope("model", reuse=True):
            validation_model = TrainingGraph(wsd_model=wsdmodel,
                                             batch_size=args.batch_size,
                                             lookup_table=lookup_table,
                                             learning_rate=args.learning_rate,
                                             run_mode=RunMode.validation,
                                             alpha=0.3)

        # init all variables
        sess.run(tf.global_variables_initializer())
        print("graphs are created")
        saver = tf.train.Saver(max_to_keep=0)
        mod_embeddings = open("./modified.txt", "w")
        while  epoch < 200:
            train_loss = 0.0
            validation_loss = 0.0

            for tidx in range(td.no_batches-1):
                assert (td.modifier_batches[tidx].shape
                        == td.head_batches[tidx].shape
                        == td.compound_batches[tidx].shape), "error: each batch has to have the same shape"
                assert (td.modifier_batches[tidx].shape != ()), "error: funny shaped batch"

                pb = generic_utils.Progbar(td.no_batches)
                # only executed if the user wants to use tensorboard
                # calculate loss for each batch for each epoch
                tloss, modified_input, _ = sess.run(
                    [train_model.loss, train_model.model.specialized_modifier, train_model.train_op],
                    feed_dict={train_model.is_training: True,
                               train_model.original_vector: td.compound_batches[tidx],
                               train_model.model._u: td.modifier_batches[tidx],
                               train_model.model._v: td.head_batches[tidx]})

                train_loss += tloss
                if epoch > 195:
                    for el in range(len(td.modifier_batches[tidx])):
                        print("k nearest original")
                        print(gensimmodel.similar_by_vector(gensimmodel.syn0[td.modifier_batches[tidx][el]]))
                        print("k nearest modified")
                        print(gensimmodel.similar_by_vector(modified_input[el]))

                pb.update(tidx + 1)

            for vidx in range(vd.no_batches-1):
                pb = generic_utils.Progbar(vd.no_batches)
                vloss,_ = sess.run(
                    [validation_model.loss, validation_model.train_op],
                    feed_dict={validation_model.original_vector: vd.compound_batches[vidx],
                               validation_model.model._u: vd.modifier_batches[vidx],
                               validation_model.model._v: vd.head_batches[vidx]})

                validation_loss += vloss
                pb.update(vidx + 1)

            train_loss /= td.total_size
            validation_loss /= vd.total_size

            if (lowest_loss - validation_loss > tolerance):
                lowest_loss = validation_loss
                best_epoch = epoch
                saver.save(sess, args.model_path)
                current_patience = 0
            else:
                current_patience += 1

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            logger.info("(%d) epoch %d - train loss: %.5f validation loss: %.5f" % (
            current_patience, epoch, train_loss, validation_loss))
            epoch += 1
        for el in modified_input:

            mod_embeddings.write(str(el))
    return (train_losses, validation_losses, lowest_loss, best_epoch, saver)



if __name__ == '__main__':
    # define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", type=str,
                        help="path to the file that contains word embeddings, format: .bin/.txt")
    parser.add_argument("data_dir", type=str, help="path to the directory that contains the train/test/dev data")
    parser.add_argument("--unknown_phrase_key", type=str,
                        help="string corresponding to the unknown phrase embedding in the embedding file",
                        default="<unk>")
    parser.add_argument("--unknown_w1_key", type=str,
                        help="string corresponding to the unknown word 1 embedding in the embedding file",
                        default="<unk_w1>")
    parser.add_argument("--unknown_w2_key", type=str,
                        help="string corresponding to the unknown word 2 embedding in the embedding file",
                        default="<unk_w2>")
    parser.add_argument("--batch_size", type=int, help="how many instances should be contained in one batch?",
                        default=100)
    parser.add_argument("--patience", type=int, help="number of epochs to wait after the best model", default=5)
    parser.add_argument("--learning_rate", type=float, help="learning rate for optimization", default=0.01)
    parser.add_argument("--seed", type=int, help="number to which random seed is set", default=1)
    parser.add_argument("--save_path", type=str, help="file path to save the best model", default="./models")
    parser.add_argument("--vocab_size", type=int, help="number of words to include in the fulllex model", default=25619,
                        choices=[7132, 25619])
    parser.add_argument("--nonlinearity", type=str,
                        help="what kind of nonlinear function should be applied to the model. set to 'identity' if no nonlinearity should be applied",
                        default='tanh', choices=["tanh", "identity"])
    args = parser.parse_args()

    # log cpu/gpu info, prevent allocating so much memory
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    args.config = config

    args.training_data = str(Path(args.data_dir).joinpath("train_text.txt"))
    args.validation_data = str(Path(args.data_dir).joinpath("dev_text.txt"))
    args.test_data = str(Path(args.data_dir).joinpath("test_text.txt"))

    ts = time.gmtime()
    args.save_name = format("%s_%s" % ("identity", time.strftime("%Y-%m-%d-%H_%M_%S", ts)))
    args.model_path = str(Path(args.save_path).joinpath(args.save_name))

    # setup logging
    args.log_file = str(Path(args.save_path).joinpath(args.save_name + "_log.txt"))
    logging.config.dictConfig(logger_config.create_config(args.log_file))
    logger = logging.getLogger("train")

    logger.info("Training %s composition model. Logging to %s" % ("identity", args.log_file))
    logger.info("Arguments")
    for k, v in vars(args).items():
        logger.info("%s: %s" % (k, v))

    nonlinear_functions = {
        "tanh": tf.nn.tanh,
        "identity": tf.identity
    }

    # read in the wordembeddings
    gensim_model = Data.read_word_embeddings(args.embeddings)
    logger.info("Read embeddings from %s." % args.embeddings)

    # generate batches from data
    word2index = gensim_model.wv.vocab
    training_data = Data.generate_instances(args.batch_size, args.training_data, word2index,
                                            args.unknown_phrase_key, args.unknown_w1_key, args.unknown_w2_key)
    logger.info("%d training batches" % training_data.no_batches)
    validation_data = Data.generate_instances(args.batch_size, args.validation_data, word2index,
                                              args.unknown_phrase_key, args.unknown_w1_key, args.unknown_w2_key)
    logger.info("%d validation batches" % validation_data.no_batches)
    test_data = Data.generate_instances(args.batch_size, args.test_data, word2index,
                                        args.unknown_phrase_key, args.unknown_w1_key, args.unknown_w2_key)
    logger.info("%d test batches" % test_data.no_batches)
    lookup_table = gensim_model.wv.syn0
    logger.info("Batches have been generated using the data from %s" % args.data_dir)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    tf.set_random_seed(args.seed)

    saver = None
    wsdmodel = Identity_Model(vocab_size=args.vocab_size, no_senses=4)
    train_losses, validation_losses, best_loss, best_epoch, saver = train(args, wsdmodel, training_data, validation_data, gensim_model)
    logger.info("Training ended. Best epoch: %d, best loss: %.5f" % (best_epoch, best_loss))

    #dev_predictions, dev_loss = predict(args, saver, args.model_path, validation_data, composition_model, lookup_table)
    #logger.info("dev loss %.5f" % dev_loss)
