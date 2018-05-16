import argparse
import tensorflow as tf
from keras.utils import generic_utils
import os
from pathlib import Path
import time
import logging
import numpy as np

from holst import Data,  logger_config, evaluation
from TrainingGraph import TrainingGraph
from TrainingGraph import RunMode
from identity_model import Identity_Model
from identity_model_head import Identity_Model_Head
from Identity_Model_bilinear_head import Identity_Model_Head_bilinear
from Identity_Model_bilinear import Identity_Model_Mod_bilinear
from identity_bilinear_both_positions import Identity_Model_bilineaer_both_positions
import Evaluation

def train(args, wsdmodel, training_data, validation_data, gensimmodel, both_positions):
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
            wsdmodel.create_architecture(batch_size=None, lookup=lookup_table)
            train_model = TrainingGraph(wsd_model=wsdmodel,
                                        batch_size=None,
                                        lookup_table=lookup_table,
                                        learning_rate=args.learning_rate,
                                        run_mode=RunMode.training,
                                        alpha=0.0,
                                        both_positions=both_positions)
        with tf.variable_scope("model", reuse=True):
            validation_model = TrainingGraph(wsd_model=wsdmodel,
                                             batch_size=None,
                                             lookup_table=lookup_table,
                                             learning_rate=args.learning_rate,
                                             run_mode=RunMode.validation,
                                             alpha=0.0,
                                             both_positions=both_positions)

        # init all variables
        sess.run(tf.global_variables_initializer())
        print("graphs are created")
        saver = tf.train.Saver(max_to_keep=0)
        while current_patience < args.patience:
        #while epoch < 2:

            train_loss = 0.0
            validation_loss = 0.0

            for tidx in range(td.no_batches):
                assert (td.modifier_batches[tidx].shape
                        == td.head_batches[tidx].shape
                        == td.compound_batches[tidx].shape), "error: each batch has to have the same shape"
                assert (td.modifier_batches[tidx].shape != ()), "error: funny shaped batch"

                pb = generic_utils.Progbar(td.no_batches)
                # only executed if the user wants to use tensorboard
                # calculate loss for each batch for each epoch
                tloss, sense_matrix, _ = sess.run(
                    [train_model.loss, train_model.model.senseMatrix, train_model.train_op],
                    feed_dict={train_model.is_training: True,
                               train_model.original_vector: td.compound_batches[tidx],
                               train_model.model._u: td.modifier_batches[tidx],
                               train_model.model._v: td.head_batches[tidx]})

                train_loss += tloss


                pb.update(tidx + 1)

            for vidx in range(vd.no_batches):
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
    return (train_losses, validation_losses, lowest_loss, best_epoch, saver, sense_matrix)


def predict(args, saver, model_file, data, wsd_model, lookup_table, both_positions):
    predictions = []

    logger.info("Loading best model from %s" % model_file)
    with tf.Session(config=args.config) as sess:
        with tf.variable_scope("model", reuse=True):
            wsd_model.create_architecture(batch_size=None, lookup=lookup_table)
            best_model = TrainingGraph(wsd_model=wsd_model,
                                            batch_size=None,
                                            lookup_table=lookup_table,
                                            learning_rate=args.learning_rate,
                                            run_mode=RunMode.validation,
                                            alpha=0.7,
                                       both_positions=both_positions)
            if saver != None:
                saver.restore(sess, model_file)

            logger.info("Generating predictions...")
            loss = 0
            for idx in range(data.no_batches):
                pb = generic_utils.Progbar(data.no_batches)
                batch_predictions, batch_loss = sess.run(
                    [best_model.predictions, best_model.loss],
                    feed_dict={best_model.original_vector: data.compound_batches[idx],
                               best_model.model._u: data.modifier_batches[idx],
                               best_model.model._v: data.head_batches[idx]})
                predictions.extend(batch_predictions)
                loss += batch_loss
                pb.update(idx + 1)
            loss /= data.total_size

    logger.info("Predictions generated.")
    return np.vstack(predictions), loss

def do_eval(logger, args, split, loss, word_embeddings):
    logger.info("%s loss %.5f" % (split, loss))
    predictions_file = str(Path(args.save_path).joinpath(args.save_name + "_%s_predictions.txt" % split))
    ranks_file = str(Path(args.save_path).joinpath(args.save_name + "_%s_ranks.txt" % split))
    ranks = evaluation.get_all_ranks(predictions_file=predictions_file, word_embeddings=word_embeddings,
        max_rank=1000, batch_size=100, path_to_ranks=ranks_file)
    logger.info("%s quartiles" % split)
    logger.info(evaluation.calculate_quartiles(ranks))


def save_predictions(predictions, data, output_file):
    out = open(output_file, mode="w", encoding="utf8")
    format_str = "%s " + "%.7f " * (predictions.shape[1] - 1) + "%.7f\n"

    for i in range(predictions.shape[0]):
        tup = (data.text_compounds[i],) + tuple(predictions[i])
        out.write(format_str % tup)

    logger.info("Predictions saved to %s" % output_file)


    out.close()

if __name__ == '__main__':
    # define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", type=str,
                        help="path to the file that contains word embeddings, format: .bin/.txt")
    parser.add_argument("data_dir", type=str, help="path to the directory that contains the train/test/dev data")
    parser.add_argument("--all_embeddings", type=str, help="path to the file that contains the embeddings with all vocabulary")
    parser.add_argument("--new_vspace", type=str, help="path to the file that will contain the sense specific embeddings")
    parser.add_argument("--nn_filename", type=str, help="path to save nearest neighbours")
    parser.add_argument("--unknown_phrase_key", type=str,
                        help="string corresponding to the unknown phrase embedding in the embedding file",
                        default="<unk>")
    parser.add_argument("--unknown_w1_key", type=str,
                        help="string corresponding to the unknown word 1 embedding in the embedding file",
                        default="<unk_w1>")
    parser.add_argument("--unknown_w2_key", type=str,
                        help="string corresponding to the unknown word 2 embedding in the embedding file",
                        default="<unk_w2>")
    parser.add_argument("--wsd_model", type=str,
                        choices=["identity_bilinear", "identity_matrix", "identity_bilinear_head", "identity_matrix_head", "both"],
                        help="which type of word sense disambuguation model should be used", default="identity_matrix")
    parser.add_argument("--glove", type=str, default="/Users/neelewitte/Desktop/progamming/glove/twe-lemmas.bin")
    parser.add_argument("--batch_size", type=int, help="how many instances should be contained in one batch?",
                        default=100)
    parser.add_argument("--patience", type=int, help="number of epochs to wait after the best model", default=5)
    parser.add_argument("--learning_rate", type=float, help="learning rate for optimization", default=0.04)
    parser.add_argument("--seed", type=int, help="number to which random seed is set", default=1)
    parser.add_argument("--save_path", type=str, help="file path to save the best model", default="/home/neele/programmieren/WSD_by_composition/models")
    parser.add_argument("--vocab_size", type=int, help="number of words to include in the fulllex model", default=24033,
                        choices=[8080, 24033, 16737])
    #24033, 8080
    parser.add_argument("--use_nvspace",action='store_true', default=False)
    parser.add_argument("--big_embedding_nn", type=str)
    parser.add_argument("--new_vspace_nn", type=str)
    parser.add_argument("--nonlinearity", type=str,
                        help="what kind of nonlinear function should be applied to the model. set to 'identity' if no nonlinearity should be applied",
                        default='tanh', choices=["tanh", "identity"])
    parser.add_argument("--dropout", type=float, help="dropout rate", default=0.7)

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

    wsd_models = {
        "identity_bilinear_head": Identity_Model_Head_bilinear(vocab_size=args.vocab_size, no_senses=2,
                                                          dropout_bilinear_forms=args.dropout, dropout_matrix=args.dropout),
        "identity_bilinear": Identity_Model_Mod_bilinear(vocab_size=args.vocab_size, no_senses=2,
                                                          dropout_bilinear_forms=args.dropout, dropout_matrix=args.dropout),
        "identity_matrix": Identity_Model(vocab_size=args.vocab_size, no_senses=2),
        "identity_matrix_head": Identity_Model_Head(vocab_size=args.vocab_size, no_senses=2),
        "both" : Identity_Model_bilineaer_both_positions(vocab_size=args.vocab_size, no_senses=2,
                                                          dropout_bilinear_forms=args.dropout, dropout_matrix=args.dropout)
    }
    # read in the wordembeddings
    gensim_model = Data.read_word_embeddings(args.embeddings)
    #glove_model = Data.read_word_embeddings(args.glove)
    #glove_model = Data.read_word_embeddings("/Users/neelewitte/Desktop/progamming/glove/twe-adj-n.bin")
    print("use new space")
    print(args.use_nvspace)
    print("vocab small")
    print(len(gensim_model.wv.vocab))
    print("vocab big")
    #print(len(glove_model.wv.vocab))
    if args.use_nvspace:
        sense_embedding_model = Data.read_word_embeddings(args.new_vspace)
    #sense_embedding_model = Data.read_word_embeddings("/Users/neelewitte/Desktop/progamming/sense_embeddings/adjN_matrix_head.txt")
    #sense_embedding_model = Data.read_word_embeddings("/Users/neelewitte/Desktop/progamming/sense_embeddings/adjN_matrix_head.txt")

    logger.info("Read embeddings from %s." % args.embeddings)

    # generate batches from data
    word2index = gensim_model.wv.vocab
    training_data = Data.generate_instances(args.batch_size, args.training_data, word2index,
                                            args.unknown_phrase_key, args.unknown_w1_key, args.unknown_w2_key)
    logger.info("%d training batches" % training_data.no_batches)
    validation_data = Data.generate_instances(args.batch_size, args.validation_data, word2index,
                                              args.unknown_phrase_key, args.unknown_w1_key, args.unknown_w2_key)
    logger.info("%d validation batches" % validation_data.no_batches)
    #test_data = Data.generate_instances(args.batch_size, args.test_data, word2index,
     #                                   args.unknown_phrase_key, args.unknown_w1_key, args.unknown_w2_key)
    #logger.info("%d test batches" % test_data.no_batches)
    lookup_table = gensim_model.wv.syn0
    logger.info("Batches have been generated using the data from %s" % args.data_dir)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    tf.set_random_seed(args.seed)

    saver = None
    both_positions = False
    if args.wsd_model == "both":
        both_positions = True

    wsdmodel = wsd_models[args.wsd_model]
    print("training starts")
    train_losses, validation_losses, best_loss, best_epoch, saver, sense_matrix = train(args, wsdmodel, training_data, validation_data, gensim_model, both_positions)
    logger.info("Training ended. Best epoch: %d, best loss: %.5f" % (best_epoch, best_loss))
    dev_predictions, dev_loss = predict(args, saver, args.model_path, validation_data, wsdmodel, lookup_table, both_positions)
    save_predictions(dev_predictions, validation_data,
                     str(Path(args.save_path).joinpath(args.save_name + "_dev_predictions.txt")))

    #sense_embeddings = "/Users/neelewitte/Desktop/progamming/sense_embeddings/adjN_bilinear_head.txt"
    print(args.use_nvspace)
    if args.use_nvspace:
        #print(Evaluation.get_nearest_neighbours_of_new_vspace(sense_matrix=sense_matrix, gensimmodel=gensim_model,
         #                                                     sensimmodel=glove_model, write_to_file=True,
          #                                                    threshold_distance=0.03, filename=args.big_embedding_nn))
        print(Evaluation.get_nearest_neighbours_of_new_vspace(sense_matrix=sense_matrix, gensimmodel=gensim_model,
                                                              sensimmodel=sense_embedding_model, write_to_file=True,
                                                              threshold_distance=0.03, filename=args.new_vspace_nn))

    else:
        #Evaluation.write_sensevectors_to_gensimfile(sense_matrix, args.new_vspace, gensim_model)
        #Evaluation.get_nearestn_of_original_vspace(sense_matrix=sense_matrix, gensimmodel=gensim_model,
         #                                          threshold_distance=0.2, filename=args.nn_filename,
          #                                         write_to_file=True)
        print("sense vectors have been written")

    print(Evaluation.get_nearestn_of_original_vspace(sense_matrix=sense_matrix, gensimmodel=gensim_model, threshold_distance=0.2, filename=args.nn_filename, write_to_file=False))
    do_eval(logger=logger, args=args, split="dev", loss=dev_loss, word_embeddings=gensim_model)
    logger.info("dev loss %.5f" % dev_loss)