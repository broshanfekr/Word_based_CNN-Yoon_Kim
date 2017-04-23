import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
import tensorflow as tf
from sklearn.cross_validation import train_test_split

############################### creating a parser for arguments###################################
import argparse
def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--input_dataset_path', '-input_dataset_path', type=str, default="yelp_academic_dataset_review.txt",
        metavar='PATH',
        help="Path of the dataset"
    )
    parser.add_argument(
        '--word2vec_model_path', '-word2vec_model_path', type=str, default="",
        metavar='PATH',
        help="the path of trained word2vec model."
    )
    parser.add_argument(
        '--is_separated', '-separate', type=bool, default=False,
        help="does the dataset have official train/test split or not?"
    )
    parser.add_argument(
        '--output_path', '-output_path', type=str, default="CNN_Sentiment_result.txt",
        metavar='PATH',
        help="output file that contain logs during train and test"
    )

    ############################# model hyper parameters #################################
    parser.add_argument(
        '--max_seq_len_cutoff', '-max_seq_len_cutoff', type=int, default=1500,
        help="the max len of a sentence."
    )
    parser.add_argument(
        '--learning_rate', '-learning_rate', type=float, default=0.001,
        help="learning rate parameter"
    )
    parser.add_argument(
        '--n_classes', '-n_classes', type=int, default=5,
        help="number of classes in classification problem"
    )
    parser.add_argument(
        '--batch_size', '-batch_size', type=int, default=100,
        help="size of the batch in training step"
    )
    parser.add_argument(
        '--num_epochs', '-num_epochs', type=int, default=50,
        help="defines the number of training epochs."
    )
    parser.add_argument(
        '--filter_sizes', '-filter_sizes', type=str, default="3,4,5",
        help="Comma-separated filter sizes (default: '3,4,5')"
    )
    parser.add_argument(
        '--num_filters', '-num_filters', type=int, default=150,
        help="Number of filters per filter size (default: 150)"
    )
    parser.add_argument(
        '--embedding_dim', '-embedding_dim', type=int, default=150,
        help="Dimensionality of word embedding"
    )

    parser.add_argument(
        '--evaluate_every', '-evaluate_every', type=int, default=3,
        help="Evaluate model on dev set after this many steps"
    )
    parser.add_argument(
        '--checkpoint_every', '-checkpoint_every', type=int, default=2,
        help="Save model after this many steps"
    )
    parser.add_argument(
        '--l2_reg_lambda', '-l2_reg_lambda', type=float, default=0.1,
        help="L2 regularizaion lambda"
    )

    parser.add_argument(
        '--dropout_keep_prob', '-dropout_keep_prob', type=float, default=0.5,
        help="Dropout keep probability (default: 0.5)"
    )
    parser.add_argument(
        '--checkpoint_dir', '-checkpoint_dir', type=str, default="runs/checkpoints",
        help="Checkpoint directory from training run"
    )
    parser.add_argument(
        '--allow_soft_placement', '-allow_soft_placement', type=bool, default=True,
        help="Allow device soft device placement"
    )
    parser.add_argument(
        '--log_device_placement', "-log_device_placement", type=bool, default=False,
        help="Log placement of ops on devices"
    )
    parser.add_argument(
        '--is_load_model', '-is_load_model', type=bool, default=False,
        help="do we want to load previes model?"
    )
    parser.add_argument(
        '--is_evaluation', '-is_evaluation', type=bool, default=False,
        help="do we want to use train set as evaluation?"
    )

    return parser

##################################################################################################
parser = create_parser()
FLAGS = parser.parse_args()
destfile = open(FLAGS.output_path, "w")

print("\nParameters:")
destfile.write("Parameters")
destfile.write("\n")
for arg in vars(FLAGS):
    print("{} : {}".format(arg, getattr(FLAGS, arg)))
    destfile.write("{} : {}".format(arg, getattr(FLAGS, arg)))
    destfile.write("\n\n")
print("")
##############################   Reading Data   ###################################################
def read_from_dataset(dataset_path, word2vec_model_path, n_classes, max_seq_len_cutoff):
    print("Loading data...")
    x, y, seq_max_len, vocabulary, vocabulary_inv, word2vec_vocab, word2vec_vec = data_helpers.load_data(
            dataset_path, word2vec_model_path, n_classes, max_seq_len_cutoff)

    myvocab_size = len(vocabulary)


    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.3, random_state=42)

    if(FLAGS.is_evaluation == True):
        x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        FLAGS.evaluate_every=1


    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    print("max_seq_len is: ", seq_max_len)

    return x_train, x_dev, y_train, y_dev, seq_max_len, vocabulary, vocabulary_inv, word2vec_vocab, word2vec_vec
#####################################################################################################################

def My_main():
    x_train, x_dev, y_train, y_dev, seq_max_len, vocabulary, vocabulary_inv, word2vec_vocab, word2vec_vec = read_from_dataset(
        FLAGS.input_dataset_path, FLAGS.word2vec_model_path, FLAGS.n_classes, FLAGS.max_seq_len_cutoff)


    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=seq_max_len,
                num_classes=FLAGS.n_classes,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                vocab_size=len(vocabulary),
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)


            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)

            # Initialize all variables
            sess.run(tf.initialize_all_variables())
            # Initialize all variables
            #sess.run(tf.global_variables_initializer())


            def train_step(x_batch, y_batch, epoch_num):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy, correct_predict, weigth_norm = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss,
                     cnn.accuracy, cnn.correct_predictions, cnn.weigth_norm],
                    feed_dict)

                print("epoch: {:g}, iteration: {:g}, weigth_norm: {:.6f},  loss: {:.4f}, acc: {:.4f}".format(epoch_num,
                                                                                                             step,
                                                                                                             weigth_norm,
                                                                                                             loss, accuracy))
                destfile.write("epoch: {:g}, iteration: {:g}, weigth_norm: {:.6f},  loss: {:.4f}, acc: {:.4f}\n".format(epoch_num,
                                                                                                             step,
                                                                                                             weigth_norm,
                                                                                                             loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, test_set = "dev"):
                """
                Evaluates model on a dev set
                """
                dev_baches = data_helpers.batch_iter(list(zip(x_batch, y_batch)),
                                                     batch_size=FLAGS.batch_size,
                                                     seq_length=seq_max_len,
                                                     emmbedding_size=FLAGS.embedding_dim,
                                                     word2vec_vocab=word2vec_vocab,
                                                     word2vec_vec=word2vec_vec,
                                                     is_shuffle=False)

                total_loss = 0
                total_acc = 0
                index = 0
                total_correct_predictions = 0
                total_dev_data = 0
                for batch in dev_baches:
                    if (len(batch[0]) == 0):
                        continue
                    x_batch, y_batch = zip(*batch[0])
                    total_dev_data += len(x_batch)

                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0
                    }

                    step, summaries, loss, accuracy, correct_predict = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.correct_predictions],
                        feed_dict)

                    print("on {}, test index: {:g}, Minibatch Loss: {:.6f}, acc: {:.5f}".format(test_set, index, loss, accuracy))
                    destfile.write(
                        "on {}, test index: {:g}, Minibatch Loss: {:.6f}, acc: {:.5f}\n".format(test_set, index, loss, accuracy))

                    total_loss += loss
                    total_acc += accuracy
                    index += 1
                    total_correct_predictions += np.sum(correct_predict)

                print("#################################################################\n")
                destfile.write("####################################################################\n\n")
                avg_loss = total_loss / (index)
                avg_acc = total_acc / (index)
                real_acc = (total_correct_predictions*1.0) / (total_dev_data)

                print("on {}, avarage_Loss: {:.6f}, avarage_acc: {:.5f}, real_acc: {:.5f}\n".format(test_set, avg_loss, avg_acc, real_acc))
                destfile.write(
                    "on {}, avarage_Loss: {:.6f}, avarage_acc: {:.5f}, real_acc: {:.5f}\n\n".format(test_set, avg_loss, avg_acc, real_acc))
                if(test_set == "dev"):
                    dev_summary_writer.add_summary(summaries, step)

                return avg_loss, real_acc



            train_loss_list = []
            train_acc_list = []
            test_loss_list = []
            test_acc_list = []

            for epoch in range(FLAGS.num_epochs):
                batches = data_helpers.batch_iter(list(zip(x_train, y_train)),
                                                  batch_size=FLAGS.batch_size,
                                                  seq_length=seq_max_len,
                                                  emmbedding_size=FLAGS.embedding_dim,
                                                  word2vec_vocab=word2vec_vocab,
                                                  word2vec_vec=word2vec_vec)
                # Training loop. For each batch...
                for batch in batches:
                    if (len(batch[0]) == 0):
                        continue
                    x_batch, y_batch = zip(*batch[0])
                    current_step = tf.train.global_step(sess, global_step)
                    train_step(x_batch, y_batch, epoch)

                if ((epoch+1) % FLAGS.checkpoint_every == 0):
                    path = saver.save(sess, checkpoint_prefix, global_step=(epoch+1))
                    print("Saved model checkpoint to {}\n".format(path))

                if ((epoch+1) % FLAGS.evaluate_every) == 0:
                    print("testing on dev set: ")
                    destfile.write("testing on dev set:\n")
                    avg_loss, real_acc = dev_step(x_dev, y_dev)

                    test_loss_list.append(avg_loss)
                    test_acc_list.append(real_acc)

                    if (FLAGS.is_evaluation == True):
                        print("###############################################")
                        destfile.write("###############################################\n")
                        print("testing on train set: ")
                        destfile.write("testing on train set: \n")
                        avg_loss, real_acc = dev_step(x_train, y_train, test_set="train")

                        train_loss_list.append(avg_loss)
                        train_acc_list.append(real_acc)

            path = saver.save(sess, checkpoint_prefix, global_step=FLAGS.num_epochs)
            print("Saved model checkpoint to {}\n".format(path))
            print("Optimization Finished!")
            print("\nEvaluation:")
            dev_step(x_dev, y_dev)
            print("")

            return train_acc_list, train_loss_list, test_acc_list, test_loss_list


if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    train_acc_list, train_loss_list, test_acc_list, test_loss_list = My_main()
    stop = timeit.default_timer()

    spent_time = int(stop - start)
    sec = spent_time % 60
    spent_time = int(spent_time / 60)
    minute = spent_time % 60
    spent_time = int(spent_time / 60)
    hours = spent_time
    print("h: ", hours, "  minutes: ", minute, "  secunds: ", sec)
    destfile.write("\nh: " + str(hours) + "  minutes: " + str(minute) + "  secunds: " + str(sec) + "\n")
    destfile.close()

    if FLAGS.is_evaluation:
        acc_validation = open("validation_acc.txt", "w")
        loss_validation = open("validation_loss.txt", "w")

        for i in range(len(train_acc_list)):
            acc_validation.write(str(train_acc_list[i]) + " " + str(test_acc_list[i]) + "\n")
            loss_validation.write(str(train_loss_list[i]) + " " + str(test_loss_list[i]) + "\n")

        acc_validation.close()
        loss_validation.close()