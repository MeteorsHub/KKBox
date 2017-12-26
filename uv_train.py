import tensorflow as tf
import numpy as np
from scipy.sparse import load_npz
from prepare_data import load_csv

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity("INFO")

tf.flags.DEFINE_integer("K", 100,
                        "how many features to learn, this should not be changed unless you want to retrain")
tf.flags.DEFINE_integer("iterations", 100000,
                        "how many iterations to train")
tf.flags.DEFINE_float("learning_rate", 0.01,
                      "learning rate")
tf.flags.DEFINE_string("strategy", "sgd",
                       "training strategy")
tf.flags.DEFINE_string("r_file", "./data/r.npz",
                       "path to R matrix file")
tf.flags.DEFINE_string("user_file", "./data/user_lookup.csv",
                       "path to user info file")
tf.flags.DEFINE_string("song_file", "./data/song_lookup.csv",
                       "path to song info file")
tf.flags.DEFINE_string("p_file", "./data/p-%d.npz" % FLAGS.iterations,
                       "path to trained P matrix")
tf.flags.DEFINE_string("q_file", "./data/q-%d.npz" % FLAGS.iterations,
                       "path to trained Q matrix")


def get_model(user_num, song_num):
    Q = tf.Variable(initial_value=tf.random_normal([user_num, FLAGS.K], dtype=tf.float16),
                    dtype=tf.float16,
                    name="Q_matrix")
    P = tf.Variable(initial_value=tf.random_normal([song_num, FLAGS.K], dtype=tf.float16),
                    dtype=tf.float16,
                    name="P_matrix")
    # prediction R
    R_ = tf.matmul(Q, P, transpose_b=True, name="R_predict")
    # real R
    R = load_npz(FLAGS.r_file).toarray().astype(np.float16)
    mask = R.astype(bool)
    objective = tf.boolean_mask(tf.pow(R - R_, 2), mask, name="masked_objective")
    loss = tf.reduce_sum(tf.reduce_sum(objective, axis=0), axis=0, name="loss")

    return Q, P, R_, loss


if __name__ == "__main__":
    _, user_data = load_csv(FLAGS.user_file)
    _, song_data = load_csv(FLAGS.song_file)

    Q, P, R_, loss = get_model(len(user_data), len(song_data))
    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(FLAGS.iterations):
        _, l, q, p = sess.run([train_op, loss, Q, P])
        if (i+1) % 100 == 0:
            tf.logging.info("loss: %0.2f finish training step %s in %s" % (l, (i+1), FLAGS.iterations))
    tf.logging.info("saving P and Q matrix...")
    p.savez(FLAGS.p_file)
    q.savez(FLAGS.q_file)

