# This runs cross-validation for RNN. Note that the output of this file is only the saved models (and not the
# actual performance of the model on the test data). For evaluating how the saved models perform
# on test dataset, file 'sim/rnn_cv.py' should be
# used on the output of this file.

import sys
from multiprocessing.pool import Pool
from actionflow.data.data_process import DataProcess
from actionflow.rnn.lstm_beh import LSTMBeh
from actionflow.rnn.opt_beh import OptBEH
from actionflow.util.helper import cv_list, get_total_pionts
from actionflow.util.logger import LogFile, DLogger
from GL.data.data_reader import DataReader
from GL.util.paths import Paths
import tensorflow as tf

# cv_counts = 10
# cv_lists = {}
# data = DataReader.read_GL()
#
# ids=list(data.keys())
# cv_lists = cv_list(ids, cv_counts)
#
# configs = []
#
# for lr in [1e-2]:
#     for cells in [5, 10, 20, 30]:
#         for i in range(cv_counts):
#             configs.append({'lr': lr, 'cells': cells, 'cv_index': i})
#
#
# def run_GL_RNN(i):
#     tf.reset_default_graph()
#     ncells = configs[i]['cells']
#     learning_rate = configs[i]['lr']
#     cv_index = configs[i]['cv_index']
#
#     output_path = Paths.local_path + 'GL/rnn-cv/' + str(ncells) + 'cells/' + 'fold' + str(cv_index) + '/'
#     with LogFile(output_path, 'run.log'):
#         indx_data = cv_lists[cv_index]
#         train, test = DataProcess.train_test_between_subject(data, indx_data,
#                                                              [1, 2, 3, 4])
#
#         indx_data.to_csv(output_path + 'train_test.csv')
#         train_merged = DataProcess.merge_data(train)
#         DLogger.logger().debug("total points: " + str(get_total_pionts(train_merged)))
#         del train
#
#         worker = LSTMBeh(2, 2, n_cells=ncells)
#         OptBEH.optimise(worker, output_path, train_merged, None,
#                         learning_rate=learning_rate,
#                         global_iters=3000,
#                         load_model_path='../inits/rnn-init/' + str(ncells) + 'cells/model-final/'
#                         )
#
# def run_cv(f, n_proc):
#     p = Pool(n_proc)
#     p.map(f, range(len(configs)))
#     p.close() # no more tasks
#     p.join()  # wrap up current tasks


if __name__ == '__main__':
    DataReader.read_GL()
    # if len(sys.argv) == 2:
    #     n_proc = int(sys.argv[1])
    # elif len(sys.argv) == 1:
    #     n_proc = 1
    # else:
    #     raise Exception('invalid argument')
    #
    # run_cv(run_GL_RNN, n_proc)
