import numpy as np
import cython
cimport numpy as np
from cython.parallel cimport prange, parallel

# np.ndarray[np.float64_t, ndim=1]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int compute_delta_one_step_smart(double[:] y_pred,
                           double[:] y_true, 
                           int[:] event):

    cdef Py_ssize_t i, j
    cdef float y_pred_sw_i, y_pred_sw_j
    cdef Py_ssize_t n = y_true.shape[0]
    cdef int delta = 0

    for i in prange(n, nogil=True, schedule='dynamic', num_threads=12):
        if (event[i]):
            for j in range(n):

                if (i == 0) or (j == n-1):

                    if(y_true[i] < y_true[j]):

                        y_pred_sw_i = y_pred[i]
                        y_pred_sw_j = y_pred[j]

                        if i == 0:
                            y_pred_sw_i = y_pred[-1]
                        if j == n-1:
                            y_pred_sw_j = y_pred[0]

                        if (y_pred[i] < y_pred[j]) and (y_pred_sw_i > y_pred_sw_j):
                            delta += 1
                        
                        elif (y_pred[i] > y_pred[j]) and (y_pred_sw_i < y_pred_sw_j):
                            delta += -1
    
    return abs(delta)


def compute_lambda(np.ndarray[np.float64_t, ndim=1] true_times, 
                   np.ndarray[np.int32_t, ndim=1] true_events, 
                   np.ndarray[np.float64_t, ndim=1] predicted_scores, 
                   good_ij_pairs, np.int32_t all_good_ij_pairs):

    cdef int i, j, num_evs
    cdef float z_ndcg, rho, rho_complement
    cdef np.ndarray[np.float64_t, ndim=1] lambdas
    cdef np.ndarray[np.float64_t, ndim=1] w

    num_evs = len(true_times) 
    lambdas = np.zeros(num_evs, dtype=np.float64)
    w = np.zeros(num_evs, dtype=np.float64)

    for i,j in good_ij_pairs:
  
        z_ndcg = compute_delta_one_step_smart(predicted_scores[i:j+1], true_times[i:j+1], true_events[i:j+1])/all_good_ij_pairs

        rho = 1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
        rho_complement = 1.0 - rho
        lambda_val = z_ndcg * rho
        lambdas[i] += lambda_val
        lambdas[j] -= lambda_val

        w_val = rho * rho_complement * z_ndcg
        w[i] += w_val
        w[j] += w_val

    return lambdas, w