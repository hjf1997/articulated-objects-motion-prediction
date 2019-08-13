# implemented by JunfengHu
# create time: 7/20/2019
import sys
import time
import numpy as np
import copy

def expmap2rotmat(A):
    theta = np.linalg.norm(A)
    if theta == 0:
        R = np.identity(3)
    else:
        A = A / theta
        cross_matrix = np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])
        R = np.identity(3) + np.sin(theta) * cross_matrix + (1 - np.cos(theta)) * np.matmul(cross_matrix, cross_matrix)

    return R

def rotmat2euler(R):
    if R[0, 2] == 1 or R[0, 2] == -1:
        E3 = 0
        dlta = np.arctan2(R[0, 1], R[0, 2])
        if R[0, 2] == -1:
            E2 = np.pi/2
            E1 = E3 + dlta
        else:
            E2 = -np.pi/2
            E1 = -E3 + dlta
    else:
        E2 = -np.arcsin(R[0, 2])
        E1 = np.arctan2(R[1, 2]/np.cos(E2), R[2, 2]/np.cos(E2))
        E3 = np.arctan2(R[0, 1]/np.cos(E2), R[0, 0]/np.cos(E2))

    eul = np.array([E1, E2, E3])

    return eul


def mean_euler_error(config, action, y_predict, y_test):
    # Convert from exponential map to Euler angles
    n_batch = y_predict.shape[0]
    nframes = y_predict.shape[1]

    mean_errors = np.zeros([n_batch, nframes])
    for i in range(n_batch):
        for j in range(nframes):
            if config.dataset == 'Human':
                pass
                #pred = unNormalizeData(y_predict[i], config.data_mean, config.data_std, config.dim_to_ignore)
                #gt = unNormalizeData(y_test[i], config.data_mean, config.data_std, config.dim_to_ignore)
            else:
                pred = copy.deepcopy(y_predict[i])
                gt = copy.deepcopy(y_test[i])
            for k in np.arange(3, pred.shape[1]-2, 3):
                pred[j, k:k + 3] = rotmat2euler(expmap2rotmat(pred[j, k:k + 3]))
                gt[j, k:k + 3] = rotmat2euler(expmap2rotmat(gt[j, k:k + 3]))
        pred[:, 0:6] = 0
        gt[:, 0:6] = 0

        idx_to_use = np.where(np.std(gt, 0) > 1e-4)[0]

        euc_error = np.power(gt[:, idx_to_use] - pred[:, idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt(euc_error)
        mean_errors[i, :] = euc_error

    mme = np.mean(mean_errors, 0)

    print("\n" + action)
    toprint_idx = np.array([1, 3, 7, 9, 13, 15, 17, 24])
    idx = np.where(toprint_idx < len(mme))[0]
    toprint_list = ["& {:.2f} ".format(mme[toprint_idx[i]]) for i in idx]
    print("".join(toprint_list))

    mme_mean = np.mean(mme[toprint_idx[idx]])
    return mme, mme_mean


def forward_kinematics(data, config, bone):

    nframes = data.shape[0]
    data = data.reshape([nframes, -1, 3])

    njoints = data.shape[1] + 1

    lie_params = np.zeros([nframes, njoints, 6])

    for i in range(njoints - 1):
        lie_params[:, i, 0:3] = data[:, i, :]

    lie_params[:, :, 3:6] = bone
    lie_params[:, 0, 3:6] = np.zeros([3])

    joint_xyz_f = np.zeros([nframes, njoints, 3])

    for i in range(nframes):
        joint_xyz_f[i, :, :] = computelie(np.squeeze(lie_params[i, :, :]))
    return joint_xyz_f


def computelie(lie_params):
    njoints = np.shape(lie_params)[0]
    A = np.zeros((njoints, 4, 4))

    for j in range(njoints):
        if j == 0:
            A[j, :, :] = lietomatrix(lie_params[j, 0: 3].T, lie_params[j, 3:6].T)
        else:
            A[j, :, :] = np.matmul(np.squeeze(A[j - 1, :, :]),
                                   lietomatrix(lie_params[j, 0:3].T, lie_params[j, 3:6].T))

    joint_xyz = np.zeros((njoints, 3))

    for j in range(njoints):
        coor = np.array([0, 0, 0, 1]).reshape((4, 1))
        xyz = np.matmul(np.squeeze(A[j, :, :]), coor)
        joint_xyz[j, :] = xyz[0:3, 0]

    return joint_xyz

def lietomatrix(angle, trans):
    R = expmap2rotmat(angle)
    T = trans
    SEmatrix = np.concatenate((np.concatenate((R, T.reshape(3, 1)), axis=1), np.array([[0, 0, 0, 1]])))

    return SEmatrix

def fk(data, config, bone):
    if config.dataset == 'Human':
        pass
        # xyz = []
        # for frame in range(config.test_output_window):
        #     xyz_new = forward_kinematics_h36m(data[frame])
        #     xyz.append(xyz_new)
        # xyz = np.array(xyz)
    else:
        xyz = forward_kinematics(data, config, bone)
    return xyz


class Progbar(object):
    """Progbar class copied from https://github.com/fchollet/keras/

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """
    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)