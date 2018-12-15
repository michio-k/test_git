
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold

# ロード
x_train, y_train, x_test = load_fashionmnist()


# 学習データと検証データに分割
#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)


# KFold
kf = KFold(n_splits=4, shuffle=True, random_state=0)
kf_train, kf_test = {}, {}
for i, (train_idx, test_idx) in enumerate(kf.split(x_train)):
    kf_train.update({i: train_idx})
    kf_test.update({i: test_idx})


# パラメータ初期化
W_fmnist = np.random.uniform(low=-0.08, high=0.08, size=(784, 10)).astype('float32')
b_fmnist = np.zeros(shape=(10,)).astype('float32')


# softmax関数
def softmax(x):
    x -= x.max(axis=1, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)    


# logの中身が0になるのを防ぐ
def np_log(x):
    return np.log(np.clip(a=x, a_min=1e-10, a_max=x))


# train関数
def train(x, t, eps=0.1):
    global W_fmnist, b_fmnist

    y = softmax(np.matmul(x, W_fmnist) + b_fmnist)

    cost = (- t * np_log(y)).sum(axis=1).mean()

    delta = y - t
    batch_size = x.shape[0]

    dW = np.matmul(x.T, delta) / batch_size
    db = np.matmul(np.ones(shape=(batch_size,)), delta) / batch_size

    W_fmnist -= eps * dW
    b_fmnist -= eps * db

    return cost


# valid関数
def valid(x, t):
    y = softmax(np.matmul(x, W_fmnist) + b_fmnist)
    cost = (- t * np_log(y)).sum(axis=1).mean()
    return cost, y


# train
for epoch in range(20):
    for (k, train_idx), (_, test_idx) in zip(kf_train.items(), kf_test.items()):
        for x, t in zip(x_train[train_idx, :], y_train[train_idx, :]):
            cost = train(x.reshape(1, -1), t.reshape(1, -1))

        cost, y_pred = valid(x_train[test_idx, :], y_train[test_idx, :])

        print('  ' + str(k+1) + ' fold : {:.3f}'.format(cost))
        print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
                epoch+1,
                cost,
                accuracy_score(y_train[test_idx, :].argmax(axis=1),
                                             y_pred.argmax(axis=1))))


# predict
y_pred = softmax(np.matmul(x_test, W_fmnist) + b_fmnist).argmax(axis=1)


# export
submission = pd.Series(y_pred, name='label')
submission.to_csv('submission_pred.csv', header=True, index_label='id')