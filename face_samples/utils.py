import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import scipy.stats as stats
import pickle
import sklearn
from sklearn.decomposition import PCA

def gibbs_sampling(nu, cov, sample_size):
    """
    ギブスサンプリングを用いて与えられた共分散, 平均値を持つ
    多次元ガウス分布からのサンプリングを行う関数

    :param np.ndarray nu: 平均値
    :param np.ndarray cov: 共分散
    :param int sample_size: サンプリングする数
    :return:
    :rtype: np.ndarray
    """
    samples = []
    tmp = []
    n_dim = nu.shape[0]
    # start point of sampling
    start = nu
    samples.append(start)
    search_dim = 0

    for i in range(sample_size):
        if search_dim == n_dim - 1:
            """
            search dimension selection is cyclic.
            it can be replaced random choice.
            """
            search_dim = 0
        else:
            search_dim = search_dim + 1

        prev_sample = samples[-1][:]
        A = cov[search_dim][search_dim - 1] / float(cov[search_dim - 1][search_dim - 1])  # A*Σ_yy = Σ_xy
        _y = prev_sample[search_dim - 1]  # previous values of other dimension

        # p(x|y) ~ N(x|nu[x]+A(_y-nu[y]),Σ_zz)
        # Σ_zz = Σ_xx - A0*Σ_yx

        mean = nu[search_dim] + A * (_y - nu[search_dim - 1])
        sigma_zz = cov[search_dim][search_dim] - A * cov[search_dim - 1][search_dim]

        sample_x = np.random.normal(loc=mean, scale=np.power(sigma_zz, .5), size=1)
        prev_sample[search_dim] = sample_x[0]
        samples.append(np.array(prev_sample))

    return np.array(samples)

def get_image(group_name, n_samples):
    model = tf.compat.v2.saved_model.load('models/1024-synthesis')
    generate = model.signatures['synthesis']
    mean = np.load("distributions/params/" + group_name + "_mean.npy")
    cov = np.load("distributions/params/" + group_name + "_cov.npy")
    with open("distributions/pca/" + group_name + "_pca.pkl", "rb") as f:
        pca = pickle.load(f) #読み出し
    samples = gibbs_sampling(mean, cov, 10000)[2000:]
    sample_index = np.random.choice(samples.shape[0], n_samples, replace = False)
    dls = []
    for index in sample_index:
        sample_latent = pca.inverse_transform(samples[index])
        dls.append(sample_latent)
    dls = tf.convert_to_tensor(np.reshape(dls, (n_samples, 18, 512)), np.float32) # (18, 512)に整形
    outputs = generate(dlatents=dls)['images']
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.initializers.tables_initializer())
        img = Image.fromarray(np.concatenate(sess.run(outputs), axis=1)).resize((768, 256))
        buffer = io.BytesIO() # メモリ上への仮保管先を生成
        img.save(buffer, format="PNG") # pillowのImage.saveメソッドで仮保管先へ保存
        base64_img = base64.b64encode(buffer.getvalue()).decode().replace("'", "")
    return base64_img
